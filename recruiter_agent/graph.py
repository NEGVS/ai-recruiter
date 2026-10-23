"""LangGraph 编排图 —— 项目的骨架。

完整链路：

    START
      ↓
    [intent] 意图识别 + 实体抽取
      ↓ (置信度低？)
    [clarify] ──────────────────────────────→ END
      ↓
    [plan] 生成工具计划
      ↓
    [tool] ──(还有待办工具？)──→ [tool]  ⟲
      ↓
    [knowledge]  ← policy_qa 走知识库
      ↓
    [retrieve]  ← dense + BM25 + 画像 三路召回
      ↓
    [rerank]    ← 业务感知精排
      ↓
    [score]     ← 多维可解释打分
      ↓
    [shortlist] ← 写入清单（副作用节点）
      ↓
    [answer]    ← 基于事实底稿生成
      ↓
    [reflect]   ← 5 项硬检查 + LLM 批判
      ↓ (decision=revise 且未超预算？)
    ├──→ [answer]        仅文字润色问题 → 重新生成
    ├──→ [retrieve]      召回不足 → 放宽约束重召
    └──→ [tool]          信息缺失 → 补充工具调用 ⟲
      ↓ (accept)
    END

关键点：Reflection 的路由是**定向**的 —— 不同类型的问题回流到不同节点，
这正是 LangGraph 相比「一个大 while 循环」的真正价值。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .config import Settings, get_settings
from .nodes.answer_node import answer_node
from .nodes.intent_node import clarify_node, intent_node
from .nodes.reflect_node import reflect_node
from .nodes.retrieval_node import knowledge_node, rerank_node, retrieve_node, score_node
from .nodes.tool_node import plan_node, pending_tool, tool_node
from .state import RecruitState, new_state

log = logging.getLogger(__name__)


# ================================================================= 条件边
def route_after_intent(state: RecruitState) -> Literal["clarify", "plan"]:
    """意图置信度过低 → 反问澄清，而不是硬猜。"""
    if state.get("clarified") or state.get("intent") == "chitchat":
        # chitchat 也走 plan（plan 为空 → 直接 answer）
        return "clarify" if state.get("clarified") else "plan"
    if state.get("intent_confidence", 0) < 0.45:
        log.info("意图置信度 %.2f 偏低，触发澄清", state.get("intent_confidence", 0))
        return "clarify"
    return "plan"


def route_after_tool(state: RecruitState) -> Literal["tool", "knowledge", "retrieve", "score", "answer"]:
    """工具链：执行完所有工具后，按计划进入下一阶段。"""
    # 必填参数缺失 → 直接去 answer 节点输出追问（该节点会短路后续反思）
    if state.get("pending_input"):
        return "answer"

    if pending_tool(state):
        return "tool"
    plan = state.get("plan") or []
    steps = [p.get("step") for p in plan]
    if "knowledge_retrieval" in steps:
        return "knowledge"
    # job_recommend 的评分对象是「岗位」，候选池来自 list_jobs 工具而非向量召回，
    # 直接进 score 节点，跳过候选人检索/精排（否则会白跑一遍候选人召回）。
    if state.get("intent") == "job_recommend":
        return "score"
    if "retrieval" in steps or state.get("intent") in (
        "candidate_search", "candidate_evaluate"
    ):
        return "retrieve"
    return "answer"


def route_after_retrieve(state: RecruitState) -> Literal["rerank", "answer"]:
    chunks = state.get("retrieved_chunks") or []
    if not chunks:
        log.info("检索结果为空，跳过 rerank 直达回答")
        return "answer"
    # 制度问答已在 knowledge 节点完成精排，不应再进入候选人打分链路
    if state.get("intent") == "policy_qa":
        return "answer"
    return "rerank"


def route_after_rerank(state: RecruitState) -> Literal["score", "answer"]:
    if not state.get("reranked_chunks"):
        return "answer"
    return "score"


def route_after_reflect(state: RecruitState) -> Literal["answer", "retrieve", "rerank", "tool", "knowledge", "end"]:
    """Reflection 的定向回流。"""
    from .config import get_settings

    s = get_settings()
    ref = state.get("reflection") or {}
    decision = ref.get("decision", "accept")

    if decision != "revise":
        return "end"
    if state.get("revision_count", 0) > s.max_reflection_rounds:
        log.info("反思轮次已达上限 %d，接受当前答案", s.max_reflection_rounds)
        return "end"

    # 制度问答的缺失只能靠补充知识库检索解决，绝不能回流到候选人链路
    if state.get("intent") == "policy_qa":
        return "knowledge"

    # job_recommend 不走候选人召回链路，修订只可能是「重写答案」；
    # 回流到 retrieve/rerank 会清空岗位评分（score 节点对 job_recommend 自成一路）。
    if state.get("intent") == "job_recommend":
        return "answer"

    actions = " ".join(ref.get("next_actions") or []) + " ".join(ref.get("issues") or [])

    # 优先级：补工具 > 重检索 > 重写
    # 注意必须校验「确实还有待办工具」，否则会在无工具可调时形成
    # reflect -> tool -> answer -> reflect 的死循环
    if any(k in actions for k in ("补充工具", "薪酬数据", "补充调用")) and pending_tool(state):
        return "tool"
    if any(k in actions for k in ("放宽", "重新召回", "召回")):
        # 只有「重召回真能带来新结果」时才回流，否则是空转。
        # 早前写的是 `if state.get("retrieved_chunks") or True:` —— 恒真，
        # 导致每轮 revise 都被推回 retrieve，空库时连跑 3 轮白工。
        if _relaxation_can_help(state):
            return "retrieve"
    if state.get("retrieved_chunks") and not state.get("reranked_chunks"):
        return "rerank"
    # 能重写答案才值得回 answer；否则再写一遍也没有新证据，直接收口。
    # 触发条件：没有任何评分结果或制度证据（无证据不可改善），
    # 且 Reflection 提不出实质性动作（如「移除/补全」），说明只能空转改写。
    no_evidence = not state.get("scores") and not state.get("knowledge_chunks")
    actionable = any(k in actions for k in ("重新组织", "移除", "补全", "重写"))
    if no_evidence and not actionable:
        log.info("无新证据可补充，接受当前答案而非循环改写")
        return "end"
    return "answer"


def _relaxation_can_help(state: RecruitState) -> bool:
    """重召回是否可能改善结果。库空 + 已放宽过 → 再试也没用，直接收口。"""
    entities = state.get("entities") or {}
    # 约束已放宽过且仍 0 结果 → 继续 retrieve 是空转
    if entities.get("_relaxed"):
        return False
    # 库里本来就空（未索引/后端未就绪）→ 放宽条件也救不回来
    stats = state.get("retrieval_stats") or {}
    if (stats.get("dense", 0) + stats.get("bm25", 0) + stats.get("profile", 0)) == 0:
        return False
    return True


def route_after_answer(state: RecruitState) -> Literal["reflect", "end"]:
    """闲聊与「向用户追问」无需自我反思；面试确认也直接结束。"""
    if state.get("intent") in ("chitchat", "confirm_interview"):
        return "end"
    if state.get("pending_input") or state.get("pending_interview"):
        return "end"
    return "reflect"


# ================================================================= 图构建
def build_graph(s: Settings | None = None) -> CompiledStateGraph:
    s = s or get_settings()
    g = StateGraph(RecruitState)

    g.add_node("intent", intent_node)
    g.add_node("clarify", clarify_node)
    g.add_node("plan", plan_node)
    g.add_node("tool", tool_node)
    g.add_node("knowledge", knowledge_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("score", score_node)
    g.add_node("answer", answer_node)
    g.add_node("reflect", reflect_node)

    g.add_edge(START, "intent")
    g.add_conditional_edges("intent", route_after_intent,
                            {"clarify": "clarify", "plan": "plan"})
    g.add_edge("clarify", END)
    g.add_edge("plan", "tool")
    g.add_conditional_edges("tool", route_after_tool,
                            {"tool": "tool", "knowledge": "knowledge", "retrieve": "retrieve",
                             "score": "score", "answer": "answer"})
    g.add_edge("knowledge", "answer")
    g.add_conditional_edges("retrieve", route_after_retrieve, {"rerank": "rerank", "answer": "answer"})
    g.add_conditional_edges("rerank", route_after_rerank, {"score": "score", "answer": "answer"})
    g.add_edge("score", "answer")
    g.add_conditional_edges("answer", route_after_answer, {"reflect": "reflect", "end": END})
    g.add_conditional_edges("reflect", route_after_reflect,
                            {"answer": "answer", "retrieve": "retrieve", "rerank": "rerank",
                             "tool": "tool", "knowledge": "knowledge", "end": END})

    return g.compile()


_compiled: CompiledStateGraph | None = None


def get_graph() -> CompiledStateGraph:
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


# ================================================================= 运行时封装
class RecruitAgent:
    """对外的 Agent 门面。"""

    def __init__(self, s: Settings | None = None) -> None:
        self.s = s or get_settings()
        self.graph = get_graph()
        # 跨轮会话缓存：{session_id: {candidate_name, pending_interview, ...}}
        # 用于桥接「给张伟排期 → 回复确认第2个」这类两轮交互。
        # 无持久化（重启丢失），但对 CLI/API 单进程已够用。
        self._session: dict[str, dict[str, Any]] = {}

    def run(self, query: str, session_id: str = "cli") -> dict[str, Any]:
        init = new_state(query, session_id)

        # 如果本轮是「确认面试时段」，从缓存中恢复上一轮的候选人信息
        ctx = self._session.get(session_id, {})
        if ctx:
            ents = dict(init.get("entities") or {})
            if not ents.get("candidate_name") and ctx.get("candidate_name"):
                ents["candidate_name"] = ctx["candidate_name"]
            if not ents.get("candidate_id") and ctx.get("candidate_id"):
                ents["candidate_id"] = ctx["candidate_id"]
            # 把上一轮的时段列表注入实体，工具层按序号选 slot
            if ctx.get("pending_slots"):
                ents["_pending_slots"] = ctx["pending_slots"]
            init["entities"] = ents

        t0 = time.time()
        final = self.graph.invoke(init, config={
            "recursion_limit": 40,
            "tags": [f"session:{session_id}"],
        })
        total_ms = int((time.time() - t0) * 1000)

        # 缓存本轮产出的会话上下文，供下一轮使用
        session_update: dict[str, Any] = {}
        pending_iv = final.get("pending_interview")
        if pending_iv:
            session_update["candidate_name"] = pending_iv.get("candidate")
            session_update["pending_slots"] = pending_iv.get("candidates") or []
        tool_results = final.get("tool_results") or []
        for t in tool_results:
            if t.get("tool") == "get_candidate" and t.get("ok"):
                p = t.get("_payload") or {}
                c = p.get("candidate") or {}
                if c.get("name"):
                    session_update["candidate_name"] = c["name"]
                if c.get("candidate_id"):
                    session_update["candidate_id"] = c["candidate_id"]
        if session_update:
            self._session[session_id] = {**ctx, **session_update}
        elif not pending_iv:
            # 面试已确认 → 清空缓存，避免下次串台
            self._session.pop(session_id, None)

        return {
            "query": query,
            "answer": final.get("final_answer", ""),
            "intent": final.get("intent"),
            "intent_confidence": final.get("intent_confidence"),
            "entities": _safe(final.get("entities")),
            "plan": _safe(final.get("plan")),
            "tool_results": _safe(final.get("tool_results")),
            "retrieval_stats": _safe(final.get("retrieval_stats")),
            "shortlist": _safe(final.get("scores")),
            "score_meta": _safe(final.get("score_meta")),
            "reranked": [
                {"id": c["id"], "name": c.get("metadata", {}).get("name"),
                 "title": c.get("metadata", {}).get("title"),
                 "rerank_score": c.get("rerank_score"),
                 "evidence": c.get("evidence", [])}
                for c in (final.get("reranked_chunks") or [])
            ],
            "reflection": _safe(final.get("reflection")),
            "reflection_history": _safe(final.get("reflection_history")),
            "answer_meta": _safe(final.get("answer_meta")),
            "trace": final.get("trace") or [],
            "total_ms": total_ms,
            "errors": final.get("errors") or [],
        }


def _safe(v: Any) -> Any:
    try:
        json.dumps(v, ensure_ascii=False, default=str)
        return v
    except Exception:  # noqa: BLE001
        return str(v)


_agent: RecruitAgent | None = None


def get_agent() -> RecruitAgent:
    global _agent
    if _agent is None:
        _agent = RecruitAgent()
    return _agent

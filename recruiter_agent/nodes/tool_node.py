"""节点 2：Planner + 节点 3：Tool 执行。

Planner 把「意图 + 实体」翻译成一条**工具调用计划**（ReAct 风格的显式计划，
而非让模型自由发挥），然后 tool_node 逐步执行。

为什么不让 LLM 直接决定调哪个工具？
  - 招聘工具涉及「写库」动作（排期、写清单），必须可控可审计；
  - 规则化的计划让 Token 成本从 O(轮次) 降到 O(1)，且行为可回归测试；
  - Reflection 发现信息缺失时，可以**精确地**把缺失项映射为补充工具调用。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..llm import get_llm
from ..state import RecruitState, ToolResult
from ..tools import all_tools, get_tool
from .common import timed

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- 计划规则
PLANS: dict[str, list[dict[str, Any]]] = {
    "candidate_search": [
        {"step": "query_candidates", "reason": "结构化条件预筛，确定候选池与总量"},
        {"step": "retrieval", "reason": "语义+BM25 召回并精排"},
        {"step": "scoring", "reason": "多维打分排序"},
    ],
    "candidate_evaluate": [
        {"step": "get_candidate", "reason": "获取候选人完整档案"},
        {"step": "get_job", "reason": "获取目标岗位要求"},
        {"step": "retrieval", "reason": "补充相关项目证据"},
        {"step": "scoring", "reason": "多维匹配度打分"},
    ],
    "job_recommend": [
        {"step": "get_candidate", "reason": "获取候选人档案"},
        {"step": "list_jobs", "reason": "拉取在招岗位"},
        {"step": "retrieval", "reason": "语义匹配"},
        {"step": "scoring", "reason": "候选人对岗位打分"},
    ],
    "candidate_detail": [
        {"step": "get_candidate", "reason": "读取候选人完整档案与进展"},
    ],
    "job_detail": [
        {"step": "get_job", "reason": "读取岗位详情与招聘进度"},
    ],
    "interview_list": [
        {"step": "list_interviews", "reason": "查询面试日程"},
    ],
    "interview_schedule": [
        {"step": "get_candidate", "reason": "确认候选人身份"},
        {"step": "schedule_interview", "reason": "检查档期冲突并生成候选时段"},
    ],
    "offer_negotiation": [
        {"step": "get_candidate", "reason": "获取期望薪资"},
        {"step": "calc_offer", "reason": "计算总包与覆盖率"},
    ],
    "market_insight": [
        {"step": "market_insight", "reason": "人才市场供需与薪资分位分析"},
    ],
    "policy_qa": [
        {"step": "knowledge_retrieval", "reason": "检索制度文档"},
    ],
    "chitchat": [
        {"step": "none", "reason": "闲聊直接回答"},
    ],
    "confirm_interview": [
        {"step": "schedule_interview", "reason": "写入确认时段"},
    ],
}


@timed("plan")
def plan_node(state: RecruitState) -> dict:
    intent = state.get("intent", "chitchat")
    entities = dict(state.get("entities") or {})

    steps = [dict(s) for s in PLANS.get(intent, PLANS["chitchat"])]

    # 动态修正：仅在岗位名「像岗位名」时才去查 JD。
    # 「有哪些后端」这类疑问句成分不是岗位名，强行调用 get_job 只会产生无谓失败。
    if intent == "candidate_search" and _looks_like_job(entities.get("job_title")):
        steps.insert(0, {"step": "get_job", "reason": f"解析岗位「{entities['job_title']}」的具体要求"})

    # 求职者自助模式：用户没有报候选人姓名，档案不在库里，跳过 get_candidate，
    # 否则会卡在「请告诉我要推荐给哪位候选人」——而用户问的正是「我适合什么岗位」。
    if intent == "job_recommend" and not entities.get("candidate_name") \
            and not entities.get("candidate_id"):
        steps = [p for p in steps if p.get("step") != "get_candidate"]

    # 工具参数绑定
    for s in steps:
        s["args"] = _bind_args(s["step"], intent, entities, state.get("user_query", ""))

    return {"plan": steps, "entities": entities}


_JOB_LIKE_SUFFIX = ("工程师", "专家", "架构师", "经理", "主管", "负责人", "师", "岗", "职位")
_JOB_QUESTION_WORDS = ("有哪些", "有什么", "哪些", "多少", "几个", "怎么", "如何", "是什么")


def _looks_like_job(title: str | None) -> bool:
    """粗判一个字符串是否像岗位名（避免拿疑问句成分去查 JD）。"""
    if not title or len(title) < 2:
        return False
    if any(q in title for q in _JOB_QUESTION_WORDS):
        return False
    if any(title.endswith(sfx) or sfx in title for sfx in _JOB_LIKE_SUFFIX):
        return True
    # 无后缀时，若已在实体中作为技能/方向词出现，也可视为岗位方向
    return len(title) <= 8


def _bind_args(step: str, intent: str, e: dict, query: str = "") -> dict[str, Any]:
    args: dict[str, Any] = {}
    if step == "query_candidates":
        if e.get("cities"):
            args["city"] = ",".join(e["cities"])
        if e.get("min_years") is not None:
            args["min_years"] = e["min_years"]
        if e.get("max_years") is not None:
            args["max_years"] = e["max_years"]
        if e.get("education"):
            args["education"] = e["education"]
        if e.get("max_salary") is not None:
            args["max_salary"] = e["max_salary"] // 1000
        if e.get("job_intent"):
            args["job_intent"] = e["job_intent"]
        if e.get("skills"):
            args["skill"] = ",".join(e["skills"][:4])
        args["limit"] = max(20, (e.get("top_n") or 5) * 4)
    elif step == "get_candidate":
        key = e.get("candidate_name") or e.get("candidate_id")
        args["key"] = key or ""
    elif step == "get_job":
        args["key"] = e.get("job_title") or e.get("job_id") or ""
    elif step == "list_interviews":
        # 「下周 / 本周 / 明天」→ 对应日期区间；不指定则返回全部（按日期排序）
        from datetime import date, timedelta
        today = date.today()
        wd = today.weekday()
        if re.search(r"下周", e.get("_raw_query", "")):
            args["date_from"] = (today + timedelta(days=7 - wd)).isoformat()
            args["date_to"] = (today + timedelta(days=13 - wd)).isoformat()
        elif re.search(r"这周|本周", e.get("_raw_query", "")):
            args["date_from"] = today.isoformat()
            args["date_to"] = (today + timedelta(days=6 - wd)).isoformat()
        elif re.search(r"明天", e.get("_raw_query", "")):
            d = (today + timedelta(days=1)).isoformat()
            args["date_from"] = args["date_to"] = d
    elif step == "list_jobs":
        if e.get("cities"):
            args["city"] = e["cities"][0]
    elif step == "schedule_interview":
        args["candidate"] = e.get("candidate_name") or e.get("candidate_id") or ""
        if e.get("date"):
            args["date"] = e["date"]
        # confirm_interview 意图：直接带 confirm=yes 和时段序号
        if e.get("_confirm_slot") is not None:
            args["confirm"] = "yes"
            args["_slot_index"] = int(e["_confirm_slot"])
    elif step == "calc_offer":
        # expected = 用户口述的目标年薪（「总包 60 万」→ 600000）。
        # max_salary 是用户给的**薪资上限**，与「offer 目标」不是一回事；
        # 早前用它兜底会把上限当成目标，算出语义错误的覆盖率。
        target = e.get("expected_salary") or 0
        if target:
            args["expected"] = int(target)
        # 候选人档案里的期望是月薪，已在 _enrich_entities 折算成 _candidate_expected（年薪）
        if e.get("_candidate_expected"):
            args["candidate_expected"] = e["_candidate_expected"]
        if e.get("_candidate_expected_monthly"):
            args["_candidate_expected_monthly"] = e["_candidate_expected_monthly"]
    elif step == "market_insight":
        topic = e.get("_market_topic") or " ".join(e.get("skills", [])) or e.get("job_title") or "技术"
        args["topic"] = topic
        if e.get("cities"):
            args["city"] = e["cities"][0]
        args["compare"] = "yes"
    return args


# ---------------------------------------------------------------- Tool 执行
@timed("tool")
def tool_node(state: RecruitState) -> dict:
    """执行当前待办工具。返回 updates，同时把执行结果累积进 tool_results。"""
    from ..config import get_settings

    s = get_settings()
    plan: list[dict] = list(state.get("plan") or [])
    entities: dict = dict(state.get("entities") or {})

    # 找第一个非 retrieval / scoring / knowledge / none 的工具步骤
    step = next((p for p in plan if p.get("step") in all_tools() and not p.get("done")), None)
    if not step:
        return {}

    name = step["step"]
    tool = get_tool(name)
    if not tool:
        return {"errors": [f"未注册工具: {name}"]}

    args = _merge_reflection_args(step.get("args") or {}, state)

    # 计划在 plan 阶段绑定参数，但前序工具的产出可能补充了新实体
    # （例如 get_candidate 拿到期望薪资后，calc_offer 需要用到），
    # 因此这里用「最新 entities」再绑定一次补齐缺省参数。
    for k, v in _bind_args(name, state.get("intent", ""), entities,
                           state.get("user_query", "")).items():
        if v not in (None, "", [], {}):
            args.setdefault(k, v)
    if entities.get("_candidate_expected"):
        args.setdefault("candidate_expected", entities["_candidate_expected"])

    # 必填参数缺失 → 不执行，记为 skipped 并请求用户补充。
    # 关键：无论成功失败都必须把该步骤标记为 done，否则 plan → tool → tool 会死循环。
    missing = [k for k in tool.required if args.get(k) in (None, "", [])]
    if missing:
        result = ToolResult(name=name, ok=False, error=f"缺少必填参数: {missing}")
    else:
        result = tool.safe_run(**args)

    record = {
        "tool": name, "ok": result.ok, "error": result.error, "cost_ms": result.cost_ms,
        "args": args, "observations": result.observations,
        "payload_keys": list(result.payload.keys()),
        "summary": _payload_summary(name, result),
        # 完整 payload 保留在状态中，供 answer / reflect 节点做事实渲染，
        # 避免答案层再查一次数据库，也保证「答案 = 工具事实」的一致性
        "_payload": result.payload,
    }
    tool_results = list(state.get("tool_results") or [])
    tool_results.append(record)

    entities = _enrich_entities(entities, name, result)

    # 标记该步骤已完成（含失败/跳过）
    plan = [{**p, "done": True, "skipped": True} if p is step else p for p in plan]

    updates: dict[str, Any] = {
        "plan": plan,
        "tool_results": tool_results,
        "entities": entities,
        "tool_call_count": state.get("tool_call_count", 0) + 1,
    }
    if not result.ok:
        updates["errors"] = list(state.get("errors") or []) + [f"tool {name}: {result.error}"]
        if missing:
            updates["pending_input"] = {
                "tool": name,
                "missing": missing,
                "question": _missing_param_question(name, missing, entities),
            }

    # 排期类工具：若返回候选时段且未确认，直接给出交互式回答
    if name == "schedule_interview" and result.ok and not result.payload.get("confirmed"):
        updates["pending_interview"] = result.payload
    if name == "schedule_interview" and result.ok and result.payload.get("confirmed"):
        updates["shortlist"] = [{"candidate": result.payload.get("candidate"),
                                 "interview": {k: result.payload.get(k)
                                               for k in ("date", "hour", "interviewer", "round")}}]

    return updates


# 必填参数缺失时，向用户要什么
_MISSING_QUESTIONS = {
    "get_candidate": "请告诉我要分析哪位候选人（姓名即可，例如「张伟」）。",
    "get_job": "请告诉我要对比的岗位名称或岗位编号（例如「AI平台工程师」或「JD-1001」）。",
    "schedule_interview": "请告诉我要为哪位候选人安排面试。",
    "calc_offer": "请提供候选人的期望薪资（月薪或总包均可）。",
    "market_insight": "请说明要分析哪个方向（如 LLM、后端、算法）。",
}


def _missing_param_question(tool: str, missing: list[str], entities: dict) -> str:
    base = _MISSING_QUESTIONS.get(tool, f"请补充 {tool} 所需的信息。")
    known = []
    if entities.get("cities"):
        known.append(f"城市={'、'.join(entities['cities'])}")
    if entities.get("skills"):
        known.append(f"技能={'、'.join(entities['skills'][:5])}")
    if known:
        base += f"\n（已识别：{'；'.join(known)}）"
    return base


def _merge_reflection_args(args: dict, state: RecruitState) -> dict:
    """Reflection 提出的补充动作（missing_aspects）映射为额外工具调用参数。"""
    out = dict(args)
    ref = state.get("reflection") or {}
    if not ref:
        return out
    for aspect in ref.get("missing_aspects", []) or []:
        a = str(aspect)
        if "薪酬" in a or "薪资" in a:
            out.setdefault("with_salary", "yes")
        if "放宽" in a:
            out["relax"] = "yes"
    return out


def _enrich_entities(e: dict, tool: str, r: ToolResult) -> dict:
    """把工具产出并入 entities。"""
    if tool == "query_candidates" and r.ok:
        e["_candidate_pool"] = r.payload.get("count", 0)
        e["_pool_total"] = r.payload.get("total_matched", 0)
    elif tool == "get_candidate" and r.ok:
        c = r.payload["candidate"]
        e["candidate_id"] = c["candidate_id"]
        e.setdefault("candidate_name", c["name"])
        if c.get("expected_salary"):
            # 档案里的 expected_salary 是**月薪**；calc_offer 的口径是年薪。
            # 只记到 _candidate_expected*，不要把月薪值写回 expected_salary ——
            # 早前这么做让「给张伟测算 offer」拿到 65000 当年薪，算出总包 0 万。
            e["_candidate_expected"] = int(c["expected_salary"]) * 12
            e["_candidate_expected_monthly"] = int(c["expected_salary"])
        e["_candidate_city"] = c.get("city")
        e["_candidate_years"] = c.get("years")
        e.setdefault("skills", c.get("skills", [])[:8])
    elif tool == "get_job" and r.ok:
        j = r.payload["job"]
        e["job_id"] = j["job_id"]
        e["job_title"] = j["title"]
        e["_job"] = j
        # 岗位的城市/年限/学历是**岗位要求**，不是候选人的准入约束。
        # 早前把岗位城市塞进 entities["cities"]，导致评估北京的候选人时
        # 检索被上海这个 filter 限死，本人根本没进候选池。
        if not e.get("candidate_name"):
            e.setdefault("cities", [j["city"]] if j["city"] else [])
            # 用户显式给过年限就**以用户为准**：JD 的带宽只用于评分（_job），
            # 不能反过来覆盖用户的硬条件，否则「不考虑 5 年以上」会被补成 5-9 年。
            if e.get("min_years") is None and e.get("max_years") is None:
                e.setdefault("min_years", j["min_years"])
                e.setdefault("max_years", j["max_years"])
            e.setdefault("education", j.get("education_req"))
        e.setdefault("skills", j.get("hard_skills", []))
    elif tool == "list_jobs" and r.ok:
        e["_jobs"] = r.payload.get("jobs", [])
    return e


def _payload_summary(tool: str, r: ToolResult) -> str:
    if not r.ok:
        return f"失败: {r.error}"
    p = r.payload
    if tool == "query_candidates":
        return f"命中 {p.get('total_matched')} 人，返回 {p.get('count')} 条"
    if tool == "get_candidate":
        c = p["candidate"]
        return f"{c['name']}｜{c['title']}｜{c['years']}年｜{c['city']}｜期望 {c['expected_salary']//1000}K"
    if tool == "get_job":
        j = p["job"]
        return f"{j['title']}｜{j['level']}｜{j['min_years']}-{j['max_years']}年｜{j['salary_min']//1000}-{j['salary_max']//1000}K"
    if tool == "market_insight":
        return (f"{p['topic']}: 命中 {p['matched']}/{p['pool']} 人，薪资P50 {p['salary_p50']//1000}K，"
                f"供给指数 {p['supply_index']}")
    if tool == "schedule_interview":
        if p.get("confirmed"):
            return f"已排期 {p.get('date')} {p.get('hour')}:00 ({p.get('interviewer')})"
        return f"待确认时段 {len(p.get('candidates', []))} 个"
    if tool == "calc_offer":
        return f"总包 {p['total_package']//10000}万，覆盖 {p['coverage']*100:.0f}%"
    if tool == "list_jobs":
        return f"在招 {p.get('count')} 个岗位"
    if tool == "list_interviews":
        return f"{p.get('count')} 条面试"
    if tool == "manage_shortlist":
        return f"清单 {p.get('count', 0)} 人"
    return "完成"


# ---------------------------------------------------------------- 路由辅助
def pending_tool(state: RecruitState) -> str | None:
    """返回下一个待执行的工具名；没有则返回 None。"""
    for p in state.get("plan", []):
        if p.get("step") in all_tools() and not p.get("done"):
            return p["step"]
    return None


def has_pending_step(state: RecruitState, *steps: str) -> bool:
    return any(p.get("step") in steps and not p.get("done") for p in state.get("plan", []))

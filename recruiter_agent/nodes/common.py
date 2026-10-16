"""节点公共工具：计时、Trace 记录、异常兜底。

每个节点都通过 `@timed` 装饰，节点内部只管业务逻辑，
耗时统计与 trace 追加由装饰器统一完成，保证 Trace 的完整性。
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

log = logging.getLogger(__name__)


def timed(name: str | None = None):
    """节点计时 + Trace 记录装饰器。"""

    def deco(fn: Callable) -> Callable:
        node_name = name or fn.__name__.replace("_node", "")

        @functools.wraps(fn)
        def wrapper(state: dict) -> dict:
            t0 = time.time()
            try:
                out = fn(state) or {}
            except Exception as e:  # noqa: BLE001
                ms = int((time.time() - t0) * 1000)
                log.exception("节点 %s 执行失败", node_name)
                trace = list(state.get("trace", []))
                trace.append({"node": node_name, "ms": ms, "status": "error",
                              "summary": f"{type(e).__name__}: {e}"})
                return {
                    "trace": trace,
                    "errors": list(state.get("errors", [])) + [f"[{node_name}] {type(e).__name__}: {e}"],
                    "iteration": state.get("iteration", 0) + 1,
                }

            ms = int((time.time() - t0) * 1000)
            trace = list(state.get("trace", []))
            summary, payload = _summarize(node_name, out)
            trace.append({"node": node_name, "ms": ms, "status": "ok",
                          "summary": summary, **payload})
            out["trace"] = trace
            out.setdefault("iteration", state.get("iteration", 0) + 1)
            return out

        return wrapper

    return deco


def _summarize(node: str, out: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """为每个节点生成一行人类可读的 trace 摘要。"""
    if node == "intent":
        return (f"意图={out.get('intent')}({out.get('intent_confidence')}) "
                f"实体={list((out.get('entities') or {}).keys())}", {})
    if node == "clarify":
        msg = (out.get("final_answer") or "")[:60]
        return (f"信息不足，返回澄清问题: {msg}", {})
    if node == "plan":
        plan = out.get("plan") or []
        steps = " → ".join(p.get("step", "?") for p in plan[:5])
        return (f"规划 {len(plan)} 步: {steps}", {"plan": plan})
    if node == "tool":
        if not out.get("tool_results"):
            return ("跳过（本意图无需调用工具）", {})
        results = out.get("tool_results") or []
        last = results[-1]
        return (f"调用工具 {last.get('tool', '?')} → {'成功' if last.get('ok') else '失败'}"
                f"（{last.get('cost_ms', 0)}ms）{last.get('summary') or last.get('error') or ''}",
                {"tool": last})
    if node == "retrieve":
        stats = (out.get("retrieval_stats") or {})
        return (f"召回 dense={stats.get('dense')} bm25={stats.get('bm25')} "
                f"profile={stats.get('profile')} → 融合 {stats.get('fused')}",
                {"retrieval_stats": stats})
    if node == "rerank":
        chunks = out.get("reranked_chunks") or []
        top = chunks[0] if chunks else {}
        return (f"Rerank 保留 {len(chunks)} 条，Top1={top.get('metadata', {}).get('name', '?')} "
                f"score={top.get('rerank_score')}", {"top": top})
    if node == "score":
        scores = out.get("scores") or []
        top = scores[0] if scores else {}
        return (f"评分 {len(scores)} 人，Top={top.get('candidate_id')} {top.get('total')}分", {})
    if node == "reflect":
        ref = out.get("reflection") or {}
        return (f"Reflection 第{ref.get('round', 0)}轮 → {ref.get('decision')} | "
                f"问题{len(ref.get('issues', []))} 缺失{len(ref.get('missing_aspects', []))}",
                {"reflection": ref})
    if node == "answer":
        return (f"生成答案 {len(out.get('final_answer', ''))} 字", {})
    if node == "shortlist":
        sl = out.get("shortlist") or []
        return (f"清单写入/更新 {len(sl)} 人", {})
    return (f"{node} 完成", {})


def budget_exceeded(state: dict) -> bool:
    from ..config import get_settings

    s = get_settings()
    return (state.get("iteration", 0) >= s.max_reflection_rounds + 3
            or state.get("revision_count", 0) >= s.max_reflection_rounds)

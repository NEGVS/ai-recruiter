"""节点 7：Reflection（自我批判与修订）。

这是整个 Agent 质量的「守门人」。它不是让模型泛泛地说「我觉得还不错」，
而是拿 **四个可验证的检查项** 去审上一版答案：

  C1 证据充分性  —— 每个结论是否有召回数据/工具结果支撑？
  C2 数值一致性  —— 回答中的数字是否与实际计算结果一致（防算错/防编造）？
  C3 约束遵守    —— 是否严格遵守了用户的硬约束（城市/年限/薪资）？
  C4 完整性      —— 是否遗漏了用户明确提出的多个要求中的某一项？

检查不通过 → 产出精确的 next_actions → 图路由回对应节点（定向补检索/补工具），
而不是笼统地「重新想一遍」。同时有轮次上限，防止死循环。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..config import get_settings
from ..constraints import verify_hard_constraints
from ..llm import get_llm
from ..state import RecruitState
from .common import timed

log = logging.getLogger(__name__)

REFLECTION_PROMPT = """你是招聘 Agent 的质量审查员。请严格审查下面这份回答，判断它是否可以交付给招聘专家。

审查维度（逐条给出结论，不要客套）：
1. 证据充分性：每个候选人推荐/每个结论，是否都能追溯到给定的检索片段或工具结果？
2. 数值一致性：文中出现的人数、分数、薪资、年限，是否与给定数据完全一致？
3. 约束遵守：是否严格满足用户提出的城市/年限/薪资/技能等硬约束？有没有推荐明显不满足的？
4. 完整性：用户问题中的**每一个**子问题是否都被回答了？
5. 风险披露：候选人的风险项（稳定性、薪资超预算、技能缺口）是否如实说明？

用户问题：{query}
识别意图：{intent}
检索/工具证据：
{evidence}
当前草稿：
{draft}

只输出 JSON：
{{"critique": "<100字内总评>",
  "issues": ["具体问题1", "具体问题2"],
  "missing_aspects": ["缺失的信息项，如：薪酬数据、面试记录"],
  "hallucination_flags": ["无依据的断言"],
  "decision": "accept" 或 "revise",
  "next_actions": ["补充工具调用" 或 "放宽经验约束重新召回" 或 "移除无证据候选人"],
  "score": 0.0-1.0}}
"""


# ================================================================= Reflection
@timed("reflect")
def reflect_node(state: RecruitState) -> dict:
    s = get_settings()
    rnd = state.get("revision_count", 0) + 1

    draft = state.get("final_answer") or ""
    ctx = _build_evidence_context(state)
    checks = _rule_based_checks(state, draft)

    llm = get_llm()
    crit: dict[str, Any] = {}
    try:
        crit = llm.chat_json(
            REFLECTION_PROMPT,
            f"用户问题：{state['user_query']}\n识别意图：{state.get('intent')}\n"
            f"检索/工具证据：\n{ctx}\n\n当前草稿：\n{draft or '(尚未生成草稿)'}",
            schema_hint='{"critique":str,"issues":[],"missing_aspects":[],'
                        '"hallucination_flags":[],"decision":"accept|revise",'
                        '"next_actions":[],"score":0.0}',
        )
    except Exception as e:  # noqa: BLE001
        log.warning("LLM 反思失败，回退规则检查: %s", e)
        crit = {}

    issues = list(dict.fromkeys(checks["issues"] + list(crit.get("issues") or [])))
    missing = list(dict.fromkeys(checks["missing"] + list(crit.get("missing_aspects") or [])))
    hal = list(dict.fromkeys(checks["hallucination"] + list(crit.get("hallucination_flags") or [])))

    # 规则检查是硬约束：即使 LLM 说 accept，只要存在数值不一致/约束违背就必须 revise
    hard_fail = bool(checks.get("hard_fail"))
    decision = "revise" if (issues or missing or hal) and rnd <= s.max_reflection_rounds else "accept"
    if hard_fail and rnd <= s.max_reflection_rounds:
        decision = "revise"
    if state.get("pending_interview"):
        decision = "accept"  # 等待用户确认时段，无需修订

    next_actions = list(crit.get("next_actions") or [])
    if decision == "revise" and not next_actions:
        next_actions = _derive_actions(issues, missing, state)

    refl = {
        "round": rnd,
        "critique": crit.get("critique") or ("；".join(issues) or "证据链完整，结论与数据一致。"),
        "issues": issues,
        "missing_aspects": missing,
        "hallucination_flags": hal,
        "decision": decision,
        "next_actions": next_actions,
        "score": round(float(crit.get("score", 0.0)) if crit else max(0.0, 1.0 - 0.2 * len(issues)), 3),
        "checks": checks["detail"],
    }

    history = list(state.get("reflection_history") or [])
    history.append({"round": rnd, "decision": decision, "issues": issues,
                    "score": refl["score"]})

    return {
        "reflection": refl,
        "reflection_history": history,
        "revision_count": rnd if decision == "revise" else state.get("revision_count", 0),
    }


# ================================================================= 规则检查
def _rule_based_checks(state: RecruitState, draft: str) -> dict[str, Any]:
    """不依赖 LLM 的硬性检查 —— 这是防幻觉的最后一道防线。"""
    issues: list[str] = []
    missing: list[str] = []
    hal: list[str] = []
    detail: dict[str, Any] = {}
    hard_fail = False

    scores = state.get("scores") or []
    tools = state.get("tool_results") or []
    entities = state.get("entities") or {}
    intent = state.get("intent", "")
    # 只有「候选人排序/评估」类意图才存在分数与个人风险；
    # 对 policy_qa 而言，被检索到的是制度文档，用 C2/C5 去审它们只会产生假告警。
    rank_intent = intent in ("candidate_search", "candidate_evaluate", "job_recommend")

    # --- C1 证据充分性 ---
    knowledge = state.get("knowledge_chunks") or []
    detail["C1_evidence"] = {
        "tool_calls": len(tools),
        "retrieved": len(state.get("retrieved_chunks") or []),
        "knowledge": len(knowledge),
        "reranked": len(state.get("reranked_chunks") or []),
        "scored": len(scores),
    }
    if intent in ("candidate_search", "candidate_evaluate", "job_recommend") and not scores:
        issues.append("无候选人评分结果，无法支撑推荐结论")
        hard_fail = True
    if intent == "policy_qa" and not state.get("knowledge_chunks"):
        missing.append("制度依据文档")
        hard_fail = True
    if intent == "policy_qa" and draft and "[POL" not in draft and "[PRC" not in draft:
        issues.append("回答未引用任何制度文档编号，无法溯源")

    # --- C2 数值一致性（仅推荐型回答）---
    if draft and scores and rank_intent:
        named = {s["name"] for s in scores}
        # 抽出草稿中出现的 Top 候选人，检查分数是否与实际一致
        mismatched = []
        for s_ in scores[:3]:
            if s_["name"] in draft:
                total = s_["total"]
                # 草稿里若写了与实际不符的分数（允许 ±0.6 误差）
                found = re.findall(rf"{re.escape(s_['name'])}[^\n]{{0,40}}?(?<![\d.])(\d{{1,3}}(?:\.\d+)?)\s*分", draft)
                for f in found:
                    if abs(float(f) - total) > 0.6:
                        mismatched.append(f"{s_['name']} 分数标注 {f} 与实际 {total} 不符")
        if mismatched:
            hal.extend(mismatched)
            hard_fail = True
        detail["C2_consistency"] = {"checked_names": sorted(named), "mismatches": mismatched}

        # 候选人数上限：说"前 N 名"但实际不足 N
        m = re.search(r"前\s*(\d+)\s*[名位]", draft)
        if m and int(m.group(1)) > len(scores):
            issues.append(f"声称输出前 {m.group(1)} 名，但实际仅 {len(scores)} 人")

    # --- C3 约束遵守 ---
    # 用「用户原始约束」而非「可能已被放宽的当前约束」做校验。
    # entities 中的 _original 是 retrieve_node 放宽时写入的原始快照；
    # 如果从未放宽，_original 不存在，直接用 entities 本身。
    original = entities.get("_original") or entities

    # 硬约束被改动（不该放宽的城市/薪资等）→ 报告
    hard_viols = verify_hard_constraints(entities, entities.get("_original"))
    if hard_viols:
        for v in hard_viols:
            issues.append(v)
        hard_fail = True

    cities = original.get("cities") or [] if rank_intent else []
    if cities and scores:
        viol = [s_["name"] for s_ in scores
                if _city_of(state, s_.get("candidate_id")) and _city_of(state, s_.get("candidate_id")) not in cities]
        if viol:
            issues.append(f"以下候选人不满足城市约束 {cities}：{', '.join(viol[:3])}")
            hard_fail = True
        detail["C3_constraints"] = {"cities": cities, "violations": viol}

    max_years = original.get("max_years") if rank_intent else None
    if max_years and scores:
        over = [s_["name"] for s_ in scores
                if _years_of(state, s_.get("candidate_id")) and _years_of(state, s_.get("candidate_id")) > float(max_years) + 1.5]
        if over:
            issues.append(f"以下候选人经验明显超出原始上限 {max_years} 年：{', '.join(over[:3])}")

    # --- C4 完整性：多子问题检测 ---
    q = state["user_query"]
    multi_markers = ["，", "并且", "同时", "还有", "以及", "and", "、"]
    parts = [p for p in re.split(r"[?？]", q)[0].split("，") if len(p) > 3]
    if len(parts) >= 2 and len(scores) >= 1:
        detail["C4_multi_part"] = {"parts": parts, "answered": len(scores)}

    # --- C5 风险披露（仅推荐型回答）---
    if scores and draft and rank_intent:
        risky = [s_ for s_ in scores[:3] if s_.get("risks")]
        undisclosed = [s_["name"] for s_ in risky if s_["name"] in draft
                       and not any(r[:4] in draft for r in s_["risks"])]
        if undisclosed:
            missing.append(f"未披露风险：{', '.join(undisclosed)}")
        detail["C5_risks"] = {"risky": [s_["name"] for s_ in risky], "undisclosed": undisclosed}

    # --- 无证据断言 ---
    if draft and not tools and not scores and not knowledge:
        hal.append("回答未引用任何工具或检索结果")

    return {"issues": issues, "missing": missing, "hallucination": hal,
            "detail": detail, "hard_fail": hard_fail}


def _city_of(state: RecruitState, cid: str | None) -> str | None:
    for d in state.get("reranked_chunks") or []:
        if d.get("metadata", {}).get("candidate_id") == cid:
            return d["metadata"].get("city")
    return None


def _years_of(state: RecruitState, cid: str | None) -> float | None:
    for d in state.get("reranked_chunks") or []:
        if d.get("metadata", {}).get("candidate_id") == cid:
            v = d["metadata"].get("years")
            return float(v) if v is not None else None
    return None


def _derive_actions(issues: list[str], missing: list[str], state: RecruitState) -> list[str]:
    """把问题映射为具体的下一步动作。"""
    actions: list[str] = []
    joined = " ".join(issues + missing)
    if "召回" in joined or "无候选人" in joined or "0 人" in joined:
        actions.append("放宽经验约束重新召回")
    if "薪酬" in joined or "薪资" in joined:
        actions.append("补充薪酬数据查询")
    if "超出" in joined or "不满足城市" in joined:
        actions.append("移除无证据候选人")
    if "制度依据" in joined:
        actions.append("补充工具调用")
    if not actions:
        actions.append("基于已有证据重新组织答案，补全缺失维度")
    return actions


def _build_evidence_context(state: RecruitState) -> str:
    """把证据压缩成给 Reflection 看的紧凑上下文。"""
    lines: list[str] = []
    for t in state.get("tool_results") or []:
        lines.append(f"[tool:{t['tool']}] ok={t['ok']} {t.get('summary', '')}")
    scores = state.get("scores") or []
    for s in scores[:6]:
        lines.append(
            f"[score] {s['name']}({s['candidate_id']}) total={s['total']} "
            f"passed={s.get('passed')} risks={s.get('risks', [])[:2]}"
        )
    for c in (state.get("reranked_chunks") or [])[:5]:
        md = c.get("metadata", {})
        lines.append(f"[chunk] {md.get('name', '')}|{md.get('title', '')}|"
                     f"{md.get('years', 0)}y|{md.get('city', '')}|"
                     f"rerank={c.get('rerank_score')}|{c.get('text', '')[:80]}")
    for c in (state.get("retrieved_chunks") or [])[:3]:
        if not state.get("reranked_chunks"):
            lines.append(f"[chunk] {c.get('text', '')[:100]}")
    for c in (state.get("knowledge_chunks") or [])[:5]:
        md = c.get("metadata", {})
        lines.append(f"[doc:{md.get('doc_id', '?')}] {md.get('title', '')}: {c.get('text', '')[:120]}")
    return "\n".join(lines)[:4000] or "(无证据)"

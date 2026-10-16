"""节点 8：答案生成（Grounded Answer Generation）。

铁律：**每一句话都必须有出处**。
因此本节点不是「把结果丢给 LLM 让它自由发挥」，而是：
  1. 先用确定性模板把结构化结果渲染成一份**事实底稿**（含精确数字、风险、证据片段）；
  2. 真实 LLM 时，让模型基于底稿做**改写与润色**（可调整组织方式，但不得新增数字）；
  3. 渲染后再跑一次数值校验（Reflection 已做一次，这里做输出侧的最终兜底）。

这样即便模型幻觉，也只会幻觉在「措辞」而不会幻觉在「事实与数字」。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..config import get_settings
from ..llm import get_llm
from ..state import RecruitState
from .common import timed

log = logging.getLogger(__name__)

ANSWER_PROMPT = """你是资深技术招聘顾问。基于给定的**事实底稿**回答用户问题。

规则（必须严格遵守）：
1. 只能使用底稿中出现的事实与数字，**绝对不能新增、修改、四舍五入任何数字**。
2. 不得推荐底稿中未出现的候选人。
3. 若底稿中某信息缺失，明确写「暂无数据」，不要猜测。
4. 保留底稿中的风险提示与判断依据，不要弱化。
5. 用中文回答，结构清晰（可分点），直接给结论，不要寒暄。

用户问题：{query}
意图：{intent}

事实底稿：
{facts}

请输出最终回答。
"""


@timed("answer")
def answer_node(state: RecruitState) -> dict:
    intent = state.get("intent", "chitchat")
    facts = build_facts(state)

    # 特殊意图直接用确定性渲染
    if state.get("pending_input"):
        return {"final_answer": state["pending_input"]["question"],
                "answer_meta": _meta(state, "clarification")}
    if intent == "interview_schedule" and state.get("pending_interview"):
        return {"final_answer": _render_interview(state), "answer_meta": _meta(state, "interview_pending")}

    text = ""
    llm = get_llm()
    if llm.name != "mock":
        try:
            text = llm.chat(
                ANSWER_PROMPT.replace("{query}", state["user_query"]).replace("{intent}", intent).replace("{facts}", facts)
            )
        except Exception as e:  # noqa: BLE001
            log.warning("LLM 生成失败，回退模板渲染: %s", e)
            text = ""

    if not text.strip():
        text = render_final(state, facts)

    text = _sanitize(text, state)

    return {
        "final_answer": text,
        "answer_meta": _meta(state, "generated"),
        "facts": facts,
    }


# ================================================================= 事实底稿
def build_facts(state: RecruitState) -> str:
    """把中间结果渲染为事实底稿 —— 这是答案生成的唯一事实来源。"""
    intent = state.get("intent", "")
    L: list[str] = []
    s = get_settings()

    tools = state.get("tool_results") or []
    if tools:
        L.append("## 工具执行结果")
        for t in tools:
            L.append(f"- {t['tool']}: {'成功' if t['ok'] else '失败'}"
                     + (f" — {t.get('summary', '')}" if t.get("summary") else "")
                     + (f" — 错误: {t['error']}" if t.get("error") else ""))
            for o in (t.get("observations") or [])[:3]:
                L.append(f"  - {o}")

    scores = state.get("scores") or []
    if scores:
        meta = state.get("score_meta") or {}
        L.append(f"\n## 候选人评分（岗位：{meta.get('job', 'N/A')}，"
                 f"通过线 {meta.get('pass_score', s.pass_score)}）")
        L.append("| 排名 | 姓名 | 当前职位 | 年限 | 城市 | 总分 | 通过 | 关键风险 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for sc in scores:
            dims = " ".join(
                f"{k}:{v['score']:.2f}×{v['weight']:.2f}"
                for k, v in (sc.get("dimensions") or {}).items()
            )
            risks = "；".join(r for r in (sc.get("risks") or [])[:2]) or "无"
            L.append(
                f"| {sc['rank']} | {sc['name']}({sc['candidate_id']}) | {sc['title']} | "
                f"{_years_of(state, sc['candidate_id'])} | {_city_of(state, sc['candidate_id'])} | "
                f"**{sc['total']}** | {'✅' if sc.get('passed') else '❌'} | {risks} |"
            )
            L.append(f"  - 维度明细：{dims}")
            for r in (sc.get("reasons") or [])[:3]:
                L.append(f"  - ✅ {r}")
            for r in (sc.get("risks") or [])[:3]:
                L.append(f"  - ⚠️ {r}")
            for e in (sc.get("evidence") or [])[:2]:
                L.append(f"  - 证据：{e}")

    chunks = state.get("reranked_chunks") or state.get("retrieved_chunks") or []
    if chunks:
        L.append("\n## 检索证据（供引用，不得超出此范围）")
        for c in chunks[:6]:
            md = c.get("metadata", {})
            title = md.get("title") or md.get("doc_id", "")
            L.append(f"- [{c.get('id')}] rerank={c.get('rerank_score', c.get('score', 0))} "
                     f"{title}: {c.get('text', '')[:200]}")

    ent = state.get("entities") or {}
    if ent.get("job_title") or ent.get("_job"):
        L.append(f"\n## 检索条件\n- 技能: {ent.get('skills', [])} | 城市: {ent.get('cities', [])} "
                 f"| 年限: {ent.get('min_years', '-')}~{ent.get('max_years', '-')} "
                 f"| 薪资上限: {ent.get('max_salary', '不限')}")
        if ent.get("_job"):
            j = ent["_job"]
            L.append(f"- 岗位要求：{j['title']} | 硬技能 {j.get('hard_skills', [])} | "
                     f"预算 {j.get('salary_min', 0)//1000}-{j.get('salary_max', 0)//1000}K | "
                     f"紧急度 {j.get('urgency')}")

    ref = state.get("reflection") or {}
    if ref:
        L.append(f"\n## 自我审查（第 {ref.get('round')} 轮）\n- 结论: {ref.get('decision')}\n"
                 f"- 总评: {ref.get('critique', '')}")

    if state.get("errors"):
        L.append(f"\n## 执行异常\n" + "\n".join(f"- {e}" for e in state["errors"][:3]))

    return "\n".join(L) or "（无可用事实）"


# ================================================================= 模板渲染
def render_final(state: RecruitState, facts: str) -> str:
    intent = state.get("intent", "")
    scores = state.get("scores") or []
    meta = state.get("score_meta") or {}
    s = get_settings()
    top_n = (state.get("entities") or {}).get("top_n", 5)

    if intent == "job_recommend" and scores:
        return _render_job_recommend(state, scores, meta, s)
    if intent in ("candidate_search", "candidate_evaluate") and scores:
        return _render_ranking(state, scores, meta, s, top_n)
    if intent == "market_insight":
        return _render_market(state)
    if intent == "offer_negotiation":
        return _render_offer(state)
    if intent == "policy_qa":
        return _render_policy(state)
    if intent == "chitchat":
        return "你好，我是 AI 招聘助手。我可以帮你做：\n- 人才检索（按技能/城市/年限找人）\n- 候选人匹配度评估\n- 面试排期\n- 薪酬与 offer 测算\n- 人才市场行情分析\n- 招聘制度问答\n\n直接说需求即可，例如「找 5 个北京 3 年以上 LLM 方向的候选人」。"
    if intent == "job_recommend":
        note = meta.get("note") or "未找到可推荐的岗位"
        who = meta.get("candidate") or "该候选人"
        return (f"## 岗位推荐\n\n暂未能为 {who} 生成岗位推荐：{note}。\n"
                "请确认候选人姓名是否正确，或先确保有在招岗位。")
    if scores:
        return _render_ranking(state, scores, meta, s, top_n)
    return "已完成处理，但未产出可展示的结论。建议补充更具体的需求（城市/技能/年限）。"


def _render_ranking(state, scores, meta, s, top_n) -> str:
    L: list[str] = []
    ent = state.get("entities") or {}
    intent = state.get("intent", "")
    passed = [x for x in scores if x.get("passed")]

    cond = []
    if ent.get("skills"):
        cond.append(f"技能 {'/'.join(ent['skills'][:5])}")
    if ent.get("cities"):
        cond.append(f"城市 {'、'.join(ent['cities'])}")
    if ent.get("min_years") is not None:
        cond.append(f"{ent['min_years']}+ 年经验")
    if ent.get("max_salary"):
        cond.append(f"薪资 ≤ {ent['max_salary']//1000}K")
    head = "候选人匹配度分析" if intent == "candidate_evaluate" else "候选人推荐"
    L.append(f"## {head}")
    L.append(f"**筛选条件**：{' ｜ '.join(cond) if cond else '按语义相关性'}")
    L.append(f"**评估岗位**：{meta.get('job', 'N/A')} ｜ **通过线**：{meta.get('pass_score', s.pass_score)} 分")
    L.append(f"**召回 → 精排 → 评分**：{len(state.get('retrieved_chunks') or [])} 条 → "
             f"{len(state.get('reranked_chunks') or [])} 条 → {len(scores)} 人入选\n")

    L.append(f"| # | 姓名 | 职位 | 年限 | 城市 | 总分 | 结论 |")
    L.append("|---|---|---|---|---|---|---|")
    for sc in scores:
        verdict = "推荐" if sc.get("passed") else "备选"
        L.append(f"| {sc['rank']} | **{sc['name']}** | {sc['title']} | "
                 f"{_years_of(state, sc['candidate_id'])} | {_city_of(state, sc['candidate_id'])} | "
                 f"**{sc['total']}** | {verdict} |")
    L.append("")

    L.append("### 逐人分析")
    for sc in scores:
        mark = "✅" if sc.get("passed") else "⚠️"
        L.append(f"\n**{mark} {sc['rank']}. {sc['name']}（{sc['title']}，{_years_of(state, sc['candidate_id'])} 年 "
                 f"｜{_city_of(state, sc['candidate_id'])}）— {sc['total']} 分**")
        dims = sc.get("dimensions") or {}
        if dims:
            parts = [f"{_dim_label(k)} {v['score']*100:.0f}分(权重{v['weight']*100:.0f}%)" for k, v in dims.items()]
            L.append(f"  - 维度：{' ｜ '.join(parts)}")
        for r in (sc.get("reasons") or [])[:3]:
            L.append(f"  - 亮点：{r}")
        for r in (sc.get("risks") or [])[:3]:
            L.append(f"  - 风险：{r}")

    L.append("\n### 结论与建议")
    vetoed = [x for x in scores if x.get("vetoed")]
    relaxed = meta.get("relaxed") or []

    if relaxed:
        for r in relaxed:
            b, a = r.get("before") or {}, r.get("after") or {}
            changed = {k: (b.get(k), a.get(k)) for k in ("min_years", "max_years", "max_salary")
                       if b.get(k) != a.get(k)}
            if changed:
                L.append(f"- ℹ️ 首轮无完全匹配结果，已在第 {r.get('round')} 轮**自动放宽**条件："
                         + "，".join(f"{k} {v[0]}→{v[1]}" for k, v in changed.items()))

    if vetoed:
        L.append(f"- ⛔ **{len(vetoed)}/{len(scores)} 位候选人触发一票否决**"
                 f"（硬技能零命中或经验严重不足），不满足岗位核心要求：")
        for v in vetoed[:4]:
            reason = next((r for r in (v.get("risks") or []) if r.startswith("⛔")), "不满足硬性要求")
            L.append(f"  - {v['name']}（{v['total']} 分）：{reason.lstrip('⛔ ')}")
        L.append("- **结论：当前候选池中无完全匹配人选。**")

    if not passed:
        top = scores[0]
        L.append(f"- ⚠️ **无候选人达到 {meta.get('pass_score', s.pass_score)} 分通过线**"
                 f"（最高分 {top['name']} {top['total']} 分）。")
        blockers = set()
        for sc in scores[:3]:
            for r in sc.get("risks") or []:
                if any(k in r for k in ("硬技能", "缺失", "低于要求", "超出", "经验")):
                    blockers.add(r.lstrip("⛔ "))
        if blockers:
            L.append(f"- 核心卡点：{list(blockers)[0]}")
        L.append("- 建议：① 放宽非关键硬技能或以 nice_skills 替代；② 扩大城市范围；"
                 "③ 启动猎头/内部转岗渠道；④ 急招岗位可考虑内部转岗优先。")
    else:
        names = "、".join(f"{p['name']}({p['total']})" for p in passed[:3])
        L.append(f"- ✅ **{len(passed)} 位候选人达到通过线**，推荐优先沟通：{names}")
        top = passed[0]
        if top.get("risks"):
            L.append(f"- ⚠️ Top1「{top['name']}」主要风险：{top['risks'][0]}")
        L.append(f"- 下一步：约面 → 面试 → offer 测算。急招岗位可先锁定 {passed[0]['name']}、{passed[1]['name'] if len(passed) > 1 else passed[0]['name']}。")
    L.append(f"\n_评分维度权重：{_weights_str(meta.get('weights') or s.scoring_weights)}_")
    return "\n".join(L)


def _render_job_recommend(state: RecruitState, scores, meta, s) -> str:
    """岗位推荐：候选人固定，对在招岗位按匹配度排序。"""
    who = meta.get("candidate") or "候选人"
    passed = [x for x in scores if x.get("passed")]
    L: list[str] = [f"## 为 {who} 推荐岗位",
                    f"**评估岗位数**：{meta.get('count', len(scores))} 个 ｜ "
                    f"**通过线**：{meta.get('pass_score', s.pass_score)} 分\n"]

    L.append("| # | 岗位 | 城市/层级/年限 | 匹配分 | 结论 |")
    L.append("|---|---|---|---|---|")
    for sc in scores:
        verdict = "推荐投递" if sc.get("passed") else ("不建议" if sc.get("vetoed") else "可考虑")
        L.append(f"| {sc['rank']} | **{sc['name']}** | {sc['title']} | "
                 f"**{sc['total']}** | {verdict} |")
    L.append("")

    L.append("### 逐岗分析")
    for sc in scores:
        mark = "✅" if sc.get("passed") else ("⛔" if sc.get("vetoed") else "⚠️")
        L.append(f"\n**{mark} {sc['rank']}. {sc['name']}（{sc['title']}）— {sc['total']} 分**")
        dims = sc.get("dimensions") or {}
        if dims:
            parts = [f"{_dim_label(k)} {v['score']*100:.0f}分(权重{v['weight']*100:.0f}%)"
                     for k, v in dims.items()]
            L.append(f"  - 维度：{' ｜ '.join(parts)}")
        for r in (sc.get("reasons") or [])[:3]:
            L.append(f"  - 亮点：{r}")
        for r in (sc.get("risks") or [])[:3]:
            L.append(f"  - 风险：{r}")

    L.append("\n### 结论与建议")
    if passed:
        names = "、".join(f"{p['name']}({p['total']})" for p in passed[:3])
        L.append(f"- ✅ **{len(passed)} 个岗位达到匹配线**，建议优先投递：{names}")
    else:
        top = scores[0]
        L.append(f"- ⚠️ **无岗位达到 {meta.get('pass_score', s.pass_score)} 分匹配线**"
                 f"（最高 {top['name']} {top['total']} 分）。")
        blockers = [r.lstrip("⛔ ") for r in (top.get("risks") or [])
                    if any(k in r for k in ("硬技能", "缺失", "低于要求", "经验"))]
        if blockers:
            L.append(f"- 核心卡点（以最高分岗位为例）：{blockers[0]}")
        L.append("- 建议：补齐岗位核心硬技能，或关注与候选人技能更匹配的新岗位。")
    L.append(f"\n_评分维度权重：{_weights_str(meta.get('weights') or s.scoring_weights)}_")
    return "\n".join(L)


def _render_interview(state: RecruitState) -> str:
    p = state.get("pending_interview") or {}
    L = [f"## 面试排期 — {p.get('candidate')}" + (f"（{p.get('round')}）" if p.get("round") else ""),
         f"面试官：{p.get('interviewer')} ｜ 检测到 {len(p.get('candidates', []))} 个无冲突时段：", ""]
    for i, slot in enumerate(p.get("candidates", []), 1):
        L.append(f"{i}. **{slot['date']} {slot['hour']}:00**（{slot.get('duration', 60)} 分钟）")
    L.append("\n回复「确认第 N 个时段」即可写入日程。")
    return "\n".join(L)


def _render_market(state: RecruitState) -> str:
    """市场分析：把工具 payload 完整渲染成人可读报告。"""
    from ..tools import get_tool

    payload = None
    for t in reversed(state.get("tool_results") or []):
        if t["tool"] == "market_insight" and t["ok"]:
            payload = t
            break
    if not payload:
        return "## 人才市场分析\n\n暂未取到市场数据，请补充要分析的方向（如 LLM、后端）。"

    # payload 未直接存在 result 里时，从 observations 重建关键信息
    p = payload.get("_payload") or {}
    if not p:
        return ("## 人才市场分析\n\n"
                + "\n".join(f"- {o}" for o in payload.get("observations", []))
                + "\n\n> 提示：补充城市与岗位名称可获得更精确的分位数据。")

    topic, city = p.get("topic", ""), p.get("city") or "全城市"
    L = [f"## 人才市场分析 — {topic}（{city}）", ""]
    L.append(f"- **库内命中**：{p.get('matched', 0)} / {p.get('pool', 0)} 人")
    L.append(f"- **供给指数**：{p.get('supply_index', 0)}（0=极度稀缺，1=供给充足）")
    L.append(f"- **平均经验**：{p.get('avg_years', 0)} 年")
    L.append("")
    L.append("### 薪资分位（期望年薪）")
    L.append(f"| P25 | P50 | P75 |")
    L.append("|---|---|---|")
    L.append(f"| {_k(p.get('salary_p25'))} | **{_k(p.get('salary_p50'))}** | {_k(p.get('salary_p75'))} |")
    L.append("")
    dist = p.get("intent_dist") or {}
    if dist:
        total = sum(dist.values()) or 1
        parts = [f"{k} {v}人({v*100//total}%)" for k, v in dist.items()]
        L.append(f"**求职意向分布**：{' ｜ '.join(parts)}")
    if p.get("assessment"):
        L.append(f"\n**供需评估**：{p['assessment']}")
    tc = p.get("top_candidates") or []
    if tc:
        L.append("\n### 该方向活跃候选人（经验排序）")
        L.append("| 姓名 | 职位 | 年限 | 期望薪资 | 求职意向 |")
        L.append("|---|---|---|---|---|")
        for c in tc:
            L.append(f"| {c['name']} | {c['title']} | {c['years']}年 | "
                     f"{_k(c.get('expected_salary'))} | {c.get('job_intent','')} |")
    L.append("\n**建议**：" + _market_advice(p))
    return "\n".join(L)


def _k(v: Any) -> str:
    try:
        return f"{int(v)//1000}K"
    except Exception:  # noqa: BLE001
        return "-"


def _market_advice(p: dict) -> str:
    m = p.get("matched", 0)
    if m == 0:
        return "库内无完全命中样本，建议放宽到相邻技能或扩大城市范围，并启动外部渠道。"
    if m < 5:
        return f"供给紧张（仅 {m} 人），建议：① 放宽相邻技能要求；② 提高预算 10%~15%；③ 同步启动猎头。"
    dist = p.get("intent_dist") or {}
    active_ratio = dist.get("active", 0) / max(sum(dist.values()), 1)
    if active_ratio > 0.6:
        return f"活跃候选人占比 {active_ratio*100:.0f}%，竞争激烈，建议提前锁定并适度提薪。"
    return "供需相对平衡，可按标准流程推进。"


def _render_offer(state: RecruitState) -> str:
    """Offer 测算：完整渲染薪酬结构与审批提示。"""
    p = None
    for t in reversed(state.get("tool_results") or []):
        if t["tool"] == "calc_offer" and t["ok"]:
            p = t.get("_payload") or {}
            break
    if not p:
        return "## Offer 测算\n\n请提供候选人的期望薪资（月薪或总包均可）。"

    L = ["## Offer 测算", ""]
    L.append("| 项目 | 金额 |")
    L.append("|---|---|")
    L.append(f"| 候选人期望年薪 | {_wan(p.get('expected_annual'))} |")
    L.append(f"| 月薪底薪 | {_wan(p.get('monthly_base'))} |")
    L.append(f"| 发薪月数 | {p.get('months')} |")
    L.append(f"| 年度底薪合计 | {_wan(p.get('annual_base'))} |")
    if p.get("bonus"):
        L.append(f"| 绩效年薪 | {_wan(p.get('bonus'))} |")
    if p.get("signing"):
        L.append(f"| 签字费 | {_wan(p.get('signing'))} |")
    L.append(f"| **总包** | **{_wan(p.get('total_package'))}** |")
    L.append("")
    L.append(f"- **覆盖率**：{(p.get('coverage') or 0)*100:.0f}%（{p.get('verdict')}）")
    for n in p.get("notes", []):
        L.append(f"- {n}")
    ce = p.get("candidate_expected")
    if ce:
        L.append(f"- **候选人实际期望**：{_wan(ce)} → 本方案 "
                 f"{'低于' if (p.get('gap_to_candidate') or 0) < 0 else '高于'} "
                 f"{_wan(abs(p.get('gap_to_candidate') or 0))}")

    # 候选人信息
    for t in state.get("tool_results") or []:
        if t["tool"] == "get_candidate" and t["ok"]:
            L.append(f"\n> 候选人：{t.get('summary', '')}")

    L.append("\n**审批提示**：总包超岗位带宽上限 110% 需 VP 级审批；带宽内但处于 P90 分位以上需总监级审批；"
             "签字费单人不超过 3 个月薪资。")
    return "\n".join(L)


def _wan(v: Any) -> str:
    try:
        n = float(v)
    except Exception:  # noqa: BLE001
        return "-"
    if n >= 10000:
        return f"{n/10000:.1f}万"
    return f"{n:.0f}元"


def _render_policy(state: RecruitState) -> str:
    chunks = state.get("knowledge_chunks") or []
    if not chunks:
        return "## 制度问答\n\n未检索到相关制度文档。"
    L = ["## 制度依据"]
    for c in chunks[:3]:
        md = c.get("metadata", {})
        L.append(f"\n**[{md.get('doc_id', md.get('source', '文档'))}] {md.get('title', '')}**")
        L.append(f"  {c.get('text', '').strip()}")
    L.append("\n_以上内容来自公司内部制度文档，如与实际流程冲突请以 HRBP 确认为准。_")
    return "\n".join(L)


# ================================================================= 后处理
_DIM_LABELS = {
    "hard_skill": "硬技能", "domain_experience": "领域经验", "seniority": "资历匹配",
    "interview": "面试表现", "stability": "稳定性", "salary_fit": "薪酬匹配",
    "job_intent": "求职意向",
}


def _dim_label(k: str) -> str:
    return _DIM_LABELS.get(k, k)


def _weights_str(w: dict) -> str:
    return "，".join(f"{_DIM_LABELS.get(k, k)} {v*100:.0f}%" for k, v in w.items())


def _years_of(state, cid) -> str:
    for d in state.get("reranked_chunks") or []:
        if d.get("metadata", {}).get("candidate_id") == cid:
            return f"{d['metadata'].get('years', 0):g}"
    return "-"


def _city_of(state, cid) -> str:
    for d in state.get("reranked_chunks") or []:
        if d.get("metadata", {}).get("candidate_id") == cid:
            return d["metadata"].get("city", "")
    return "-"


def _sanitize(text: str, state: RecruitState) -> str:
    """输出侧最终兜底：删除任何与实际评分不符的数字。

    正则必须避免把「98.95」的一部分（如「95」）误当成独立分数，
    因此加负向后视断言：数字左侧不能再有数字或小数点。
    """
    import re

    scores = {s["name"]: float(s["total"]) for s in (state.get("scores") or [])}
    if not scores:
        return text

    names = "|".join(re.escape(n) for n in sorted(scores, key=len, reverse=True))
    # 姓名后 40 字符内出现的「独立分数」
    pattern = re.compile(
        rf"({names})([^\n]{{0,40}}?)(?<![\d.])(\d{{1,3}}(?:\.\d+)?)\s*分"
    )

    def fix(m: re.Match) -> str:
        name, mid, num = m.group(1), m.group(2), m.group(3)
        actual = scores.get(name)
        if actual is None:
            return m.group(0)
        if abs(float(num) - actual) <= 0.05:
            return m.group(0)
        log.warning("修正幻觉分数：%s %s → %s", name, num, actual)
        return f"{name}{mid}{actual:g} 分"

    return pattern.sub(fix, text)


def _meta(state: RecruitState, kind: str) -> dict[str, Any]:
    return {
        "type": kind,
        "intent": state.get("intent"),
        "confidence": state.get("intent_confidence"),
        "tool_calls": state.get("tool_call_count", 0),
        "retrieved": len(state.get("retrieved_chunks") or []),
        "reranked": len(state.get("reranked_chunks") or []),
        "scored": len(state.get("scores") or []),
        "reflection_rounds": state.get("revision_count", 0),
        "iteration": state.get("iteration", 0),
        "errors": state.get("errors", []),
    }

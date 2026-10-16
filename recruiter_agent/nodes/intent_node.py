"""节点 1：意图识别（Intent Recognition）

双通道：
  * 规则通道（intent_rules）先跑，产出意图候选集 + 实体
  * LLM 通道在候选集内做精细判别（真实 LLM 时生效）

实体抽取的可靠性是整条链路的地基：城市/年限/技能一旦抽错，
后续的 filter 会把正确候选人直接过滤掉。所以这里做了「规则兜底 + LLM 补充」：
LLM 只允许**新增**规则没抽到的字段，不允许覆盖规则已确认的值。
"""
from __future__ import annotations

import json
import logging

from ..intent_rules import INTENT_PROMPT, INTENTS, candidate_intents, extract_entities, rule_based_intent
from ..llm import get_llm
from ..state import RecruitState
from .common import timed

log = logging.getLogger(__name__)

# 置信度低于此值 → 认为意图不明，需要澄清
CLARIFY_THRESHOLD = 0.45

CLARIFY_QUESTIONS = {
    "candidate_search": "请补充城市、工作年限和核心技能方向（例如：北京 / 3年以上 / LLM、RAG），我来精准检索。",
    "candidate_evaluate": "请告诉我要评估哪位候选人、匹配哪个岗位（或岗位名称），我来做匹配度分析。",
    "job_recommend": "请告诉我要推荐给哪位候选人（姓名），或者说明候选人的技能方向，我来匹配岗位。",
    "interview_schedule": "请说明要为哪位候选人安排哪一轮面试、期望日期（可选），我来检查面试官档期并给出可选时段。",
    "offer_negotiation": "请说明是哪位候选人的 offer 谈判，以及期望月薪或总包，我会计算覆盖率并给出审批建议。",
    "market_insight": "请说明要分析哪个方向（如 LLM、后端）的市场行情，我给出供需与薪资分位分析。",
    "policy_qa": "请说明你想了解的具体政策问题（如面试改期、offer 审批、背景调查），我来查证公司制度。",
    "chitchat": "",
}


@timed("intent")
def intent_node(state: RecruitState) -> dict:
    q = state["user_query"]

    # --- 规则通道 ---
    rule = rule_based_intent(q)
    rule_intent = rule["intent"]
    rule_conf = float(rule["confidence"])
    entities = dict(rule["entities"])

    llm = get_llm()
    intent, conf, reason = rule_intent, rule_conf, f"规则:{rule['reason']}"

    # --- LLM 通道（仅在候选集与规则结论有分歧时才调用，节省 token）---
    try:
        cands = candidate_intents(q)
        schema = {"intent": "", "confidence": 0.0, "reason": "", "entities": {}}
        if llm.name != "mock" or rule_conf < 0.7:
            prompt_intents = "\n".join(
                f"- {k}: {v}" for k, v in INTENTS.items() if k in cands)
            data = llm.chat_json(
                INTENT_PROMPT.replace("{intent_list}", prompt_intents), q,
                schema_hint=json.dumps(schema, ensure_ascii=False),
            )
            cand_intent = data.get("intent", "")
            if cand_intent in cands or cand_intent in INTENTS:
                if cand_intent != rule_intent:
                    # 分歧时以 LLM 为准，但要求 LLM 置信度更高
                    if float(data.get("confidence", 0)) >= rule_conf * 0.9:
                        intent = cand_intent
                        conf = float(data.get("confidence", rule_conf))
                        reason = f"LLM:{data.get('reason', '')}（规则为 {rule_intent}，采信 LLM）"
                    else:
                        reason = f"规则:{rule['reason']}；LLM 倾向 {cand_intent} 但置信度低，采信规则"
                else:
                    conf = max(conf, float(data.get("confidence", conf)))
                    reason = f"规则+LLM 一致：{data.get('reason', '')}"
                # 实体合并：只补充，规则已有字段优先
                for k, v in (data.get("entities") or {}).items():
                    if k not in entities or not entities[k]:
                        entities[k] = v
    except Exception as e:  # noqa: BLE001
        log.warning("LLM 意图识别失败，回退规则: %s", e)
        reason += f"；LLM 失败({e})，采信规则"

    # --- 实体后处理 ---
    entities = _normalize_entities(entities, q, intent)
    need_clarify = conf < CLARIFY_THRESHOLD

    return {
        "intent": intent,
        "intent_confidence": round(min(1.0, conf), 3),
        "entities": entities,
        "clarified": need_clarify,
        "intent_reason": reason,
    }


def _normalize_entities(entities: dict, query: str, intent: str) -> dict:
    """补齐与纠正实体。"""
    # 兜底重抽一次（LLM 可能改动过）
    base = extract_entities(query)
    for k, v in base.items():
        if k not in entities or not entities[k]:
            entities[k] = v

    # top_n 归一
    if "top_n" in entities:
        try:
            entities["top_n"] = max(1, min(30, int(entities["top_n"])))
        except Exception:  # noqa: BLE001
            entities.pop("top_n", None)
    entities.setdefault("top_n", 5 if intent in ("candidate_search", "candidate_evaluate") else 3)

    # 薪资：extract_entities 已统一折算为年薪，此处只做区间归一
    if isinstance(entities.get("salary_kw"), list) and entities["salary_kw"]:
        vals = [int(v) for v in entities["salary_kw"]]
        if len(vals) == 1:
            entities["max_salary"] = vals[0]
        else:
            entities["max_salary"] = max(vals)
            entities["min_salary"] = min(vals)
        entities["expected_salary"] = vals[-1]

    # 清理空值：None 会污染 Trace 摘要，也会让下游 `.get()` 判断失真
    entities = {k: v for k, v in entities.items() if v is not None and v != [] and v != {}}

    # 经验字段合理性
    if "min_years" in entities and entities["min_years"] > 25:
        entities.pop("min_years")
    if "max_years" in entities and "min_years" not in entities:
        entities["min_years"] = max(0.0, entities["max_years"] - 3)

    return entities


@timed("clarify")
def clarify_node(state: RecruitState) -> dict:
    """信息不足 → 反问，而不是硬猜。"""
    intent = state.get("intent", "chitchat")
    entities = state.get("entities", {})
    base = CLARIFY_QUESTIONS.get(intent, "请补充更多细节，我再帮你处理。")

    # 追加「已识别」信息，减少用户重复输入
    known = []
    if entities.get("cities"):
        known.append(f"城市={','.join(entities['cities'])}")
    if entities.get("min_years") is not None:
        known.append(f"年限={entities['min_years']}年")
    if entities.get("skills"):
        known.append(f"技能={','.join(entities['skills'][:5])}")
    if known:
        base += f"\n（已识别：{'；'.join(known)}，只需补充缺失部分）"

    return {
        "final_answer": base,
        "clarified": True,
        "answer_meta": {"type": "clarification", "intent": intent},
    }

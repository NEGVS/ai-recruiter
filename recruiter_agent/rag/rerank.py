"""Rerank 模块。

三级策略，按可用性自动降级：
1. cross_encoder —— CrossEncoder 本地精排（招聘场景对「硬技能是否命中」判别力最强）
2. llm           —— 交给 LLM 做 pairwise/listwise 打分（可解释，但慢）
3. rrf           —— 纯规则：RRF 融合分 + 硬技能命中加分 + 经验/意向匹配加分

第 3 种即便没有模型权重也能给出**业务感知**的排序，而不是单纯的向量相似度，
这是这个项目里 rerank 层真正的价值所在。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from ..config import Settings, get_settings
from ..llm import get_llm
from ..state import Chunk, ScoreBreakdown

log = logging.getLogger(__name__)

_cross_encoder = None


def get_cross_encoder(model_name: str):
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


# ------------------------------------------------------------------ 硬技能匹配
def _norm(s: str) -> str:
    return re.sub(r"[\s\-_/()（）]", "", (s or "").lower())


def skill_match(query_skills: Sequence[str], chunk: Chunk) -> tuple[float, list[str]]:
    """硬技能命中率 —— 招聘场景最关键的一票否决式信号。"""
    if not query_skills:
        return 0.0, []
    haystack = _norm(chunk.text + " " + " ".join(map(str, chunk.metadata.get("skills", [])))
                     + " " + str(chunk.metadata.get("highlights", [])))
    hits = [s for s in query_skills if _norm(s) in haystack]
    return len(hits) / len(query_skills), hits


# ------------------------------------------------------------------ 规则 rerank
def rule_rerank(
    chunks: list[Chunk],
    query: str,
    entities: dict[str, Any],
    top_n: int = 5,
    rrf_k: int = 60,
) -> list[Chunk]:
    """业务感知精排。"""
    q_skills: list[str] = list(entities.get("skills", []))
    min_years = entities.get("min_years")
    max_years = entities.get("max_years")
    cities = set(entities.get("cities", []))
    job_intent = entities.get("job_intent")

    for c in chunks:
        base = 1.0 / (rrf_k + (c.dense_rank or 99)) + 1.0 / (rrf_k + (c.sparse_rank or 99))
        skill_ratio, hits = skill_match(q_skills, c)
        md = c.metadata

        bonus = 0.0
        # 硬技能命中（权重最高）
        bonus += 0.9 * skill_ratio
        if hits:
            c.evidence.append(f"命中技能：{', '.join(hits)}")
        # 经验区间
        yrs = float(md.get("years", 0) or 0)
        if min_years is not None and yrs >= float(min_years):
            bonus += 0.25
            c.evidence.append(f"经验 {yrs}年 ≥ 要求 {min_years}年")
        if max_years is not None and yrs > float(max_years) + 1.5:
            bonus -= 0.35  # 明显超配，实践中易谈崩
        # 城市
        if cities and str(md.get("city", "")) in cities:
            bonus += 0.2
            c.evidence.append(f"城市匹配 {md.get('city')}")
        # 求职意向：主动候选人优先
        if job_intent and md.get("job_intent") == job_intent:
            bonus += 0.15
        elif md.get("job_intent") == "active":
            bonus += 0.05
        # 可到岗时间
        av = int(md.get("availability_days", 30) or 30)
        if av <= 14:
            bonus += 0.12
            c.evidence.append(f"{av}天可到岗")
        # 面试完成度：chunk metadata 存的是 interview_score（扁平字段），
        # 不是 interview 字典——早期误读 md.get("interview") 导致此加分恒不触发。
        iv_score = float(md.get("interview_score", 0) or 0)
        if iv_score > 0:
            bonus += 0.1 * min(1.0, iv_score / 100)
        # 一年内多次跳动降权
        if int(md.get("job_changes_3y", 1) or 1) >= 3:
            bonus -= 0.2
            c.evidence.append("⚠️ 3年内3次以上跳动")

        c.rerank_score = round(base * 2.0 + bonus, 6)
        c.metadata["_skill_ratio"] = round(skill_ratio, 3)
        c.metadata["_rerank_bonus"] = round(bonus, 3)

    chunks.sort(key=lambda c: c.rerank_score or -1, reverse=True)
    return chunks[:top_n]


# ------------------------------------------------------------------ CrossEncoder
def cross_encoder_rerank(chunks: list[Chunk], query: str, top_n: int = 5) -> list[Chunk]:
    pairs = [(query, c.text) for c in chunks]
    ce = get_cross_encoder(get_settings().rerank_model)
    scores = ce.predict(pairs)
    for c, s in zip(chunks, scores):
        c.rerank_score = float(s)
        c.metadata["_skill_ratio"] = skill_match([], c)[0]
    chunks.sort(key=lambda c: c.rerank_score or -1, reverse=True)
    return chunks[:top_n]


# ------------------------------------------------------------------ LLM rerank
LLM_RERANK_PROMPT = """你是资深技术招聘官。为每个候选片段按【与检索需求的相关度】打分 0-100。

评分标准：
- 100：完全命中全部硬技能，且经验/城市/意向完全匹配
- 70-90：命中主要硬技能，个别次要项缺失
- 40-69：部分相关，需二次确认
- 0-39：与需求无关

检索需求：{query}
附加约束：{entities}

候选片段：
{candidates}

只输出 JSON：{{"scores": [{{"id": "<chunk_id>", "score": 85, "why": "<15字内理由>"}}]}}
"""


def llm_rerank(chunks: list[Chunk], query: str, entities: dict[str, Any], top_n: int = 5) -> list[Chunk]:
    llm = get_llm()
    cands = "\n".join(
        f'- id={c.id} | {c.metadata.get("name", c.candidate_id)} | {c.metadata.get("title","")} | '
        f'{c.metadata.get("years",0)}年 | 技能:{",".join(map(str, c.metadata.get("skills", [])[:8]))} | 片段:{c.text[:160]}'
        for c in chunks
    )
    data = llm.chat_json(
        LLM_RERANK_PROMPT,
        f"检索需求：{query}\n附加约束：{entities}\n\n候选片段：\n{cands}",
        schema_hint='{"scores":[{"id":str,"score":int,"why":str}]}',
    )
    mapping = {d.get("id"): (d.get("score", 0), d.get("why", "")) for d in data.get("scores", [])}
    for c in chunks:
        sc, why = mapping.get(c.id, (0.0, ""))
        c.rerank_score = float(sc)
        c.metadata["_rerank_why"] = why
        if why:
            c.evidence.append(f"LLM: {why}")
    chunks.sort(key=lambda c: c.rerank_score or -1, reverse=True)
    return chunks[:top_n]


# ------------------------------------------------------------------ 统一入口
def rerank(
    chunks: list[Chunk],
    query: str,
    entities: dict[str, Any] | None = None,
    top_n: int | None = None,
    s: Settings | None = None,
) -> list[Chunk]:
    s = s or get_settings()
    entities = entities or {}
    top_n = top_n or s.rerank_top_n
    if not chunks:
        return []

    if s.rerank_provider == "cross_encoder":
        try:
            out = cross_encoder_rerank(chunks, query, top_n)
            _attach_business_signals(out, entities)
            return out
        except Exception as e:  # noqa: BLE001
            log.warning("CrossEncoder 不可用(%s)，回退规则 rerank", e)
    elif s.rerank_provider == "llm":
        try:
            out = llm_rerank(chunks, query, entities, top_n)
            _attach_business_signals(out, entities)
            return out
        except Exception as e:  # noqa: BLE001
            log.warning("LLM rerank 失败(%s)，回退规则 rerank", e)

    return rule_rerank(chunks, query, entities, top_n, s.rrf_k)


def _attach_business_signals(chunks: list[Chunk], entities: dict[str, Any]) -> None:
    """在模型精排之上叠加业务信号（硬技能命中/城市/意向），不改变主排序语义。"""
    q_skills = list(entities.get("skills", []))
    cities = set(entities.get("cities", []))
    for c in chunks:
        ratio, hits = skill_match(q_skills, c)
        md = c.metadata
        adj = 0.15 * ratio
        if cities and str(md.get("city", "")) in cities:
            adj += 0.05
        c.rerank_score = round((c.rerank_score or 0.0) + adj, 6)
        c.metadata["_skill_ratio"] = round(ratio, 3)
        if hits:
            c.evidence.append(f"命中技能：{', '.join(hits)}")
    chunks.sort(key=lambda c: c.rerank_score or -1, reverse=True)

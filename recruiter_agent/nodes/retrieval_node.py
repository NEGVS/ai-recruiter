"""节点 4/5/6：RAG 检索 → Rerank → 评分。

这三者是流水线上的核心计算环节，各自独立成节点的好处：
  * 每一环的召回/精排效果都能单独评测（见 eval/）；
  * Reflection 发现问题时，可以精确回到某一环（如「召回不足 → 回 retrieval 放宽条件」），
    而不是整条链重跑。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..rag import get_retriever
from ..scoring import score_candidate
from ..state import Candidate, Chunk, JobRequisition, RecruitState
from ..vectorstore import COLLECTION_CANDIDATES, COLLECTION_KNOWLEDGE
from .common import timed

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
PROFILE_CACHE = ROOT / "data" / "profiles_cache.json"


# ================================================================= 4. 检索
@timed("retrieve")
def retrieve_node(state: RecruitState) -> dict:
    intent = state.get("intent", "candidate_search")
    entities = dict(state.get("entities") or {})
    query = state["user_query"]
    r = get_retriever()

    # 岗位已确定时，把 JD 要求并入检索查询（提升语义召回精度）
    job = entities.get("_job")
    if job:
        jd_text = f"{job['title']} {' '.join(job.get('hard_skills', []))} {' '.join(job.get('nice_skills', []))}"
        query = f"{query} {jd_text}".strip()

    # Reflection 提出「放宽硬约束」→ 放宽一次（下调门槛，而非清零约束）
    ref = state.get("reflection") or {}
    relax_hint = " ".join(ref.get("next_actions") or []) + " ".join(ref.get("issues") or [])
    if "放宽" in relax_hint:
        before = {k: entities.get(k) for k in ("min_years", "max_years", "max_salary")}
        relaxed = dict(entities)
        if relaxed.get("min_years") is not None:
            relaxed["min_years"] = max(0.0, float(relaxed["min_years"]) - 2.0)
        if relaxed.get("max_years") is not None and float(relaxed["max_years"]) < 12:
            relaxed["max_years"] = float(relaxed["max_years"]) + 3.0
        if relaxed.get("max_salary"):
            relaxed["max_salary"] = float(relaxed["max_salary"]) * 1.15
        after = {k: relaxed.get(k) for k in ("min_years", "max_years", "max_salary")}
        if after != before:
            hist = list(relaxed.get("_relaxed") or [])
            hist.append({"round": ref.get("round"), "before": before, "after": after})
            relaxed["_relaxed"] = hist
            entities = relaxed
            log.info("Reflection 第%s轮放宽约束：%s → %s", ref.get("round"), before, after)

    res = r.retrieve(query, entities, intent, COLLECTION_CANDIDATES, do_rerank=False)
    return {
        "retrieved_chunks": [_dump(c) for c in res.chunks],
        "retrieval_stats": res.stats,
        "rewritten_query": res.rewritten_query,
        "expanded_terms": res.expanded_terms,
        "entities": entities,
    }


@timed("knowledge")
def knowledge_node(state: RecruitState) -> dict:
    """政策/制度问答走独立知识库。"""
    r = get_retriever()
    res = r.retrieve_knowledge(state["user_query"], do_rerank=True)
    # 知识库结果单独存放：与候选人 chunk 混在一起会让反思节点
    # 把制度文档当成「候选人」去做数值一致性/风险披露检查。
    return {
        "knowledge_chunks": [_dump(c) for c in res.chunks],
        "retrieval_stats": res.stats,
        "rewritten_query": res.rewritten_query,
    }


# ================================================================= 5. Rerank
@timed("rerank")
def rerank_node(state: RecruitState) -> dict:
    from ..rag import rerank as do_rerank
    from ..config import get_settings

    s = get_settings()
    chunks = [_load(d) for d in state.get("retrieved_chunks", [])]
    if not chunks:
        return {"reranked_chunks": [], "rerank_meta": {"count": 0}}

    entities = state.get("entities") or {}
    query = state.get("rewritten_query") or state["user_query"]
    top_n = max(entities.get("top_n", 5), s.rerank_top_n)

    ranked = do_rerank(chunks, query, entities, top_n=top_n, s=s)

    # 同一人多 chunk 聚合，保留最强证据
    by_cand: dict[str, dict[str, Any]] = {}
    for c in ranked:
        cid = c.candidate_id
        slot = by_cand.get(cid)
        if slot is None:
            by_cand[cid] = {"chunk": c, "best": c.rerank_score or 0.0,
                            "n_chunks": 1, "evidence": list(c.evidence)}
        else:
            slot["n_chunks"] += 1
            slot["evidence"].extend(c.evidence)
            if (c.rerank_score or 0) > slot["best"]:
                slot["best"] = c.rerank_score or 0
                slot["chunk"] = c

    final: list[dict[str, Any]] = []
    for cid, slot in by_cand.items():
        c = slot["chunk"]
        c.evidence = list(dict.fromkeys(slot["evidence"]))[:6]
        c.metadata["_chunk_hits"] = slot["n_chunks"]
        final.append(_dump(c))
    final.sort(key=lambda d: d.get("rerank_score") or -1, reverse=True)

    return {
        "reranked_chunks": final,
        "rerank_meta": {
            "count": len(final), "provider": s.rerank_provider,
            "input": len(chunks), "query": query,
        },
    }


# ================================================================= 6. 评分
@timed("score")
def score_node(state: RecruitState) -> dict:
    """对 rerank 后的候选人做多维打分。"""
    from ..config import get_settings

    s = get_settings()
    entities = state.get("entities") or {}
    intent = state.get("intent")

    # job_recommend：固定候选人，把每个在招岗位当作 JD 反向打分，
    # 复用同一套七维评分与一票否决逻辑（「这个人适合哪些岗位」= 候选人对岗位的匹配度）。
    if intent == "job_recommend":
        return _score_jobs_for_candidate(entities, s)

    chunks = [_load(d) for d in state.get("reranked_chunks", [])]

    if not chunks:
        return {"scores": [], "score_meta": {"count": 0, "note": "无候选可评分"}}

    # 构造 JD：显式 JD 优先；否则用检索条件构造虚拟 JD（保证评分有依据）
    job = _resolve_job(entities, state)

    profiles = _load_profiles()
    top_n = max(entities.get("top_n", 5), 5)
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in chunks:
        cid = c.candidate_id
        if cid in seen:
            continue
        seen.add(cid)
        prof = profiles.get(cid)
        cand = _to_candidate(cid, prof, c)
        sb = score_candidate(cand, job, c, s)
        d = {
            "candidate_id": cid,
            "name": cand.name,
            "title": cand.title,
            "total": sb.total,
            "dimensions": sb.dimensions,
            "reasons": sb.reasons,
            "risks": sb.risks,
            "evidence": sb.evidence,
            "vetoed": sb.vetoed,
            # 一票否决优先于分数线
            "passed": (not sb.vetoed) and sb.total >= s.pass_score,
        }
        results.append(d)
        if len(results) >= top_n * 2:
            break

    results.sort(key=lambda x: x["total"], reverse=True)
    results = results[:top_n]
    for i, r in enumerate(results, 1):
        r["rank"] = i

    return {
        "scores": results,
        "score_meta": {
            "count": len(results), "job": f"{job.title}({job.job_id})",
            "pass_score": s.pass_score,
            "passed": sum(1 for r in results if r["passed"]),
            "vetoed": sum(1 for r in results if r.get("vetoed")),
            "relaxed": entities.get("_relaxed") or [],
            "weights": s.scoring_weights,
        },
    }


# ---------------------------------------------------------------- job_recommend
def _score_jobs_for_candidate(entities: dict, s) -> dict:
    """给定候选人，对在招岗位逐个打分并排序（候选人↔岗位匹配度）。"""
    cid = entities.get("candidate_id")
    jobs = entities.get("_jobs") or []
    if not cid:
        return {"scores": [], "score_meta": {"count": 0, "note": "未识别候选人，无法推荐岗位",
                                             "mode": "job_recommend"}}
    if not jobs:
        return {"scores": [], "score_meta": {"count": 0, "note": "当前无在招岗位",
                                             "mode": "job_recommend"}}

    prof = _load_profiles().get(cid)
    cand = _to_candidate(cid, prof, Chunk(id=cid, text=prof.get("resume_text", "") if prof else ""))

    results: list[dict[str, Any]] = []
    for j in jobs:
        job = _job_from_row(j)
        sb = score_candidate(cand, job, None, s)
        results.append({
            "candidate_id": job.job_id,     # 复用 shortlist 结构：此处承载 job_id
            "name": job.title,              # 展示名 = 岗位名
            "title": f"{job.city}｜{job.level}｜{job.min_years:g}-{job.max_years:g}年",
            "total": sb.total,
            "dimensions": sb.dimensions,
            "reasons": sb.reasons,
            "risks": sb.risks,
            "evidence": sb.evidence,
            "vetoed": sb.vetoed,
            "passed": (not sb.vetoed) and sb.total >= s.pass_score,
        })

    results.sort(key=lambda x: x["total"], reverse=True)
    top_n = max(entities.get("top_n", 3), 3)
    results = results[:top_n]
    for i, r in enumerate(results, 1):
        r["rank"] = i

    return {
        "scores": results,
        "score_meta": {
            "count": len(results), "mode": "job_recommend",
            "candidate": cand.name, "pass_score": s.pass_score,
            "passed": sum(1 for r in results if r["passed"]),
            "vetoed": sum(1 for r in results if r.get("vetoed")),
            "weights": s.scoring_weights,
        },
    }


def _job_from_row(j: dict) -> JobRequisition:
    """把 list_jobs 返回的岗位行转成评分用的 JobRequisition。"""
    return JobRequisition(
        job_id=j.get("job_id", ""), title=j.get("title", ""), city=j.get("city", ""),
        level=j.get("level", "3-5"), min_years=float(j.get("min_years", 0)),
        max_years=float(j.get("max_years", 20)),
        salary_range=(int(j.get("salary_min", 0)), int(j.get("salary_max", 10**9))),
        hard_skills=j.get("hard_skills", []), nice_skills=j.get("nice_skills", []),
        domains=j.get("domains", []), education_req=j.get("education_req", "不限"),
        headcount=int(j.get("headcount", 1)), urgency=j.get("urgency", "normal"),
        description=j.get("description", ""),
    )


# ---------------------------------------------------------------- 辅助
def _resolve_job(entities: dict, state: RecruitState) -> JobRequisition:
    j = entities.get("_job")
    if j:
        return JobRequisition(
            job_id=j["job_id"], title=j["title"], city=j.get("city", ""),
            level=j.get("level", "3-5"), min_years=float(j.get("min_years", 0)),
            max_years=float(j.get("max_years", 20)),
            salary_range=(int(j.get("salary_min", 0)), int(j.get("salary_max", 10**9))),
            hard_skills=j.get("hard_skills", []), nice_skills=j.get("nice_skills", []),
            domains=j.get("domains", []), education_req=j.get("education_req", "不限"),
            headcount=int(j.get("headcount", 1)), urgency=j.get("urgency", "normal"),
            description=j.get("description", ""),
        )

    # 虚拟 JD：把检索条件本身当作要求来源，评分全程有据可依
    skills = entities.get("skills", [])
    min_y = float(entities.get("min_years") or 0)
    max_y = float(entities.get("max_years") or (min_y + 3 if min_y else 8))
    sal_min = int(entities.get("min_salary") or 0)
    sal_max = int(entities.get("max_salary") or 0) or max(sal_min + 20000, 80000)
    return JobRequisition(
        job_id="JD-QUERY", title=entities.get("job_title") or "按检索条件匹配",
        city=(entities.get("cities") or ["不限"])[0], level=f"{min_y:.0f}-{max_y:.0f}",
        min_years=min_y, max_years=max_y, salary_range=(sal_min, sal_max),
        hard_skills=skills, nice_skills=[], domains=[],
        education_req=entities.get("education") or "不限", urgency="normal",
        description="由用户检索条件动态构造的虚拟 JD",
    )


_profiles_cache: dict | None = None


def _load_profiles() -> dict:
    global _profiles_cache
    if _profiles_cache is None:
        if PROFILE_CACHE.exists():
            _profiles_cache = json.loads(PROFILE_CACHE.read_text(encoding="utf-8"))
        else:
            _profiles_cache = {}
    return _profiles_cache


def _to_candidate(cid: str, prof: dict | None, chunk: Chunk) -> Candidate:
    if prof:
        return Candidate(
            candidate_id=cid, name=prof["name"], title=prof["title"],
            years=float(prof.get("years", 0)), city=prof.get("city", ""),
            education=prof.get("education", ""), school=prof.get("school", ""),
            major=prof.get("major", ""), skills=prof.get("skills", []),
            domains=prof.get("domains", []), projects=prof.get("projects", []),
            job_intent=prof.get("job_intent", "active"),
            expected_salary=int(prof.get("expected_salary", 0)),
            current_salary=int(prof.get("current_salary", 0)),
            availability_days=int(prof.get("availability_days", 30)),
            stability_years_per_role=float(prof.get("stability_years_per_role", 0)),
            job_changes_3y=int(prof.get("job_changes_3y", 1)),
            interview=prof.get("interview", {}) or {},
            highlights=prof.get("highlights", []), risks=prof.get("known_risks", []),
            resume_text=prof.get("resume_text", ""), tags=prof.get("tags", []),
        )
    md = chunk.metadata
    return Candidate(
        candidate_id=cid, name=md.get("name", cid), title=md.get("title", ""),
        years=float(md.get("years", 0)), city=md.get("city", ""),
        education=md.get("education", ""), school=md.get("school", ""),
        skills=md.get("skills", []), domains=md.get("domains", []),
        job_intent=md.get("job_intent", "active"),
        expected_salary=int(md.get("expected_salary", 0)),
        availability_days=int(md.get("availability_days", 30)),
        job_changes_3y=int(md.get("job_changes_3y", 1)),
        interview={"score": md.get("interview_score", 0), "rounds": md.get("interview_rounds", 0)},
        resume_text=chunk.text, tags=md.get("tags", []),
    )


def _dump(c: Chunk) -> dict:
    return {
        "id": c.id, "text": c.text, "score": c.score, "metadata": c.metadata,
        "dense_rank": c.dense_rank, "sparse_rank": c.sparse_rank,
        "rerank_score": c.rerank_score, "evidence": c.evidence,
    }


def _load(d: dict) -> Chunk:
    return Chunk(
        id=d["id"], text=d["text"], score=d.get("score", 0.0),
        metadata=d.get("metadata", {}), dense_rank=d.get("dense_rank"),
        sparse_rank=d.get("sparse_rank"), rerank_score=d.get("rerank_score"),
        evidence=list(d.get("evidence", [])),
    )

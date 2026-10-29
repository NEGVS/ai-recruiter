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

from ..constraints import relax_soft_constraints
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

    # Reflection 提出「放宽」时，只允许放宽**软约束（年限）**。
    # 城市/薪资/学历/意向是用户的商业约束，系统无权为凑数而放开——
    # 早前版本会把 max_salary 一起放大 15%，导致「60k 以内」变成 79k，
    # 违反硬约束且答案里还把它当成"筛选条件"展示。
    ref = state.get("reflection") or {}
    relax_hint = " ".join(ref.get("next_actions") or []) + " ".join(ref.get("issues") or [])
    if "放宽" in relax_hint:
        entities, changed = relax_soft_constraints(entities)
        if changed:
            log.info("Reflection 第%s轮放宽软约束：%s", ref.get("round"), changed)

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

    # 用户明确说了「找 N 个」就按 N 返回；没说才用默认 5。
    # 早前用 max(...,5) 会让「找 2 个」返回 3 个——擅自扩容与擅自放宽同属越权。
    profiles = _load_profiles()
    top_n = _requested_top_n(entities, default=5)
    # 硬约束：院校层次 / 到岗时效。这两类条件在检索阶段无法表达（向量库里
    # 没有 school_tier 字段），但又必须真正把不合格的人剔除——只做扣分等于没筛。
    from ..intent_rules import school_matches
    school_tier = entities.get("_school_tier")
    avail_max = entities.get("_availability_max_days")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in chunks:
        cid = c.candidate_id
        if cid in seen:
            continue
        seen.add(cid)
        prof = profiles.get(cid) or {}
        if school_tier and not school_matches(str(prof.get("school", "")), school_tier):
            continue
        if avail_max is not None and int(prof.get("availability_days", 30) or 30) > int(avail_max):
            continue
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
        if len(results) >= max(top_n * 2, top_n):
            break

    results.sort(key=lambda x: x["total"], reverse=True)

    # 评估指定候选人：被评估的人必须在结果里。
    # 早前「评估郑凯和 AI平台工程师 的匹配度」返回的是范晓东/李娜/郑凯（郑凯排第 3），
    # 「评估张伟和后端岗位」更是完全没有张伟——用户问的是「这个人行不行」，
    # 答案却在评别人，这在面试里是硬伤。
    if intent == "candidate_evaluate" and entities.get("candidate_id"):
        target_id = str(entities["candidate_id"])
        hit = next((r for r in results if r["candidate_id"] == target_id), None)
        if hit is None:
            prof = profiles.get(target_id)
            chunk = Chunk(id=target_id, text=(prof or {}).get("resume_text", ""),
                          metadata={k: v for k, v in (prof or {}).items()})
            cand = _to_candidate(target_id, prof, chunk)
            sb = score_candidate(cand, job, chunk, s)
            hit = {
                "candidate_id": target_id, "name": cand.name, "title": cand.title,
                "total": sb.total, "dimensions": sb.dimensions, "reasons": sb.reasons,
                "risks": sb.risks, "evidence": sb.evidence, "vetoed": sb.vetoed,
                "passed": (not sb.vetoed) and sb.total >= s.pass_score,
                "_pinned": True,
            }
            results.append(hit)
        else:
            hit["_pinned"] = True
        # 被评估的人固定排第一：用户问的是「这个人行不行」，不是「谁更合适」。
        # 其余候选人是参照系，按分数降序排在后面。
        results.sort(key=lambda x: (0, x.get("rank", 0)) if x.get("_pinned") else (1, -x["total"]))
        results = results[:max(top_n, 1)]
    else:
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
    """给定候选人，对在招岗位逐个打分并排序（候选人↔岗位匹配度）。

    两种入口：
      1. HR 视角：指定了候选人（姓名/ID），从库里读档案；
      2. 求职者视角：「我 6 年经验在北京，找什么岗位合适」——没有候选人档案，
         用用户自述的 技能/年限/城市/期望薪资 构造一个虚拟候选人再匹配岗位。
         早前这种情况直接报「未识别候选人」，求职者根本用不了这个系统。
    """
    cid = entities.get("candidate_id")
    jobs = entities.get("_jobs") or []
    if not jobs:
        return {"scores": [], "score_meta": {"count": 0, "note": "当前无在招岗位",
                                             "mode": "job_recommend"}}

    if cid:
        prof = _load_profiles().get(cid)
        cand = _to_candidate(cid, prof, Chunk(id=cid, text=prof.get("resume_text", "") if prof else ""))
        who = cand.name
    else:
        cand = _candidate_from_self(entities)
        if cand is None:
            # 只给了城市/预算（「北京月薪 5 万以上的岗位」）——用户要的是**岗位清单**，
            # 不是岗位匹配打分。直接按条件列岗位，别去要「技能和年限」。
            if entities.get("cities") or entities.get("min_salary") or entities.get("max_salary"):
                return _list_jobs_by_condition(jobs, entities, s)
            return {"scores": [], "score_meta": {
                "count": 0, "mode": "job_recommend", "candidate": "你",
                "note": "请补充你的技能方向、工作年限和城市（例如：6 年 Java，北京）"}}
        who = "你（按自述条件匹配）"

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
    top_n = _requested_top_n(entities, default=3)
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
def _requested_top_n(entities: dict, default: int) -> int:
    """取用户显式要求的人数；未指定时才用默认值。

    注意不能写成 `max(top_n, 5)`：用户说「找 2 个」就该给 2 个，
    系统擅自扩容会让答案与用户预期不符（也容易被面试官抓住）。
    """
    raw = entities.get("top_n")
    if raw in (None, "", 0):
        return default
    try:
        return max(1, min(30, int(raw)))
    except (TypeError, ValueError):
        return default


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
    # 用户只说方向（「找 3 个后端」）时 skills 为空，这里用方向词展开出的代表技能
    # 作为硬技能——否则硬技能维度没有要求，任何人都是满分，「后端」就变成了「谁分高给谁」。
    skills = entities.get("skills") or entities.get("_direction_skills") or []
    min_y = float(entities.get("min_years") or 0)
    max_y = float(entities.get("max_years") or (min_y + 3 if min_y else 8))
    sal_min = int(entities.get("min_salary") or 0)
    sal_max = int(entities.get("max_salary") or 0) or max(sal_min + 20000, 80000)
    return JobRequisition(
        job_id="JD-QUERY",
        title=entities.get("job_title") or entities.get("_direction_label") or "按检索条件匹配",
        city=(entities.get("cities") or ["不限"])[0], level=f"{min_y:.0f}-{max_y:.0f}",
        min_years=min_y, max_years=max_y, salary_range=(sal_min, sal_max),
        hard_skills=skills, nice_skills=[], domains=[],
        education_req=entities.get("education") or "不限", urgency="normal",
        description="由用户检索条件动态构造的虚拟 JD",
    )


def _list_jobs_by_condition(jobs: list[dict], entities: dict, s) -> dict:
    """按城市/预算列出岗位（不做候选人匹配打分）。

    用于求职者只给了「城市 + 预算」的情况——他还没说自己会什么，
    强行要求「技能+年限」等于答非所问。薪资在实体层是年薪，岗位表是月薪。
    """
    city = (entities.get("cities") or [None])[0]
    min_m = (entities.get("min_salary") or 0) / 12 if entities.get("min_salary") else 0
    max_m = (entities.get("max_salary") or 0) / 12 if entities.get("max_salary") else 0

    hits = []
    for j in jobs:
        if city and j.get("city") != city:
            continue
        if min_m and float(j.get("salary_max", 0)) < min_m:
            continue
        if max_m and float(j.get("salary_min", 0)) > max_m:
            continue
        hits.append(j)

    results = []
    for j in hits:
        results.append({
            "candidate_id": j.get("job_id", ""),
            "name": j.get("title", ""),
            "title": f"{j.get('city', '')}｜{j.get('level', '')}｜"
                     f"{j.get('salary_min', 0)//1000:.0f}-{j.get('salary_max', 0)//1000:.0f}K",
            "total": 0.0,
            "dimensions": {}, "reasons": [], "risks": [],
            "evidence": [f"剩余 HC {j.get('headcount', '-')}"],
            "vetoed": False, "passed": True,
        })
    for i, r in enumerate(results, 1):
        r["rank"] = i
    return {
        "scores": results,
        "score_meta": {
            "count": len(results), "mode": "job_list", "candidate": "你",
            "note": f"按{'城市 ' + city if city else ''}"
                    f"{' ｜ 预算 ≥ ' + str(int(min_m/1000)) + 'K/月' if min_m else ''}"
                    f"{' ｜ 预算 ≤ ' + str(int(max_m/1000)) + 'K/月' if max_m else ''} 筛选",
            "pass_score": s.pass_score,
            "passed": len(results), "vetoed": 0, "weights": s.scoring_weights,
        },
    }


_profiles_cache: dict | None = None


def _candidate_from_self(entities: dict) -> Candidate | None:
    """求职者自述 → 虚拟候选人。

    只要求「技能方向」或「年限」至少有一项，否则连岗位方向都判断不了，
    交给调用方反问用户。城市/期望薪资是加分项，缺失不影响匹配。
    """
    # 方向词（「做数据开发」「后端方向」）也要算作技能，否则虚拟候选人没有任何技能，
    # 所有岗位都会因「硬技能零命中」被一票否决。
    skills = [s for s in ((entities.get("skills") or []) + (entities.get("_direction_skills") or [])) if s]
    years = entities.get("min_years")
    if not skills and years is None:
        return None
    city = (entities.get("cities") or [""])[0]
    min_y = float(years or 0)
    return Candidate(
        candidate_id="SELF",
        name="你",
        title="、".join(skills[:3]) if skills else "自述候选人",
        years=min_y,
        city=city,
        education=str(entities.get("education") or ""),
        skills=skills,
        job_intent=str(entities.get("job_intent") or "active"),
        expected_salary=int(entities.get("expected_salary") or 0),
        availability_days=15,
        resume_text=" ".join(skills),
    )


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

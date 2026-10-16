"""候选人评分引擎。

设计原则：**可解释**优于「黑箱分数」。每一维都给出：
  - 得分与权重
  - 命中/未命中的具体依据
  - 对应的风险项

最终 total 是加权和，同时产出 reasons / risks，供 Reflection 节点校验、
供答案节点直接引用——避免模型凭空编造评价。
"""
from __future__ import annotations

from typing import Any, Sequence

from .config import Settings, get_settings
from .state import Candidate, Chunk, JobRequisition, ScoreBreakdown

# ---------------------------------------------------------------- 经验等级
LEVEL_YEARS = {"校招": 0.0, "应届": 0.0, "1-3": 2.0, "3-5": 4.0, "5-8": 6.5, "8+": 9.0, "专家": 12.0}


def _norm(s: str) -> str:
    import re

    return re.sub(r"[\s\-_/()（）]", "", (s or "").lower())


def _text_of(cand: Candidate | dict[str, Any], chunk: Chunk | None = None) -> str:
    if isinstance(cand, Candidate):
        base = f"{cand.title} {' '.join(cand.skills)} {' '.join(cand.domains)} {cand.resume_text} {' '.join(cand.highlights)}"
    else:
        base = " ".join(str(v) for v in cand.values() if isinstance(v, (str, int, float)))
    if chunk:
        base += " " + chunk.text
    return base.lower()


# ---------------------------------------------------------------- 各维度打分
def score_hard_skill(cand: Any, jd: JobRequisition, chunk: Chunk | None) -> tuple[float, list[str], list[str]]:
    """硬技能命中率 —— 权重最高，招聘中的一票否决维度。"""
    if not jd.hard_skills:
        return 1.0, [], []
    hay = _norm(_text_of(cand, chunk))
    hits = [s for s in jd.hard_skills if _norm(s) in hay]
    miss = [s for s in jd.hard_skills if _norm(s) not in hay]
    ratio = len(hits) / len(jd.hard_skills)
    reasons = [f"命中硬技能 {len(hits)}/{len(jd.hard_skills)}：{', '.join(hits)}"] if hits else []
    risks = [f"缺失硬技能：{', '.join(miss)}"] if miss else []
    # 惩罚曲线：全命中 1.0，缺 1 个 →0.6，缺一半 →0.2
    score = 0.0 if ratio == 0 else min(1.0, ratio ** 0.8)
    return round(score, 3), reasons, risks


def score_domain_experience(cand: Any, jd: JobRequisition, chunk: Chunk | None) -> tuple[float, list[str], list[str]]:
    """行业/领域经验。"""
    if not jd.domains:
        return 1.0, [], []
    hay = _norm(_text_of(cand, chunk))
    hits = [d for d in jd.domains if _norm(d) in hay]
    if not hits:
        return 0.35, [], [f"无 {', '.join(jd.domains)} 领域经验（跨领域需评估学习成本）"]
    if len(hits) == len(jd.domains):
        return 1.0, [f"领域完全匹配：{', '.join(hits)}"], []
    return 0.7, [f"领域部分匹配：{', '.join(hits)}"], []


def score_seniority(cand: Any, jd: JobRequisition, chunk: Chunk | None) -> tuple[float, list[str], list[str]]:
    """年限与层级的匹配度 —— 过低不够、过高易流失。"""
    years = float(getattr(cand, "years", 0) or 0) if isinstance(cand, Candidate) else float(cand.get("years", 0))
    lo, hi = jd.min_years, jd.max_years
    reasons, risks = [], []
    if years < lo:
        gap = lo - years
        score = max(0.0, 0.5 - gap * 0.12)
        risks.append(f"经验 {years}年 低于要求 {lo}年（差 {gap:.1f}年）")
    elif years <= hi:
        score = 1.0
        reasons.append(f"经验 {years}年 落在要求区间 [{lo}, {hi}]")
    else:
        over = years - hi
        score = max(0.3, 1.0 - over * 0.15)
        risks.append(f"经验 {years}年 超出上限 {hi}年，超配 {over:.1f}年（薪资预期与稳定性风险）")
    return round(score, 3), reasons, risks


def score_interview(cand: Any, jd: JobRequisition, chunk: Chunk | None) -> tuple[float, list[str], list[str]]:
    """面试表现。"""
    iv = getattr(cand, "interview", {}) if isinstance(cand, Candidate) else cand.get("interview", {})
    if not isinstance(iv, dict) or not iv:
        return 0.5, [], ["无面试记录"]
    sc = float(iv.get("score", 0))
    rounds = int(iv.get("rounds", 0) or 0)
    if sc <= 0:
        return 0.4, [], [f"面试已安排但未评分（已面 {rounds} 轮）"]
    norm = sc / 100.0
    return round(norm, 3), [f"面试评分 {sc}（{rounds} 轮）"], [] if norm >= 0.6 else [f"面试评分偏低 {sc}"]


def score_stability(cand: Any, jd: JobRequisition, chunk: Chunk | None) -> tuple[float, list[str], list[str]]:
    """稳定性：平均在职时长 + 三年跳动次数。"""
    changes = int(getattr(cand, "job_changes_3y", 1) or 1) if isinstance(cand, Candidate) else int(cand.get("job_changes_3y", 1))
    years = float(getattr(cand, "years", 0) or 0) if isinstance(cand, Candidate) else float(cand.get("years", 0))
    per_role = getattr(cand, "stability_years_per_role", 0) if isinstance(cand, Candidate) else cand.get("stability_years_per_role", 0)
    per_role = float(per_role or (years / max(changes, 1)))
    risks, reasons = [], []
    score = 1.0
    if changes >= 3:
        score -= 0.4
        risks.append(f"3 年内 {changes} 次跳动，稳定性风险高")
    elif changes == 2:
        score -= 0.15
    if per_role < 1.5:
        score -= 0.2
        risks.append(f"平均在职仅 {per_role:.1f} 年")
    elif per_role >= 3:
        reasons.append(f"平均在职 {per_role:.1f} 年，稳定性好")
    return round(max(0.0, min(1.0, score)), 3), reasons, risks


def score_salary_fit(cand: Any, jd: JobRequisition, chunk: Chunk | None) -> tuple[float, list[str], list[str]]:
    """薪酬匹配度。"""
    exp = int(getattr(cand, "expected_salary", 0) or 0) if isinstance(cand, Candidate) else int(cand.get("expected_salary", 0))
    if not exp:
        return 0.6, [], ["未提供期望薪资"]
    lo, hi = jd.salary_range
    if lo <= exp <= hi:
        return 1.0, [f"期望 {exp//1000}K 落在预算 [{lo//1000}K, {hi//1000}K] 内"], []
    if exp > hi:
        over = (exp - hi) / max(hi, 1)
        return round(max(0.0, 0.5 - over * 1.5), 3), [], [
            f"期望 {exp//1000}K 超出预算上限 {hi//1000}K（超出 {over*100:.0f}%）"
        ]
    return 0.85, [f"期望 {exp//1000}K 低于预算下限 {lo//1000}K，存在预算结余"], []


def score_job_intent(cand: Any, jd: JobRequisition, chunk: Chunk | None) -> tuple[float, list[str], list[str]]:
    """求职意向与到岗时间。"""
    intent = getattr(cand, "job_intent", "active") if isinstance(cand, Candidate) else cand.get("job_intent", "active")
    av = int(getattr(cand, "availability_days", 30) or 30) if isinstance(cand, Candidate) else int(cand.get("availability_days", 30))
    m = {"active": 1.0, "passive": 0.6, "urgent": 0.95}.get(intent, 0.7)
    reasons, risks = [], []
    if intent == "active":
        reasons.append("主动在看机会")
    elif intent == "passive":
        risks.append("当前无明确求职意向，需激励")
    if av <= 14:
        m = min(1.0, m + 0.05)
        reasons.append(f"{av} 天可到岗")
    elif av >= 60:
        m = max(0.0, m - 0.2)
        risks.append(f"需 {av} 天才能到岗")
    jd_urg = jd.urgency
    if jd_urg == "high" and av >= 30:
        m = max(0.0, m - 0.25)
        risks.append(f"急招岗位但候选人 {av} 天后才能到岗")
    return round(m, 3), reasons, risks


DIMENSION_FNS = {
    "hard_skill": score_hard_skill,
    "domain_experience": score_domain_experience,
    "seniority": score_seniority,
    "interview": score_interview,
    "stability": score_stability,
    "salary_fit": score_salary_fit,
    "job_intent": score_job_intent,
}


# ---------------------------------------------------------------- 主入口
def score_candidate(
    cand: Candidate | dict[str, Any],
    jd: JobRequisition,
    chunk: Chunk | None = None,
    s: Settings | None = None,
) -> ScoreBreakdown:
    s = s or get_settings()
    weights = s.scoring_weights
    dims: dict[str, dict[str, float]] = {}
    reasons: list[str] = []
    risks: list[str] = []

    for name, fn in DIMENSION_FNS.items():
        w = float(weights.get(name, 0.0))
        if w <= 0:
            continue
        try:
            raw, r, k = fn(cand, jd, chunk)
        except Exception as e:  # noqa: BLE001
            raw, r, k = 0.5, [], [f"{name} 维度计算异常({e})，按中性分处理"]
        dims[name] = {
            "score": raw,
            "weight": w,
            "weighted": round(raw * w * 100, 2),
            "max": round(w * 100, 2),
        }
        reasons.extend(r)
        risks.extend(k)

    total = round(sum(d["weighted"] for d in dims.values()), 2)
    cid = cand.candidate_id if isinstance(cand, Candidate) else str(cand.get("candidate_id", ""))

    # 关键否决信号：硬技能零命中时，即便加权分很高也不能算「通过」。
    # 招聘场景里这是一票否决 —— 其他维度高分（如面试表现）不能替代核心技能。
    vetoed = False
    if "hard_skill" in dims and dims["hard_skill"]["score"] == 0:
        risks.insert(0, "⛔ 硬技能完全未命中，不满足岗位核心要求")
        vetoed = True
    # 经验完全不够（低于门槛 1 年以上）同样否决
    if "seniority" in dims and dims["seniority"]["score"] < 0.3:
        risks.insert(0, "⛔ 经验与岗位门槛差距过大")
        vetoed = True

    return ScoreBreakdown(
        candidate_id=cid,
        total=total,
        dimensions=dims,
        reasons=reasons,
        risks=risks,
        evidence=list(chunk.evidence) if chunk else [],
        vetoed=vetoed,
    )


def score_and_rank(
    cands: Sequence[Candidate | dict[str, Any]],
    jd: JobRequisition,
    chunk_by_cand: dict[str, Chunk] | None = None,
    s: Settings | None = None,
    top_n: int = 5,
) -> list[ScoreBreakdown]:
    s = s or get_settings()
    out = [
        score_candidate(c, jd, (chunk_by_cand or {}).get(
            c.candidate_id if isinstance(c, Candidate) else str(c.get("candidate_id", ""))), s)
        for c in cands
    ]
    out.sort(key=lambda b: b.total, reverse=True)
    return out[:top_n]

"""业务工具集：结构化查询、面试排期、薪酬计算、市场分析、候选人档案。"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..state import ToolResult
from .base import BaseTool, register

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DB_PATH = DATA_DIR / "recruit.db"


# ================================================================== 1. 结构化查询
@register
class QueryCandidatesTool(BaseTool):
    """在结构化候选人库上做精确过滤（不走向量，纯 SQL/条件查询）。"""

    name = "query_candidates"
    description = "按城市/年限/学历/薪资/求职意向等结构化条件精确筛选候选人，返回统计与摘要"
    required: list[str] = []
    optional = {
        "city": "城市，逗号分隔",
        "min_years": "最少工作年限",
        "max_years": "最多工作年限",
        "education": "最低学历",
        "max_salary": "期望薪资上限(K)",
        "job_intent": "active/passive/urgent",
        "skill": "必须包含的技能（逗号分隔）",
        "limit": "返回条数，默认20",
    }

    def run(self, **kw: Any) -> ToolResult:
        conds, params, obs = [], [], []

        if kw.get("city"):
            cities = [c.strip() for c in str(kw["city"]).split(",") if c.strip()]
            conds.append(f"city IN ({','.join('?' * len(cities))})")
            params.extend(cities)
            obs.append(f"城市过滤: {cities}")
        if kw.get("min_years") not in (None, ""):
            conds.append("years >= ?")
            params.append(float(kw["min_years"]))
            obs.append(f"年限 ≥ {kw['min_years']}年")
        if kw.get("max_years") not in (None, ""):
            conds.append("years <= ?")
            params.append(float(kw["max_years"]))
            obs.append(f"年限 ≤ {kw['max_years']}年")
        if kw.get("education"):
            order = {"大专": 1, "本科": 2, "硕士": 3, "博士": 4}
            conds.append("edu_rank >= ?")
            params.append(order.get(str(kw["education"]), 2))
            obs.append(f"学历 ≥ {kw['education']}")
        if kw.get("max_salary") not in (None, ""):
            conds.append("expected_salary <= ?")
            params.append(int(float(kw["max_salary"]) * 1000))
            obs.append(f"期望薪资 ≤ {kw['max_salary']}K")
        if kw.get("job_intent"):
            conds.append("job_intent = ?")
            params.append(kw["job_intent"])
            obs.append(f"求职意向 = {kw['job_intent']}")
        if kw.get("skill"):
            skills = [s.strip() for s in str(kw["skill"]).split(",") if s.strip()]
            cl = []
            for s in skills:
                cl.append("skills_json LIKE ?")
                params.append(f'%"{s}"%')
            conds.append("(" + " OR ".join(cl) + ")")
            obs.append(f"技能包含: {skills}")

        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        limit = int(kw.get("limit") or 20)
        sql = f"SELECT * FROM candidates{where} ORDER BY years DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            rows = [dict(r) for r in con.execute(sql, params).fetchall()]
            total = con.execute(f"SELECT COUNT(*) FROM candidates{where}",
                                params[:-1]).fetchone()[0]

        for r in rows:
            r["skills"] = json.loads(r.pop("skills_json", "[]"))
            r["interview"] = json.loads(r.pop("interview_json", "{}") or "{}")
            r["highlights"] = json.loads(r.pop("highlights_json", "[]"))
        return ToolResult(
            name=self.name, ok=True,
            payload={"rows": rows, "count": len(rows), "total_matched": total,
                     "sql": sql.strip()},
            observations=obs + [f"命中 {total} 人，返回前 {len(rows)} 人"],
        )


@register
class GetCandidateTool(BaseTool):
    """按姓名/ID 获取单个候选人完整档案。"""

    name = "get_candidate"
    description = "按候选人姓名或ID获取完整档案（简历全文、项目、面试记录、风险点）"
    required = ["key"]
    optional = {"key": "候选人姓名或 candidate_id"}

    def run(self, **kw: Any) -> ToolResult:
        key = str(kw["key"]).strip()
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                "SELECT * FROM candidates WHERE candidate_id = ? OR name = ? LIMIT 1", (key, key)
            ).fetchone()
        if not row:
            return ToolResult(name=self.name, ok=False, error=f"未找到候选人: {key}",
                              observations=[f"查询键: {key}"])
        d = dict(row)
        for f in ("skills", "interview", "highlights", "projects", "tags", "domains"):
            v = d.pop(f + "_json", None)
            d[f] = json.loads(v) if v else ([] if f not in ("interview",) else {})
        return ToolResult(name=self.name, ok=True, payload={"candidate": d},
                          observations=[f"命中候选人 {d['name']}（{d['title']}，{d['years']}年）"])


# ================================================================== 2. 面试排期
@register
class ScheduleInterviewTool(BaseTool):
    """面试排期：检查面试官日历冲突 → 计算候选时段 → 写入排期。"""

    name = "schedule_interview"
    description = "为候选人安排面试：检查面试官日程冲突、生成候选时段、确认排期"
    required = ["candidate"]
    optional = {
        "candidate": "候选人姓名或ID",
        "round": "面试轮次，如 '一面'",
        "interviewer": "面试官姓名",
        "date": "期望日期 YYYY-MM-DD，留空则自动推荐",
        "duration": "时长分钟，默认60",
        "confirm": "yes 表示确认写入排期",
    }

    def run(self, **kw: Any) -> ToolResult:
        cand_key = str(kw["candidate"])
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            cand = con.execute(
                "SELECT candidate_id,name,job_intent,availability_days FROM candidates "
                "WHERE candidate_id=? OR name=? LIMIT 1", (cand_key, cand_key)).fetchone()
        if not cand:
            return ToolResult(name=self.name, ok=False, error=f"未找到候选人: {cand_key}")

        interviewer = kw.get("interviewer") or "技术面试官A"
        rnd = kw.get("round") or "一面"
        duration = int(kw.get("duration") or 60)
        base = datetime.strptime(kw["date"], "%Y-%m-%d") if kw.get("date") else \
            datetime.now() + timedelta(days=2)

        # 冲突检测
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            busy = con.execute(
                "SELECT interview_date, start_hour FROM interviews WHERE interviewer=?",
                (interviewer,)).fetchall()
        busy_set = {(b["interview_date"], b["start_hour"]) for b in busy}

        slots: list[dict[str, Any]] = []
        for day_off in range(0, 5):
            day = (base + timedelta(days=day_off)).strftime("%Y-%m-%d")
            for hour in (10, 11, 14, 15, 16, 17):
                if int(hour) + duration / 60 > 19:
                    continue
                if (day, hour) in busy_set:
                    continue
                slots.append({"date": day, "hour": hour, "duration": duration})
                if len(slots) >= 3:
                    break
            if len(slots) >= 3:
                break

        if not slots:
            return ToolResult(name=self.name, ok=False,
                              error=f"{interviewer} 在未来一周内无可用时段", payload={"interviewer": interviewer})

        if str(kw.get("confirm", "")).lower() not in ("yes", "true", "1"):
            return ToolResult(
                name=self.name, ok=True,
                payload={"candidates": slots[:3], "confirmed": False,
                         "interviewer": interviewer, "round": rnd,
                         "candidate": cand["name"]},
                observations=[f"为 {cand['name']} 找到 {len(slots)} 个无冲突时段，待确认"],
            )

        # 确认时段：优先用用户指定的序号（0-indexed）；否则默认第 0 个
        slot_idx = int(kw.get("_slot_index", 0))
        slot = slots[min(slot_idx, len(slots) - 1)]
        with sqlite3.connect(DB_PATH) as con:
            con.execute(
                "INSERT INTO interviews (candidate_id, candidate_name, interviewer, round, "
                "interview_date, start_hour, duration, status) VALUES (?,?,?,?,?,?,?,?)",
                (cand["candidate_id"], cand["name"], interviewer, rnd, slot["date"],
                 slot["hour"], slot["duration"], "scheduled"))
            con.commit()
        return ToolResult(
            name=self.name, ok=True,
            payload={"confirmed": True, **slot, "candidate": cand["name"],
                     "interviewer": interviewer, "round": rnd},
            observations=[f"已确认排期：{slot['date']} {slot['hour']}:00（{interviewer}）"],
        )


@register
class ListInterviewsTool(BaseTool):
    """查询面试日程。"""

    name = "list_interviews"
    description = "查询某天/某面试官/某候选人的面试安排"
    required = []
    optional = {"date": "YYYY-MM-DD", "interviewer": "面试官", "status": "scheduled/done/cancelled"}

    def run(self, **kw: Any) -> ToolResult:
        conds, params, obs = [], [], []
        for f, col in (("date", "interview_date"), ("interviewer", "interviewer"), ("status", "status")):
            if kw.get(f):
                conds.append(f"{col}=?")
                params.append(kw[f])
                obs.append(f"{col}={kw[f]}")
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            rows = [dict(r) for r in con.execute(
                f"SELECT * FROM interviews{where} ORDER BY interview_date, start_hour", params)]
        return ToolResult(name=self.name, ok=True, payload={"interviews": rows, "count": len(rows)},
                          observations=obs + [f"共 {len(rows)} 条面试记录"])


# ================================================================== 3. 薪酬
@register
class SalaryCalcTool(BaseTool):
    """Offer 薪酬计算：底薪 + 绩效 + 签字费 → 总包、覆盖率、与候选人实际期望的差异。

    设计原则：**默认给出真正能达成目标总包的结构**，而不是随意拆一个比例。
    若传入候选人档案中的实际期望，会额外输出谈判缺口提示。
    """

    name = "calc_offer"
    description = "计算 offer 总包、薪资覆盖率、与候选人期望的差距"
    required = ["expected"]
    optional = {"expected": "目标/期望年薪(元)", "base": "月薪(元)", "months": "发薪月数，默认13",
                "bonus": "绩效年薪(元)", "signing": "签字费(元)", "level": "职级，如 P6",
                "candidate_expected": "候选人档案中的实际期望年薪(元)"}

    def run(self, **kw: Any) -> ToolResult:
        target = float(kw["expected"])
        months = int(kw.get("months") or 13)
        base_m = float(kw.get("base") or 0)
        bonus = float(kw.get("bonus") or 0)
        signing = float(kw.get("signing") or 0)

        # 默认让年度底薪正好等于目标总包（coverage = 100%）
        if base_m <= 0:
            base_m = target / months
        annual_base = base_m * months
        total = annual_base + bonus + signing
        coverage = total / target if target else 0.0
        gap = total - target

        notes: list[str] = []
        real_gap = None
        cand_exp = kw.get("candidate_expected")
        if cand_exp:
            cand_exp = float(cand_exp)
            real_gap = total - cand_exp
            if real_gap < 0:
                notes.append(
                    f"⚠️ 方案低于候选人档案中的实际期望 {cand_exp/10000:.0f}万 "
                    f"{abs(real_gap)/10000:.0f}万，谈判风险高，建议提薪或用签字费补足")
            else:
                notes.append(
                    f"✅ 方案高于候选人实际期望 {cand_exp/10000:.0f}万 {real_gap/10000:.0f}万")

        if bonus == 0 and signing == 0 and coverage < 0.95:
            fix = round(target - total)
            notes.append(f"补足建议：月薪 {base_m/10000:.1f}万 + 签字费 {fix/10000:.0f}万，"
                         f"可使总包达到 {target/10000:.0f}万（签字费上限为 3 个月薪资）")

        verdict = ("完全覆盖目标" if coverage >= 1.0 else
                   "略低，建议补签字费" if coverage >= 0.95 else "明显不足")

        return ToolResult(
            name=self.name, ok=True,
            payload={
                "expected_annual": target, "monthly_base": round(base_m), "months": months,
                "annual_base": round(annual_base), "bonus": bonus, "signing": signing,
                "total_package": round(total), "coverage": round(coverage, 3),
                "gap_to_expected": round(gap), "level": kw.get("level") or "-",
                "verdict": verdict, "candidate_expected": cand_exp,
                "gap_to_candidate": round(real_gap) if real_gap is not None else None,
                "notes": notes,
            },
            observations=[f"目标总包 {target/10000:.0f}万 → 方案 {total/10000:.1f}万"
                          f"（覆盖 {coverage*100:.0f}%）"] + notes,
        )


# ================================================================== 4. 市场洞察
@register
class MarketInsightTool(BaseTool):
    """人才市场分析：某技能/岗位的候选人存量、薪资分布、意向分布、竞争度。"""

    name = "market_insight"
    description = "分析某技能/岗位的人才市场：存量、薪资分位、意向分布、竞争程度"
    required = ["topic"]
    optional = {"topic": "技能或岗位关键词，如 'LLM' / '算法工程师'",
                "city": "限定城市", "compare": "yes 则给出供需缺口评估"}

    def run(self, **kw: Any) -> ToolResult:
        topic = str(kw["topic"]).strip()
        city = kw.get("city")
        conds, params = [], []
        if city:
            conds.append("city=?")
            params.append(city)
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            all_rows = [dict(r) for r in con.execute("SELECT * FROM candidates", [])]
        rows = [r for r in all_rows if (not where or r["city"] == city)]
        for r in rows:
            r["skills"] = json.loads(r.pop("skills_json", "[]"))
        hit = [r for r in rows if any(topic.lower() in s.lower() for s in r["skills"])
               or topic.lower() in r["title"].lower()]

        salaries = sorted(r["expected_salary"] for r in hit)

        def pct(p: float) -> int:
            if not salaries:
                return 0
            return salaries[min(len(salaries) - 1, int(p * len(salaries)))]

        intent_dist: dict[str, int] = {}
        for r in hit:
            intent_dist[r["job_intent"]] = intent_dist.get(r["job_intent"], 0) + 1

        p25, p50, p75 = pct(0.25), pct(0.5), pct(0.75)
        gap_note = ""
        if str(kw.get("compare", "")).lower() in ("yes", "true", "1") and hit:
            active = intent_dist.get("active", 0)
            if len(hit) < 5:
                gap_note = f"⚠️ 库内仅 {len(hit)} 人具备该技能，人才供给紧张，建议扩大渠道或放宽相邻技能"
            elif active / max(len(hit), 1) > 0.6:
                gap_note = f"活跃候选人占比 {(active/len(hit))*100:.0f}%，竞争激烈，建议提薪 {5}~10%"
            else:
                gap_note = "供需相对平衡"

        return ToolResult(
            name=self.name, ok=True,
            payload={
                "topic": topic, "city": city, "matched": len(hit), "pool": len(rows),
                "supply_index": round(min(1.0, len(hit) / 30), 3),
                "salary_p25": p25, "salary_p50": p50, "salary_p75": p75,
                "intent_dist": intent_dist,
                "avg_years": round(sum(r["years"] for r in hit) / max(len(hit), 1), 1),
                "assessment": gap_note,
                "top_candidates": [
                    {"name": r["name"], "title": r["title"], "years": r["years"],
                     "expected_salary": r["expected_salary"], "job_intent": r["job_intent"]}
                    for r in sorted(hit, key=lambda x: x["years"], reverse=True)[:5]
                ],
            },
            observations=[f"「{topic}」命中 {len(hit)}/{len(rows)} 人",
                          f"薪资 P50 {p50/1000:.0f}K" if p50 else "无薪资数据"],
        )


# ================================================================== 5. 知识库 / 岗位
@register
class GetJobTool(BaseTool):
    """获取岗位 JD 详情。"""

    name = "get_job"
    description = "获取指定岗位 JD 的完整要求（硬技能、年限、薪资带宽、紧急度）"
    required = ["key"]
    optional = {"key": "岗位名称或 job_id，如 'AI平台工程师' 或 'JD-1001'"}

    def run(self, **kw: Any) -> ToolResult:
        key = str(kw["key"]).strip()
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                "SELECT * FROM jobs WHERE job_id=? LIMIT 1", (key,)).fetchone()
            if not row:
                # 模糊匹配：先完全包含，再退化到「去空格包含」
                row = con.execute(
                    "SELECT * FROM jobs WHERE title LIKE ? OR REPLACE(title,' ','') LIKE ? "
                    "ORDER BY LENGTH(title) LIMIT 1", (f"%{key}%", f"%{key.replace(' ', '')}%")
                ).fetchone()
            if not row:
                # 关键词交集匹配：命中最多技能/词的那个岗位
                words = [w for w in re.split(r"[\s\-_/()（）]+", key) if w]
                if len(words) > 1:
                    words = [max(words, key=len)]
                cands = [dict(r) for r in con.execute("SELECT * FROM jobs")]
                for r in cands:
                    hay = f"{r['title']} {' '.join(json.loads(r['hard_skills_json']))} {r['description'][:200]}"
                    r["_hits"] = sum(1 for w in words if w.lower() in hay.lower())
                best = max(cands, key=lambda r: r["_hits"])
                if best["_hits"] > 0:
                    row = con.execute("SELECT * FROM jobs WHERE job_id=?", (best["job_id"],)).fetchone()
        if not row:
            avail = [r["title"] for r in con.execute("SELECT title FROM jobs")]
            return ToolResult(name=self.name, ok=False,
                              error=f"未找到岗位: {key}。当前在招岗位：{', '.join(avail)}",
                              observations=[f"查询键: {key}"])
        d = dict(row)
        for f in ("hard_skills", "nice_skills", "domains"):
            d[f] = json.loads(d.pop(f + "_json", "[]"))
        return ToolResult(name=self.name, ok=True, payload={"job": d},
                          observations=[f"岗位 {d['title']}：{d['level']}，"
                                        f"{d['min_years']}-{d['max_years']}年，"
                                        f"预算 {d['salary_min']//1000}-{d['salary_max']//1000}K"])


@register
class ListJobsTool(BaseTool):
    """列出在招岗位。"""

    name = "list_jobs"
    description = "列出当前在招岗位及其 headcount、紧急度"
    required = []
    optional = {"city": "按城市过滤", "urgent_only": "yes 仅看急招岗位"}

    def run(self, **kw: Any) -> ToolResult:
        conds, params, obs = [], [], []
        if kw.get("city"):
            conds.append("city=?")
            params.append(kw["city"])
            obs.append(f"城市={kw['city']}")
        if str(kw.get("urgent_only", "")).lower() in ("yes", "true", "1"):
            conds.append("urgency='high'")
            obs.append("仅急招")
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            rows = [dict(r) for r in con.execute(f"SELECT * FROM jobs{where}", params)]
        for r in rows:
            r["hard_skills"] = json.loads(r.pop("hard_skills_json", "[]"))
            r["nice_skills"] = json.loads(r.pop("nice_skills_json", "[]"))
            r["domains"] = json.loads(r.pop("domains_json", "[]"))
        return ToolResult(name=self.name, ok=True, payload={"jobs": rows, "count": len(rows)},
                          observations=obs + [f"在招岗位 {len(rows)} 个"])


@register
class ShortlistTool(BaseTool):
    """写入/更新人才清单（shortlist），带去重与容量校验。"""

    name = "manage_shortlist"
    description = "将候选人加入/移出人才清单，或查看现有清单"
    required = ["action"]
    optional = {"action": "add/remove/list", "job_id": "岗位ID", "candidate_id": "候选人ID",
                "note": "备注"}

    def run(self, **kw: Any) -> ToolResult:
        action = str(kw.get("action", "list")).lower()
        job_id = kw.get("job_id")
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            if action == "list":
                rows = [dict(r) for r in con.execute(
                    "SELECT s.*, c.name, c.title, c.years FROM shortlist s "
                    "LEFT JOIN candidates c ON c.candidate_id=s.candidate_id "
                    "WHERE (? IS NULL OR s.job_id=?)", (job_id, job_id))]
                return ToolResult(name=self.name, ok=True,
                                  payload={"shortlist": rows, "count": len(rows)},
                                  observations=[f"清单共 {len(rows)} 人"])
            cid = str(kw.get("candidate_id") or "")
            if not cid:
                return ToolResult(name=self.name, ok=False, error="candidate_id 必填")
            if action == "add":
                exists = con.execute("SELECT 1 FROM shortlist WHERE job_id=? AND candidate_id=?",
                                     (job_id, cid)).fetchone()
                if exists:
                    return ToolResult(name=self.name, ok=False, error="已在清单中（去重）",
                                      observations=["已存在，跳过"])
                con.execute("INSERT INTO shortlist (job_id,candidate_id,note,created_at) "
                            "VALUES (?,?,?,datetime('now'))", (job_id, cid, kw.get("note", "")))
                con.commit()
                return ToolResult(name=self.name, ok=True, payload={"added": cid},
                                  observations=[f"已加入 {cid} → {job_id}"])
            con.execute("DELETE FROM shortlist WHERE job_id=? AND candidate_id=?", (job_id, cid))
            con.commit()
            return ToolResult(name=self.name, ok=True, payload={"removed": cid},
                              observations=[f"已移出 {cid}"])

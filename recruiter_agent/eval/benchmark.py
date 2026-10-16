"""评测集与指标。

衡量的是 Agent 系统的**端到端质量**，而不是单点指标：
  1. 意图准确率   —— 意图识别是否正确
  2. 实体 F1      —— 城市/技能/年限等槽位抽取质量
  3. 检索指标     —— Recall@k / MRR / nDCG（需要标注了相关候选人的查询）
  4. 排序质量     —— NDCG@5 是否把真正匹配的候选人排在前面
  5. 事实一致性   —— 答案中的分数/数字是否与计算结果一致（防幻觉）
  6. 效率         —— 端到端耗时、工具调用次数、反思轮数

运行： python scripts/run_eval.py
"""
from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any


# ================================================================== 评测集
@dataclass
class EvalCase:
    query: str
    expect_intent: str
    expect_entities: dict[str, Any] = field(default_factory=dict)
    # 标注：哪些候选人应当被召回/排在前列
    relevant: list[str] = field(default_factory=list)
    # 标注：哪些候选人绝不应出现（违反硬约束）
    forbidden: list[str] = field(default_factory=list)
    note: str = ""


EVAL_SET: list[EvalCase] = [
    EvalCase(
        query="找 5 个北京 3 年以上 LLM 和 RAG 方向的候选人",
        expect_intent="candidate_search",
        expect_entities={"cities": ["北京"], "min_years": 3.0},
        relevant=["C024", "C002", "C001"],
        forbidden=["C007", "C022"],  # 非北京
        note="标准多条件检索",
    ),
    EvalCase(
        query="找北京 LLM 方向的候选人",
        expect_intent="candidate_search",
        expect_entities={"cities": ["北京"]},
        relevant=["C024", "C002", "C001", "C010"],
        forbidden=["C007"],
    ),
    EvalCase(
        query="评估张伟和 AI平台工程师 的匹配度",
        expect_intent="candidate_evaluate",
        expect_entities={"candidate_name": "张伟"},
        relevant=["C001"],
    ),
    EvalCase(
        query="面试改期需要提前多久通知 HR？",
        expect_intent="policy_qa",
        note="制度问答走 RAG，不应推荐候选人",
    ),
    EvalCase(
        query="北京大模型人才市场行情如何",
        expect_intent="market_insight",
        expect_entities={"cities": ["北京"]},
    ),
    EvalCase(
        query="给候选人张伟测算 offer，总包 60 万",
        expect_intent="offer_negotiation",
        expect_entities={"candidate_name": "张伟"},
    ),
    EvalCase(
        query="你好",
        expect_intent="chitchat",
    ),
    EvalCase(
        query="找 3 个上海 5 年以上 Java 后端候选人，薪资 60k 以内",
        expect_intent="candidate_search",
        expect_entities={"cities": ["上海"], "min_years": 5.0},
        forbidden=["C024", "C001"],  # 非上海
        note="条件苛刻，可能触发反射放宽",
    ),
    EvalCase(
        query="推荐几个后端候选人",
        expect_intent="candidate_search",
        note="信息不足，应澄清或宽松检索",
    ),
    EvalCase(
        query="帮我给张伟安排明天下午的一面",
        expect_intent="interview_schedule",
        expect_entities={"candidate_name": "张伟"},
    ),
    EvalCase(
        query="给张伟推荐几个合适的岗位",
        expect_intent="job_recommend",
        expect_entities={"candidate_name": "张伟"},
        note="以候选人为主体，反向匹配岗位；走 job_recommend 而非 candidate_search",
    ),
]


# ================================================================== 指标
def intent_accuracy(results: list[dict], cases: list[EvalCase]) -> float:
    if not cases:
        return 0.0
    hit = sum(1 for c, r in zip(cases, results) if r.get("intent") == c.expect_intent)
    return hit / len(cases)


def entity_f1(results: list[dict], cases: list[EvalCase]) -> dict[str, float]:
    """按槽位统计精确率/召回率/F1（集合型槽位用 Jaccard，数值型用容差匹配）。"""
    tp = fp = fn = 0
    for c, r in zip(cases, results):
        got = r.get("entities") or {}
        for key, want in (c.expect_entities or {}).items():
            g = got.get(key)
            if isinstance(want, (list, set, tuple)):
                want_set, got_set = set(want), set(g or [])
                tp += len(want_set & got_set)
                fp += len(got_set - want_set)
                fn += len(want_set - got_set)
            else:
                if g is None:
                    fn += 1
                elif isinstance(want, (int, float)) and isinstance(g, (int, float)):
                    if abs(float(g) - float(want)) < 1e-6:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                elif g == want:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}


def retrieval_metrics(results: list[dict], cases: list[EvalCase], k: int = 5) -> dict[str, float]:
    recalls, mrrs, ndcgs = [], [], []
    viol = 0
    total_forbidden = 0
    for c, r in zip(cases, results):
        got_ids = [s.get("candidate_id") for s in (r.get("shortlist") or [])][:k]
        if c.relevant:
            hit = len(set(got_ids) & set(c.relevant))
            recalls.append(hit / len(c.relevant))
            rank = next((i + 1 for i, x in enumerate(got_ids) if x in c.relevant), 0)
            mrrs.append(1.0 / rank if rank else 0.0)
            dcg = sum(1 / _log2(i + 2) for i, x in enumerate(got_ids) if x in c.relevant)
            idcg = sum(1 / _log2(i + 2) for i in range(min(len(c.relevant), k)))
            ndcgs.append(dcg / idcg if idcg else 0.0)
        if c.forbidden:
            total_forbidden += len(c.forbidden)
            viol += len(set(got_ids) & set(c.forbidden))
    return {
        f"recall@{k}": round(statistics.mean(recalls), 3) if recalls else None,
        f"mrr@{k}": round(statistics.mean(mrrs), 3) if mrrs else None,
        f"ndcg@{k}": round(statistics.mean(ndcgs), 3) if ndcgs else None,
        "hard_violation_rate": round(viol / total_forbidden, 3) if total_forbidden else 0.0,
    }


def _log2(x: float) -> float:
    import math

    return math.log2(x) if x > 0 else 1.0


def factual_consistency(results: list[dict]) -> dict[str, float]:
    """答案中出现的「候选人分数」必须与真实评分一致 —— 幻觉检测。"""
    checked = mismatch = 0
    for r in results:
        truth = {s["name"]: float(s["total"]) for s in (r.get("shortlist") or [])}
        ans = r.get("answer") or ""
        for name, total in truth.items():
            if name not in ans:
                continue
            for m in re.finditer(rf"{re.escape(name)}[^\n]{{0,40}}?(\d+(?:\.\d+)?)\s*分", ans):
                checked += 1
                got = float(m.group(1))
                if abs(got - total) > 0.05 and abs(got - total * 100) > 1.0:
                    mismatch += 1
    return {
        "claims_checked": checked,
        "mismatches": mismatch,
        "consistency": round(1 - mismatch / checked, 4) if checked else 1.0,
    }


def efficiency(results: list[dict]) -> dict[str, float]:
    lat = [r.get("total_ms", 0) for r in results]
    tools = [len(r.get("tool_results") or []) for r in results]
    rounds = [len(r.get("reflection_history") or []) for r in results]
    return {
        "latency_p50_ms": int(statistics.median(lat)) if lat else 0,
        "latency_p95_ms": int(sorted(lat)[int(len(lat) * 0.95) - 1]) if lat else 0,
        "avg_tool_calls": round(statistics.mean(tools), 2) if tools else 0,
        "avg_reflection_rounds": round(statistics.mean(rounds), 2) if rounds else 0,
    }


def constraint_compliance(results: list[dict]) -> float:
    """答案是否覆盖了所有被提及的候选人（不得凭空出现未评分的人名）。"""
    ok = 0
    total = 0
    known = {"张伟", "李娜", "王强", "刘洋", "陈静", "赵磊", "孙悦", "周涛", "吴敏", "郑凯",
             "冯雪", "何军", "马晓丽", "高鹏", "林芳", "徐峰", "胡斌", "苏晴", "唐宇",
             "曹雪松", "彭飞", "董洁", "袁磊", "范晓东", "崔天佑"}
    for r in results:
        ans = r.get("answer") or ""
        if not (r.get("shortlist") or []):
            continue
        # job_recommend 的 shortlist 是「岗位」而非候选人，答案里出现的候选人是查询主体（已 get_candidate 载入），
        # 不属于本指标的候选人幻觉范畴，跳过。
        if r.get("intent") == "job_recommend":
            continue
        total += 1
        mentioned = {n for n in known if n in ans}
        scored = {s["name"] for s in r["shortlist"]}
        if mentioned <= scored:
            ok += 1
    return round(ok / total, 3) if total else 1.0


# ================================================================== runner
def run_eval(agent=None, verbose: bool = True) -> dict[str, Any]:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from recruiter_agent import RecruitAgent  # noqa: E402

    # 自举：确保数据与向量索引就绪，使评测可独立运行
    from recruiter_agent.vectorstore import get_store

    if not (ROOT / "data" / "recruit.db").exists():
        from scripts.init_data import build_db
        build_db()
    if get_store().count("candidates") == 0:
        from scripts.index_data import index
        index("memory")

    agent = agent or RecruitAgent()
    results: list[dict] = []
    t0 = time.time()

    for c in EVAL_SET:
        r = agent.run(c.query)
        results.append(r)
        if verbose:
            ok = "✅" if r.get("intent") == c.expect_intent else "❌"
            print(f"{ok} {c.query}")
            print(f"   意图={r.get('intent')} (期望 {c.expect_intent}) | "
                  f"{r.get('total_ms')}ms | 工具 {len(r.get('tool_results') or [])} | "
                  f"候选 {len(r.get('shortlist') or [])}")

    report = {
        "cases": len(EVAL_SET),
        "elapsed_s": round(time.time() - t0, 2),
        "intent_accuracy": round(intent_accuracy(results, EVAL_SET), 3),
        "entity": entity_f1(results, EVAL_SET),
        "retrieval": retrieval_metrics(results, EVAL_SET),
        "factual": factual_consistency(results),
        "constraint_compliance": constraint_compliance(results),
        "efficiency": efficiency(results),
    }
    if verbose:
        print("\n" + "═" * 62)
        print("评测报告")
        print("═" * 62)
        print(f"意图准确率        : {report['intent_accuracy']:.1%}")
        e = report["entity"]
        print(f"实体抽取 P/R/F1   : {e['precision']:.2f} / {e['recall']:.2f} / {e['f1']:.2f}")
        r = report["retrieval"]
        print(f"检索 Recall@5     : {r['recall@5']}")
        print(f"检索 MRR@5        : {r['mrr@5']}")
        print(f"检索 nDCG@5       : {r['ndcg@5']}")
        print(f"硬约束违反率      : {r['hard_violation_rate']:.1%}  (越低越好)")
        f = report["factual"]
        print(f"事实一致性        : {f['consistency']:.1%} "
              f"(校验 {f['claims_checked']} 处数值，{f['mismatches']} 处不符)")
        print(f"无幻觉候选人      : {report['constraint_compliance']:.1%}")
        ef = report["efficiency"]
        print(f"延迟 P50/P95      : {ef['latency_p50_ms']} / {ef['latency_p95_ms']} ms")
        print(f"平均工具调用/反思 : {ef['avg_tool_calls']} / {ef['avg_reflection_rounds']} 轮")
    return report


if __name__ == "__main__":
    run_eval()

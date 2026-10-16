#!/usr/bin/env python
"""AI 招聘 Agent 交互式 Demo。

用法：
    python run_demo.py                          # 交互模式
    python run_demo.py -q "找5个北京3年以上LLM的候选人"   # 单次问答
    python run_demo.py --trace                 # 展示完整执行链路 Trace
    python run_demo.py --json                  # 输出结构化 JSON

注意：首次使用请先执行
    python scripts/init_data.py      # 初始化 SQLite 数据
    python scripts/index_data.py     # 向量入库
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from recruiter_agent import RecruitAgent  # noqa: E402
from recruiter_agent.rag import get_retriever  # noqa: E402
from recruiter_agent.tools.business_tools import DB_PATH  # noqa: E402
from recruiter_agent.vectorstore import get_store  # noqa: E402

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║            AI 招聘 Agent · LangGraph 编排演示                ║
╠══════════════════════════════════════════════════════════════╣
║  流程：意图识别 → 规划 → Tool → RAG → ES/Milvus 混合检索     ║
║        → Rerank → 多维评分 → Reflection → 最终回答           ║
╠══════════════════════════════════════════════════════════════╣
║  试试：                                                          ║
║   · 找 5 个北京 3 年以上 LLM/RAG 方向的候选人                  ║
║   · 评估一下张伟和 AI平台工程师(JD-1001)的匹配度               ║
║   · 北京大模型人才市场行情怎么样                                ║
║   · 面试改期需要提前多久通知？                                  ║
║   · 给张伟测算 offer，总包 60 万                              ║
╚══════════════════════════════════════════════════════════════╝
"""

QUERIES = [
    "找 5 个北京 3 年以上 LLM 和 RAG 方向的候选人",
    "评估张伟和 AI平台工程师(LLM/RAG方向) 这个岗位的匹配度",
    "北京大模型方向的人才市场行情怎么样？供需如何",
    "面试改期需要提前多久通知 HR？",
    "给候选人张伟测算 offer，期望总包 60 万",
    "上海有哪些后端方向的候选人？不考虑 5 年以上的",
    "你好",
]


def preflight() -> None:
    if not DB_PATH.exists():
        print("⚠️  未找到数据库，正在自动初始化…")
        from scripts.init_data import build_db
        build_db()
    store = get_store()
    if store.count("candidates") == 0:
        print("⚠️  向量库为空，正在自动入库（内存后端；生产请先启动 ES/Milvus）…")
        from scripts.index_data import index
        index("memory")
    else:
        print(f"ℹ️  向量库已有 {store.count('candidates')} 条候选人记录（后端：{store.backend}）")


def print_trace(result: dict) -> None:
    print("\n" + "─" * 78)
    print("执行链路 Trace")
    print("─" * 78)
    for i, t in enumerate(result.get("trace", []), 1):
        icon = "✓" if t.get("status") == "ok" else "✗"
        print(f" {i:2d}. {icon} {t['node']:<12} {t['ms']:>5}ms  {t.get('summary', '')}")
    print("─" * 78)
    print(f" 总计 {result['total_ms']}ms | 工具调用 {len(result.get('tool_results') or [])} 次 | "
          f"反思 {len(result.get('reflection_history') or [])} 轮")
    st = result.get("retrieval_stats") or {}
    if isinstance(st, dict) and st:
        print(f" 检索：dense={st.get('dense')} bm25={st.get('bm25')} profile={st.get('profile')} "
              f"→ 融合 {st.get('fused')} | filter={st.get('filters')}")
    if result.get("reranked"):
        print("\n Rerank Top:")
        for c in result["reranked"][:6]:
            print(f"   · {c['name']}｜{c['title']}｜score={c['rerank_score']}"
                  + (f"｜{c['evidence'][0]}" if c.get("evidence") else ""))


def print_scores(result: dict) -> None:
    scores = result.get("shortlist") or []
    if not scores:
        return
    meta = result.get("score_meta") or {}
    print("\n" + "─" * 78)
    print(f"评分明细（岗位：{meta.get('job')}，通过线 {meta.get('pass_score')}）")
    print("─" * 78)
    for s in scores:
        flag = "✅" if s.get("passed") else "⚠️"
        dims = "  ".join(f"{k}={v['score']*100:.0f}" for k, v in (s.get("dimensions") or {}).items())
        print(f" {flag} #{s['rank']} {s['name']:<6} {s['total']:>5}分  {dims}")
        for r in (s.get("reasons") or [])[:2]:
            print(f"      + {r}")
        for r in (s.get("risks") or [])[:2]:
            print(f"      - {r}")


def print_reflection(result: dict) -> None:
    hist = result.get("reflection_history") or []
    if not hist:
        return
    print("\n" + "─" * 78)
    print("Reflection 自我审查")
    print("─" * 78)
    for h in hist:
        print(f" 第{h['round']}轮 → {h['decision']}（自评分 {h['score']}）")
        for i in h.get("issues", [])[:3]:
            print(f"    ⚠️ {i}")


def run_once(agent: RecruitAgent, q: str, show_trace: bool, as_json: bool) -> None:
    result = agent.run(q)
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    print("\n" + "█" * 78)
    print(f"❓ {q}")
    print("█" * 78)
    if show_trace:
        print_trace(result)
    print(result["answer"])
    if not show_trace:
        print_trace(result)
    print_scores(result)
    print_reflection(result)
    if result.get("errors"):
        print("\n⚠️ 异常：" + "; ".join(result["errors"][:3]))


def main() -> None:
    ap = argparse.ArgumentParser(description="AI 招聘 Agent Demo")
    ap.add_argument("-q", "--query", help="单次查询")
    ap.add_argument("--trace", action="store_true", help="显示执行 Trace")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--demo", action="store_true", help="运行内置示例集")
    ap.add_argument("--no-trace", action="store_true", help="隐藏 Trace")
    a = ap.parse_args()

    preflight()
    agent = RecruitAgent()
    show_trace = a.trace and not a.no_trace

    if a.demo:
        for q in QUERIES:
            run_once(agent, q, show_trace, a.json)
            print("\n" + "=" * 78 + "\n")
        return

    if a.query:
        run_once(agent, a.query, show_trace, a.json)
        return

    print(BANNER)
    while True:
        try:
            q = input("\n🗣  你的需求> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 👋")
            break
        if not q:
            continue
        if q in ("exit", "quit", "q"):
            print("再见 👋")
            break
        run_once(agent, q, show_trace, a.json)


if __name__ == "__main__":
    main()

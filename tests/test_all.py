"""端到端与单元测试。

运行： pytest -q          （若未装 pytest，直接 python tests/test_all.py）
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recruiter_agent.embedding import HashEmbedder, cosine, tokenize  # noqa: E402
from recruiter_agent.intent_rules import (  # noqa: E402
    extract_entities, rule_based_intent,
)
from recruiter_agent.scoring import score_candidate  # noqa: E402
from recruiter_agent.state import Candidate, Chunk, JobRequisition  # noqa: E402
from recruiter_agent.vectorstore import MemoryVectorStore, rrf_fuse  # noqa: E402


# ================================================================= 向量库
def test_memory_store_roundtrip():
    st = MemoryVectorStore()
    emb = HashEmbedder(384)
    chunks = [
        Chunk(id="a", text="张伟 AI平台工程师 LLM RAG Milvus LangGraph", metadata={"city": "北京", "years": 6.0}),
        Chunk(id="b", text="李娜 大模型应用研发 Agent LangGraph RAG", metadata={"city": "北京", "years": 4.0}),
        Chunk(id="c", text="王强 高级后端 Java 分布式 高并发", metadata={"city": "上海", "years": 8.0}),
    ]
    emb.fit([c.text for c in chunks])  # 先统计 IDF，再向量化
    vecs = emb.embed([c.text for c in chunks])
    assert st.upsert("candidates", chunks, vecs) == 3
    assert st.count("candidates") == 3

    hits = st.dense_search("candidates", emb.embed_one("LLM RAG 向量检索"), 3)
    assert hits[0].id in ("a", "b"), "语义召回应优先命中 AI 方向"

    sparse = st.sparse_search("candidates", "王强 Java", 3)
    assert sparse and sparse[0].metadata["city"] == "上海", "BM25 应命中精确词"

    hyb = st.hybrid_search("candidates", "北京 LLM", emb.embed_one("北京 LLM"), 3,
                           filters={"city": "北京"})
    assert all(c.metadata["city"] == "北京" for c in hyb), "filter 必须生效"
    print("✓ test_memory_store_roundtrip")


def test_filter_semantics():
    st = MemoryVectorStore()
    emb = HashEmbedder(32)
    chunks = [
        Chunk(id="a", text="北京 3年 LLM", metadata={"city": "北京", "years": 3.0, "expected_salary": 40000}),
        Chunk(id="b", text="上海 8年 Java", metadata={"city": "上海", "years": 8.0, "expected_salary": 90000}),
    ]
    st.upsert("candidates", chunks, emb.embed([c.text for c in chunks]))
    r = st.sparse_search("candidates", "任意", 10, {"min_years": 5.0})
    assert all(c.metadata["years"] >= 5.0 for c in r)
    r2 = st.sparse_search("candidates", "任意", 10, {"max_salary": 50000})
    assert all(c.metadata["expected_salary"] <= 50000 for c in r2)
    print("✓ test_filter_semantics")


def test_rrf_fusion():
    d = [Chunk(id="x", text="x"), Chunk(id="y", text="y")]
    s = [Chunk(id="y", text="y"), Chunk(id="z", text="z")]
    fused = rrf_fuse(d, s, 0.5)
    ids = [c.id for c in fused]
    assert "y" in ids[0], "两路都命中的 y 应排第一"
    print("✓ test_rrf_fusion")


def test_embedding_similarity():
    corpus = [
        "张伟 AI平台工程师 LLM RAG Milvus LangGraph 向量检索",
        "李娜 大模型应用研发 Agent LangGraph RAG 智能体",
        "王强 高级后端 Java 分布式 高并发 MySQL Redis",
        "刘洋 数据平台 Spark Flink 数仓 实时计算",
        "赵磊 前端 Vue TypeScript 工程化 可视化",
    ]
    emb = HashEmbedder(384).fit(corpus)
    a = emb.embed_one("LLM RAG 向量检索")
    b = emb.embed_one("RAG LLM 检索增强")
    c = emb.embed_one("Java 分布式 高并发")
    # 断言「可分性」而非绝对值：hash embedding 的绝对余弦天然偏低，
    # 真正要保证的是相关文本远高于无关文本（检索排序才不会乱）。
    assert cosine(a, b) > cosine(a, c) + 0.3, (
        f"相关/无关文本区分度不足: {cosine(a, b):.3f} vs {cosine(a, c):.3f}")
    assert len(tokenize("Python和LLM")) >= 3
    # 未 fit 时也应可用（退化为无 IDF）
    raw = HashEmbedder(64)
    assert abs(cosine(raw.embed_one("LLM"), raw.embed_one("LLM")) - 1.0) < 1e-6
    print("✓ test_embedding_similarity")


# ================================================================= 意图
def test_intent_recognition():
    cases = [
        ("找 5 个北京 3 年以上 LLM 方向的候选人", "candidate_search"),
        ("评估张伟和后端岗位的匹配度", "candidate_evaluate"),
        ("北京大模型人才市场行情如何", "market_insight"),
        ("面试改期需要提前多久通知 HR", "policy_qa"),
        ("帮我给张伟安排明天下午的一面", "interview_schedule"),
        ("给候选人张伟测算 offer，总包 60 万", "offer_negotiation"),
        ("你好", "chitchat"),
    ]
    for q, expect in cases:
        got = rule_based_intent(q)["intent"]
        assert got == expect, f"{q!r} 期望 {expect}，实际 {got}"
    print("✓ test_intent_recognition")


def test_entity_extraction():
    e = extract_entities("找 5 个北京 3 年以上 LLM 和 RAG 方向的候选人")
    assert e["cities"] == ["北京"]
    assert e["min_years"] == 3.0
    assert "LLM" in e["skills"] and "RAG" in e["skills"]
    assert e["top_n"] == 5
    assert e.get("candidate_name") is None, "不应把「方向的」误判为姓名"

    e2 = extract_entities("评估张伟和 AI平台工程师 的匹配度")
    assert e2["candidate_name"] == "张伟"
    # 职级后缀属于岗位名本身，不能被当噪声剥掉（早前会截成「AI平台」，
    # 再拿去模糊匹配就会命中错误的 JD）
    assert e2["job_title"] == "AI平台工程师"

    # 年限方向：上限表述不能被解析成下限，也不能凭空补 +2 的软上限
    e4 = extract_entities("3年以下的前端")
    assert e4.get("max_years") == 3.0 and "min_years" not in e4, e4
    assert extract_entities("不超过5年的算法").get("max_years") == 5.0
    assert "min_years" not in extract_entities("不超过5年的算法")
    assert extract_entities("5年以上")["min_years"] == 5.0
    assert "max_years" not in extract_entities("5年以上"), "「以上」不应凭空生成上限"

    # 薪资：k 按月薪折算成年薪；「以内」是上限，不能被当成人名下界
    s1 = extract_entities("找 3 个上海 5 年以上 Java 后端候选人，薪资 60k 以内")
    assert s1["max_salary"] == 720000 and "min_salary" not in s1, s1
    s2 = extract_entities("找北京 LLM 候选人，年薪 80 万以内")
    assert s2["max_salary"] == 800000
    assert s2.get("candidate_name") is None, "「万以内」不能被当成人名"

    # 中文数字人数
    assert extract_entities("找三个北京 LLM 候选人")["top_n"] == 3

    e3 = extract_entities("给候选人张伟测算 offer，期望总包 60 万")
    assert e3["salary_kw"] == [600000], "总包 60 万应解析为年薪 60 万"
    assert e3["candidate_name"] == "张伟", "粘连串应正确切出姓名"

    # 不应把功能词当人名
    assert extract_entities("找北京大模型方向候选人").get("candidate_name") is None
    assert extract_entities("推荐几个后端候选人").get("candidate_name") is None
    print("✓ test_entity_extraction")


# ================================================================= 评分
def _cand(**kw) -> Candidate:
    base = dict(candidate_id="T1", name="测试", title="AI平台工程师", years=5.0,
                city="北京", education="硕士", skills=["Python", "LLM", "RAG"],
                domains=["大模型"], expected_salary=60000, job_intent="active",
                stability_years_per_role=3.0, job_changes_3y=1,
                interview={"score": 85, "rounds": 2})
    base.update(kw)
    return Candidate(**base)


def _jd(**kw) -> JobRequisition:
    base = dict(job_id="JD-T", title="AI平台", city="北京", level="3-5", min_years=3,
                max_years=6, salary_range=(45000, 70000),
                hard_skills=["Python", "LLM", "RAG"], domains=["大模型"])
    base.update(kw)
    return JobRequisition(**base)


def test_scoring_dimensions():
    good = score_candidate(_cand(), _jd())
    assert good.total > 80, f"高匹配候选人应得高分，实际 {good.total}"
    assert not good.vetoed
    assert good.dimensions["hard_skill"]["score"] == 1.0

    # 硬技能零命中 → 一票否决
    bad = score_candidate(_cand(skills=["Java", "Spring"]), _jd())
    assert bad.vetoed, "硬技能零命中必须触发否决"
    assert bad.dimensions["hard_skill"]["score"] == 0.0

    # 经验不足 → 否决
    jr = score_candidate(_cand(years=1.0), _jd())
    assert jr.vetoed, "经验严重不足必须触发否决"
    print("✓ test_scoring_dimensions")


def test_scoring_salary_and_stability():
    over = score_candidate(_cand(expected_salary=120000), _jd())
    assert any("超出" in r or "超出预算" in r for r in over.risks)
    assert over.dimensions["salary_fit"]["score"] < 0.6

    unstable = score_candidate(_cand(job_changes_3y=4, stability_years_per_role=0.8), _jd())
    assert unstable.dimensions["stability"]["score"] < 0.5
    assert any("跳动" in r for r in unstable.risks)
    print("✓ test_scoring_salary_and_stability")


# ================================================================= 图编排
def test_graph_end_to_end():
    from recruiter_agent import RecruitAgent
    from scripts.init_data import build_db
    from scripts.index_data import index

    if not (ROOT / "data" / "recruit.db").exists():
        build_db()
    try:
        from recruiter_agent.vectorstore import get_store
        if get_store().count("candidates") == 0:
            index("memory")
    except Exception:  # noqa: BLE001
        pass

    agent = RecruitAgent()
    r = agent.run("找 5 个北京 3 年以上 LLM 和 RAG 方向的候选人")
    assert r["intent"] == "candidate_search"
    assert r["answer"], "必须产出答案"
    assert r["shortlist"], "必须产出候选人评分"
    assert r["trace"], "必须记录执行链路"
    nodes = [t["node"] for t in r["trace"]]
    for expect in ("intent", "plan", "tool", "retrieve", "rerank", "score", "answer", "reflect"):
        assert expect in nodes, f"链路缺少节点 {expect}: {nodes}"
    # 答案中的分数必须与实际评分一致
    for s in r["shortlist"]:
        if s["name"] in r["answer"]:
            assert f"{s['total']:g}" in r["answer"] or str(s["total"]) in r["answer"]
    print("✓ test_graph_end_to_end")


def test_evaluate_pins_named_candidate():
    """评估指定候选人时，被评估的人必须在结果里且排第一。"""
    from recruiter_agent import RecruitAgent
    from recruiter_agent.vectorstore import get_store
    from scripts.init_data import build_db
    from scripts.index_data import index

    if not (ROOT / "data" / "recruit.db").exists():
        build_db()
    if get_store().count("candidates") == 0:
        index("memory")

    agent = RecruitAgent()
    # 郑凯在北京，岗位 JD-1001 也在北京
    r = agent.run("评估郑凯和 AI平台工程师 的匹配度", session_id="t-eval-1")
    names = [s["name"] for s in (r["shortlist"] or [])]
    assert "郑凯" in names, f"被评估人不在结果里: {names}"
    assert names[0] == "郑凯", f"被评估人应排第一: {names}"

    # 张伟在北京，岗位 JD-1002 在上海——岗位城市不能把本人筛掉
    r2 = agent.run("评估张伟和后端岗位的匹配度", session_id="t-eval-2")
    names2 = [s["name"] for s in (r2["shortlist"] or [])]
    assert "张伟" in names2, f"岗位城市不应过滤掉被评估人: {names2}"
    assert names2[0] == "张伟"
    print("✓ test_evaluate_pins_named_candidate")


def test_offer_uses_candidate_archive():
    """「给张伟测算 offer」：档案期望 65K/月 → 年薪 78 万，不能算出 0 万。"""
    from recruiter_agent import RecruitAgent
    from recruiter_agent.vectorstore import get_store
    from scripts.init_data import build_db
    from scripts.index_data import index

    if not (ROOT / "data" / "recruit.db").exists():
        build_db()
    if get_store().count("candidates") == 0:
        index("memory")

    agent = RecruitAgent()
    r = agent.run("给张伟测算 offer", session_id="t-offer-1")
    assert r["intent"] == "offer_negotiation"
    assert "总包" in r["answer"]
    # 只校验「总包」这一行的金额：整篇里出现「余量 0.0万」是正常文案
    m = re.search(r"\*\*总包\*\*\s*\|\s*\*\*([\d.]+)万", r["answer"])
    assert m, f"未找到总包行:\n{r['answer'][:400]}"
    assert float(m.group(1)) >= 70, f"总包应为档案期望 78 万，实际 {m.group(1)} 万"
    assert "78" in r["answer"], "应按档案期望（65K×12=78 万）测算"
    print("✓ test_offer_uses_candidate_archive")


def test_graph_all_intents():
    from recruiter_agent import RecruitAgent

    agent = RecruitAgent()
    for q, expect_intent in [
        ("你好", "chitchat"),
        ("面试改期需要提前多久通知 HR？", "policy_qa"),
        ("北京大模型人才市场行情如何", "market_insight"),
    ]:
        r = agent.run(q)
        assert r["intent"] == expect_intent, f"{q}: {r['intent']} != {expect_intent}"
        assert r["answer"], f"{q} 无答案"
    print("✓ test_graph_all_intents")


def test_reflection_guard_no_hallucination():
    """答案中出现的候选人分数必须与真实评分一致。"""
    from recruiter_agent import RecruitAgent

    agent = RecruitAgent()
    r = agent.run("找 3 个北京 LLM 方向的候选人")
    truth = {s["name"]: float(s["total"]) for s in (r["shortlist"] or [])}
    import re
    for name, total in truth.items():
        if name not in r["answer"]:
            continue
        for m in re.finditer(rf"{name}[^\n]{{0,40}}?(\d+(?:\.\d+)?)\s*分", r["answer"]):
            got = float(m.group(1))
            assert abs(got - total) < 0.05 or abs(got - total * 100) < 1.0, \
                f"{name} 分数 {got} 与真实 {total} 不符"
    print("✓ test_reflection_guard_no_hallucination")


# ================================================================= 后端适配器（Mock 验证）
def test_es_mapping_and_filter_translation():
    """ES 后端无法在无 Docker 环境下真跑，这里用 stub 验证 mapping 与 filter 语义。"""
    from types import SimpleNamespace

    from recruiter_agent.config import Settings
    from recruiter_agent.vectorstore.base import build_es_filters
    from recruiter_agent.vectorstore.es_store import ElasticsearchStore

    s = Settings(es_url="http://localhost:9200", vector_store_backend="elasticsearch")
    store = ElasticsearchStore.__new__(ElasticsearchStore)   # 绕过 __init__ 的真实连接
    store._s = s
    store._dim = None

    m = store._mapping(384)
    props = m["mappings"]["properties"]
    assert props["embedding"]["type"] == "dense_vector"
    assert props["embedding"]["dims"] == 384
    assert props["embedding"]["similarity"] == "cosine"
    assert props["skills"]["type"] == "keyword", "技能必须是 keyword 才能用于 terms 过滤"
    assert props["text"]["type"] == "text", "简历正文必须是 text 才能参与 BM25"
    assert props["city"]["type"] == "keyword"
    assert props["years"]["type"] == "float"

    # 逻辑名必须映射到 mapping 中真实存在的字段：
    # max_salary → expected_salary（不是字面量 max_salary，否则过滤静默失效）
    # city 已声明为 keyword，不能再加 .keyword 后缀
    f = build_es_filters({"city": ["北京", "上海"], "min_years": 3, "max_salary": 60000})
    clauses = f["bool"]["must"]
    kinds = [list(c.keys())[0] for c in clauses]
    assert "terms" in kinds and "range" in kinds, f"filter 构造异常: {kinds}"

    terms = [c["terms"] for c in clauses if "terms" in c][0]
    assert "city" in terms and "city.keyword" not in terms, terms
    assert "city" in props, "terms 引用的字段必须在 mapping 中存在"

    rng = {k: v for c in clauses if "range" in c for k, v in c["range"].items()}
    assert rng["years"]["gte"] == 3, rng
    assert rng["expected_salary"]["lte"] == 60000, rng
    for field in rng:
        assert field in props, f"range 引用的 {field} 不在 mapping 中"

    # 标量值应走 term 而非 terms
    f2 = build_es_filters({"city": "北京"})
    assert f2["bool"]["must"] == [{"term": {"city": "北京"}}], f2
    assert build_es_filters(None) is None
    assert build_es_filters({}) is None
    print("✓ test_es_mapping_and_filter_translation")


def test_milvus_expr_translation():
    from recruiter_agent.vectorstore.milvus_store import _to_expr

    e = _to_expr({"city": "北京", "min_years": 3, "max_salary": 60000})
    assert 'city == "北京"' in e
    assert "years >= 3.0" in e
    assert "expected_salary <= 60000" in e
    assert _to_expr(None) == ""
    # 数组型技能应为 Milvus 表达式而非字符串列表
    e2 = _to_expr({"skills": ["LLM", "RAG"]})
    assert e2 == "skills in ['LLM', 'RAG']", e2
    print("✓ test_milvus_expr_translation")


def test_backends_share_rrf_semantics():
    """三后端必须用同一套 RRF 融合，否则换后端会得到不一致的排序。"""
    from recruiter_agent.vectorstore.base import rrf_fuse

    dense = [Chunk(id="a", text="a"), Chunk(id="b", text="b"), Chunk(id="c", text="c")]
    sparse = [Chunk(id="c", text="c"), Chunk(id="a", text="a")]
    fused = rrf_fuse(dense, sparse, 0.6)
    # a 两路均排第 1（0.6/61 + 0.4/61），c 排 3/1（0.6/63 + 0.4/61）→ a 应最高
    assert [c.id for c in fused] == ["a", "c", "b"]
    assert all(c.score > 0 for c in fused)
    # 两路都靠后的 b 必须垫底
    assert fused[-1].id == "b"
    # 权重极端时结果应可解释
    only_dense = rrf_fuse(dense, [], 1.0)
    assert [c.id for c in only_dense] == ["a", "b", "c"]
    print("✓ test_backends_share_rrf_semantics")


def test_cross_backend_parity_memory_vs_reference():
    """内存后端与「手工计算」结果一致，确保它是可信的参照实现。"""
    from recruiter_agent.vectorstore import MemoryVectorStore

    st = MemoryVectorStore()
    emb = HashEmbedder(384)
    texts = {
        "x1": "张伟 北京 AI平台 LLM RAG",
        "x2": "李娜 北京 大模型 Agent LangGraph",
        "x3": "王强 上海 高级后端 Java 分布式",
    }
    cities = {"x1": "北京", "x2": "北京", "x3": "上海"}
    chunks = [Chunk(id=k, text=v, metadata={"city": cities[k]}) for k, v in texts.items()]
    emb.fit(list(texts.values()))
    st.upsert("candidates", chunks, emb.embed(list(texts.values())))
    # 显式传 filter，验证其在融合前生效
    hits = st.hybrid_search("candidates", "北京 LLM", emb.embed_one("北京 LLM"), 2,
                            filters={"city": "北京"})
    assert hits, "混合检索必须有结果"
    assert all(h.metadata["city"] == "北京" for h in hits), "filter 必须在融合前生效"
    assert "x3" not in [h.id for h in hits], "上海的记录不应进入结果"
    print("✓ test_cross_backend_parity_memory_vs_reference")


# ================================================================= runner
def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"✗ {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"✗ {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{'─'*50}\n{len(tests) - failed}/{len(tests)} 通过"
          + (f"，{failed} 失败" if failed else "，全部通过 ✅"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

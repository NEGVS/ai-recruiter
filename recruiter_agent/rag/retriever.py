"""RAG 检索管线：Query 改写 → 多路召回 → 融合 → Rerank。

三个针对招聘域的关键设计：
1. **Query 改写**：把自然语言需求翻译成「检索友好的结构化查询」+ 同义词扩展
   （如 "大模型" → LLM/RAG/Agent/大模型），显著提升召回率。
2. **硬约束不交给向量**：城市、年限、学历、薪资这类可精确判定的条件走
   向量库的 filter 通道，在召回阶段就过滤掉不合格记录，减少 rerank 负担。
3. **多路召回**：向量(语义) + BM25(精确词) + 画像分(结构化打分) 三路 RRF 融合。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..config import Settings, get_settings
from ..embedding import get_embedder
from ..intent_rules import extract_skills
from ..state import Chunk
from ..vectorstore import COLLECTION_CANDIDATES, COLLECTION_KNOWLEDGE, VectorStore, get_store
from .rerank import rerank, skill_match

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- 同义词/别名
SYNONYMS: dict[str, list[str]] = {
    "大模型": ["LLM", "大语言模型", "GPT"],
    "LLM": ["大模型", "大语言模型", "语言模型"],
    "RAG": ["检索增强", "检索增强生成", "知识库问答", "RAG"],
    "Agent": ["智能体", "智能代理", "Agent", "工作流编排"],
    "推荐": ["推荐系统", "推荐算法", "召回排序", "CTR"],
    "搜索": ["检索", "搜索引擎", "搜索算法", "ES", "Elasticsearch"],
    "后端": ["服务端", "后端开发", "Backend", "API"],
    "前端": ["Frontend", "Web开发", "前端开发"],
    "数据": ["大数据", "数据开发", "数仓", "数据仓库"],
    "算法": ["算法工程师", "机器学习", "深度学习", "ML", "DL"],
    "高并发": ["高并发", "性能优化", "秒杀", "分布式"],
    "运维": ["SRE", "DevOps", "运维", "稳定性"],
}


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    rewritten_query: str
    expanded_terms: list[str]
    filters: dict[str, Any]
    stats: dict[str, Any]
    strategies: list[str]


# ---------------------------------------------------------------- Query 改写
def rewrite_query(query: str, entities: dict[str, Any], intent: str) -> tuple[str, list[str]]:
    """生成检索查询与扩展词。"""
    base_skills = list(entities.get("skills", []))
    expanded: list[str] = list(base_skills)
    for s in base_skills:
        for syn in SYNONYMS.get(s, []):
            if syn.lower() != s.lower() and syn not in expanded:
                expanded.append(syn)

    parts: list[str] = []
    for k in ("job_title", "candidate_name"):
        if entities.get(k):
            parts.append(str(entities[k]))
    parts.extend(expanded)
    if entities.get("domains"):
        parts.extend(entities["domains"])
    if intent == "candidate_search" and not parts:
        parts.append(query)

    rewritten = " ".join(dict.fromkeys(parts)) if parts else query
    return rewritten, expanded


# ---------------------------------------------------------------- 硬约束 filter
NEGATION_HINTS = ("不考虑", "不限", "忽略", "不要", "排除")


def build_filters(entities: dict[str, Any], query: str, intent: str) -> dict[str, Any]:
    """把可精确判定的条件转成向量库 filter。"""
    filters: dict[str, Any] = {}
    negated: list[str] = entities.get("negated", [])

    cities = entities.get("cities", [])
    if cities and not any(c in negated for c in ("城市", "city")):
        # 统一用 city 字段（存为单值），传 list 时各后端按「任一命中」处理
        filters["city"] = cities if len(cities) > 1 else cities[0]

    if intent in ("candidate_search", "candidate_evaluate"):
        if "min_years" in entities and "经验" not in negated and "年限" not in negated:
            filters["min_years"] = float(entities["min_years"])
        if "max_years" in entities and "经验" not in negated:
            filters["max_years"] = float(entities["max_years"])
        if "max_salary" in entities and "薪资" not in negated and "薪酬" not in negated:
            filters["max_salary"] = float(entities["max_salary"])
        if "min_salary" in entities and "薪资" not in negated and "薪酬" not in negated:
            filters["min_salary"] = float(entities["min_salary"])
        if entities.get("job_intent") and "意向" not in negated:
            filters["job_intent"] = entities["job_intent"]
        if entities.get("tags") and "标签" not in negated:
            filters["tags"] = entities["tags"]
    return filters


def detect_negation(query: str) -> list[str]:
    out: list[str] = []
    for hint in NEGATION_HINTS:
        if hint in query:
            tail = query.split(hint, 1)[1][:8]
            out.append(tail.strip("，。, ") or hint)
    return out


# ---------------------------------------------------------------- 画像分召回
def profile_search(chunks: list[Chunk], entities: dict[str, Any],
                   filters: dict[str, Any] | None = None) -> list[Chunk]:
    """结构化画像通道：不依赖文本相似度，直接按硬条件打分。

    注意：这条通道的候选池来自**无 filter** 的宽召回（否则命中数为 0 就无从打分），
    因此必须在这里自行执行硬约束过滤，否则会把明显不满足条件的候选人放进结果里。
    """
    q_skills = list(entities.get("skills", []))
    cities = set(entities.get("cities", []))
    negated = set(entities.get("negated") or [])
    min_y = entities.get("min_years")
    max_y = entities.get("max_years")
    out: list[Chunk] = []
    for c in chunks:
        md = c.metadata

        # --- 硬约束：与向量库 filter 保持同一套语义 ---
        if cities and "城市" not in negated and str(md.get("city", "")) not in cities:
            continue
        yrs = float(md.get("years", 0) or 0)
        if min_y is not None and "经验" not in negated and yrs < float(min_y):
            continue
        sal = float(md.get("expected_salary", 0) or 0)
        if (filters or {}).get("max_salary") and sal > float(filters["max_salary"]):
            continue
        if (filters or {}).get("min_salary") and sal < float(filters["min_salary"]):
            continue

        sc = 0.0
        ratio, hits = skill_match(q_skills, c)
        sc += 0.65 * ratio
        if min_y is not None and yrs >= float(min_y):
            sc += 0.2
        if max_y is not None and yrs <= float(max_y) + 1:
            sc += 0.1
        if cities:
            sc += 0.25
        if md.get("job_intent") == "active":
            sc += 0.1

        if sc > 0:
            ev = [f"画像匹配度 {sc:.2f}"]
            if hits:
                ev.append(f"命中技能：{', '.join(hits)}")
            cc = Chunk(id=c.id, text=c.text, metadata=dict(c.metadata), score=round(sc, 6),
                       evidence=ev)
            out.append(cc)
    out.sort(key=lambda x: x.score, reverse=True)
    return out


# ---------------------------------------------------------------- 主管线
class Retriever:
    def __init__(self, store: VectorStore | None = None, s: Settings | None = None) -> None:
        self._store = store  # 显式注入时固定；为 None 时每次动态解析
        self.s = s or get_settings()

    @property
    def store(self) -> VectorStore:
        """惰性解析向量库。

        不能在 __init__ 里固化 store：入库脚本可能在 Retriever 构造之后才写入数据，
        缓存下来的旧实例会导致召回恒为空。
        """
        return self._store or get_store()

    @property
    def embedder(self):
        return get_embedder()

    def retrieve(
        self,
        query: str,
        entities: dict[str, Any],
        intent: str,
        collection: str = COLLECTION_CANDIDATES,
        top_k: int | None = None,
        do_rerank: bool = True,
        use_profile_channel: bool = True,
    ) -> RetrievalResult:
        top_k = top_k or self.s.retrieval_top_k
        negated = entities.setdefault("negated", detect_negation(query))
        rewritten, expanded = rewrite_query(query, entities, intent)
        filters = build_filters(entities, query, intent)

        strategies: list[str] = []
        dense_vec = self.embedder.embed_one(rewritten)

        # 1) 向量召回
        try:
            dense = self.store.dense_search(collection, dense_vec, top_k, filters or None)
            strategies.append("dense")
        except Exception as e:  # noqa: BLE001
            log.warning("dense 召回失败: %s", e)
            dense = []

        # 2) BM25 召回
        try:
            sparse = self.store.sparse_search(collection, query, top_k, filters or None)
            strategies.append("bm25")
        except Exception as e:  # noqa: BLE001
            log.warning("bm25 召回失败: %s", e)
            sparse = []

        # 3) 画像分召回
        profile: list[Chunk] = []
        if use_profile_channel and collection == COLLECTION_CANDIDATES:
            try:
                pool = self.store.sparse_search(collection, " ".join(expanded) or query, top_k * 2, None)
                profile = profile_search(pool, entities, filters)
                if profile:
                    strategies.append("profile")
            except Exception as e:  # noqa: BLE001
                log.debug("profile 通道失败: %s", e)

        # RRF 融合三路
        from ..vectorstore.base import rrf_fuse

        merged = rrf_fuse(dense, sparse, 0.6)
        if profile:
            fused = rrf_fuse(merged, profile, 0.55)
        else:
            fused = merged

        stats = {
            "dense": len(dense),
            "bm25": len(sparse),
            "profile": len(profile),
            "fused": len(fused),
            "filters": filters,
        }

        if do_rerank:
            ranked = rerank(fused, rewritten, entities, top_n=top_k, s=self.s)
            strategies.append(f"rerank:{self.s.rerank_provider}")
        else:
            ranked = fused[:top_k]

        return RetrievalResult(
            chunks=ranked,
            rewritten_query=rewritten,
            expanded_terms=expanded,
            filters=filters,
            stats=stats,
            strategies=strategies,
        )

    def retrieve_knowledge(self, query: str, top_k: int | None = None, do_rerank: bool = True) -> RetrievalResult:
        """知识库（政策/流程/制度）检索，语义占比更高。"""
        top_k = top_k or self.s.rerank_top_n * 2
        vec = self.embedder.embed_one(query)
        dense = self.store.dense_search(COLLECTION_KNOWLEDGE, vec, top_k)
        sparse = self.store.sparse_search(COLLECTION_KNOWLEDGE, query, top_k)
        from ..vectorstore.base import rrf_fuse

        fused = rrf_fuse(dense, sparse, 0.65)
        if do_rerank:
            ranked = rerank(fused, query, {}, top_n=self.s.rerank_top_n, s=self.s)
        else:
            ranked = fused[:top_k]
        return RetrievalResult(
            chunks=ranked, rewritten_query=query, expanded_terms=[], filters={},
            stats={"dense": len(dense), "bm25": len(sparse), "fused": len(fused)},
            strategies=["knowledge_dense", "knowledge_bm25"],
        )


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def reset_retriever() -> None:
    global _retriever
    _retriever = None

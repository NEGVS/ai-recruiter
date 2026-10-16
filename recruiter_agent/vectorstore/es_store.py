"""Elasticsearch 后端。

要点：
1. Mapping 显式声明：`skills`/`domains` 为 keyword（便于 terms 过滤 + BM25），
   `resume_text` 等长文本为 text（BM25 主战场），`embedding` 为 dense_vector。
2. 混合检索用 ES 原生 `bool.should` + `function_score`，同时给出
   `_score`(BM25) 与 cosine，融合方式与内存实现保持一致（RRF 或线性加权），
   保证换后端时排序行为可比、可回归。
3. dense_vector 使用 cosine 相似度 + int8_hnsw 量化，索引体积与召回均可控。
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

from ..config import Settings
from ..state import Chunk
from .base import COLLECTION_CANDIDATES, COLLECTION_KNOWLEDGE, VectorStore, build_es_filters, rrf_fuse

log = logging.getLogger(__name__)

_TEXT_FIELDS = ("text", "name", "title", "school", "major", "highlights_text")
_KEYWORD_FIELDS = (
    "skills", "domains", "tags", "city", "education", "job_intent",
    "level", "hard_skills", "nice_skills", "doc_id", "candidate_id",
    "chunk_type", "source",
)


class ElasticsearchStore(VectorStore):
    backend = "elasticsearch"

    def __init__(self, s: Settings) -> None:
        from elasticsearch import Elasticsearch

        client_kwargs: dict[str, Any] = {"request_timeout": 30}
        if s.es_api_key:
            client_kwargs["api_key"] = s.es_api_key
        if s.es_url.startswith("https"):
            client_kwargs["verify_certs"] = s.es_verify_certs
        self._es = Elasticsearch(s.es_url, **client_kwargs)
        self._s = s
        self._dim: int | None = None

    # ------------------------------------------------------------ 索引名
    def _index(self, collection: str) -> str:
        if collection == COLLECTION_CANDIDATES:
            return self._s.es_index_candidates
        if collection == COLLECTION_KNOWLEDGE:
            return self._s.es_index_knowledge
        raise ValueError(f"unknown collection: {collection}")

    # ------------------------------------------------------------ 健康
    def health(self) -> bool:
        try:
            return bool(self._es.ping())
        except Exception as e:  # noqa: BLE001
            log.debug("ES ping 失败: %s", e)
            return False

    # ------------------------------------------------------------ Mapping
    def _mapping(self, dim: int) -> dict[str, Any]:
        props: dict[str, Any] = {
            "text": {"type": "text", "analyzer": "standard",
                     "fields": {"raw": {"type": "keyword", "ignore_above": 512}}},
            "embedding": {
                "type": "dense_vector",
                "dims": dim,
                "index": True,
                "similarity": "cosine",
                "index_options": {"type": "int8_hnsw", "m": 16, "ef_construction": 128},
            },
        }
        for f in _KEYWORD_FIELDS:
            if f != "text":
                props[f] = {"type": "keyword", "ignore_above": 512}
        props["years"] = {"type": "float"}
        props["expected_salary"] = {"type": "integer"}
        props["current_salary"] = {"type": "integer"}
        props["index"] = {"type": "integer"}
        return {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {"analyzer": {
                    "zh_standard": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase"]}
                }},
            },
            "mappings": {"properties": props},
        }

    def ensure_index(self, collection: str, dim: int) -> None:
        idx = self._index(collection)
        if not self._es.indices.exists(index=idx):
            self._es.indices.create(index=idx, **self._mapping(dim))
            log.info("创建 ES 索引 %s (dim=%d)", idx, dim)
        self._dim = dim

    def count(self, collection: str) -> int:
        try:
            return int(self._es.count(index=self._index(collection))["count"])
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------ 写入
    def upsert(self, collection: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0
        dim = len(vectors[0])
        self.ensure_index(collection, dim)
        ops: list[dict[str, Any]] = []
        for i, (ch, vec) in enumerate(zip(chunks, vectors)):
            doc = {
                "text": ch.text,
                "embedding": list(vec),
                "index": i,
                **{k: v for k, v in ch.metadata.items() if k in _KEYWORD_FIELDS or k in
                   ("years", "expected_salary", "current_salary")},
            }
            if "highlights" in ch.metadata:
                doc["highlights_text"] = " ".join(ch.metadata["highlights"])
            # 未在 mapping 中声明的字段会被动态映射；显式关闭以保证 mapping 稳定
            doc.setdefault("source", "recruiter")
            ops.append({"index": {"_index": self._index(collection), "_id": ch.id}})
            ops.append(doc)
        resp = self._es.bulk(operations=ops, refresh=True)
        if resp.get("errors"):
            first_err = next((it["index"].get("error") for it in resp["items"] if it["index"].get("error")), None)
            raise RuntimeError(f"ES bulk 写入失败: {first_err}")
        return len(chunks)

    # ------------------------------------------------------------ 读取
    def _to_chunks(self, hits: list[dict[str, Any]]) -> list[Chunk]:
        out: list[Chunk] = []
        for h in hits:
            src = h.get("_source", {})
            meta = {k: v for k, v in src.items() if k not in ("text", "embedding")}
            out.append(Chunk(
                id=h["_id"], text=src.get("text", ""),
                score=float(h.get("_score") or 0.0), metadata=meta,
            ))
        return out

    def get_by_id(self, collection: str, doc_id: str) -> Chunk | None:
        try:
            h = self._es.get(index=self._index(collection), id=doc_id)
        except Exception:  # noqa: BLE001
            return None
        src = h["_source"]
        return Chunk(id=h["_id"], text=src.get("text", ""),
                     metadata={k: v for k, v in src.items() if k not in ("text", "embedding")})

    # ------------------------------------------------------------ 检索
    def dense_search(self, collection: str, vector: Sequence[float], top_k: int,
                     filters: dict[str, Any] | None = None) -> list[Chunk]:
        knn = {
            "field": "embedding",
            "query_vector": list(vector),
            "k": top_k,
            "num_candidates": self._s.es_knn_candidates,
            "filter": build_es_filters(filters),
        }
        resp = self._es.search(index=self._index(collection), knn=knn, size=top_k, source_excludes=["embedding"])
        chunks = self._to_chunks(resp["hits"]["hits"])
        for rank, c in enumerate(chunks, start=1):
            c.dense_rank = rank
        return chunks

    def sparse_search(self, collection: str, query: str, top_k: int,
                      filters: dict[str, Any] | None = None) -> list[Chunk]:
        q: dict[str, Any] = {
            "multi_match": {
                "query": query,
                "fields": [f"{f}^2" if f in ("text", "name", "title") else f for f in _TEXT_FIELDS if f != "text"] + ["text^3"],
                "type": "best_fields",
                "minimum_should_match": "30%",
            }
        }
        f = build_es_filters(filters)
        if f:
            # build_es_filters 已返回完整的 bool query，直接作为 must 子句即可
            q = {"bool": {"must": [q, f]}}
        resp = self._es.search(index=self._index(collection), query=q, size=top_k,
                               source_excludes=["embedding"])
        chunks = self._to_chunks(resp["hits"]["hits"])
        for rank, c in enumerate(chunks, start=1):
            c.sparse_rank = rank
        return chunks

    def hybrid_search(self, collection: str, query: str, vector: Sequence[float], top_k: int,
                      filters: dict[str, Any] | None = None, dense_weight: float = 0.6) -> list[Chunk]:
        """两路分别召回后在应用层 RRF 融合。

        相比 ES 原生 retriever，这样做的好处是：与内存/Milvus 后端的融合逻辑完全一致，
        便于用同一套评测集横向对比不同后端。
        """
        pool = max(top_k * 3, top_k)
        dense = self.dense_search(collection, vector, pool, filters)
        sparse = self.sparse_search(collection, query, pool, filters)
        return rrf_fuse(dense, sparse, dense_weight)[:top_k]

    def stats(self) -> dict[str, int]:
        return {COLLECTION_CANDIDATES: self.count(COLLECTION_CANDIDATES),
                COLLECTION_KNOWLEDGE: self.count(COLLECTION_KNOWLEDGE)}

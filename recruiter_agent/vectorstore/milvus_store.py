"""Milvus 后端（pymilvus 2.4+ / Milvus Lite 亦可）。

与 ES 的差异及应对：
* Milvus 原生支持 `AnnSearchRequest` 做多路混合检索（dense + sparse BM25 Function），
  这里采用「dense + BM25 Function」服务端混合 + 本地 RRF 对齐的策略，
  保持跨后端排序行为可比。
* 动态字段：schema 显式列出高频过滤字段，其余结构化信息存入 `meta_json`。
* 标量过滤用 expression string（`city in [...] and years >= 3`）。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from ..config import Settings
from ..state import Chunk
from .base import COLLECTION_CANDIDATES, COLLECTION_KNOWLEDGE, VectorStore, rrf_fuse

log = logging.getLogger(__name__)


def _to_expr(filters: dict[str, Any] | None) -> str:
    if not filters:
        return ""
    conds: list[str] = []
    for key, val in filters.items():
        if val is None:
            continue
        if key == "ids":
            conds.append(f'candidate_id in [{",".join(chr(34) + str(v) + chr(34) for v in val)}]')
        elif key == "min_years":
            conds.append(f"years >= {float(val)}")
        elif key == "max_salary":
            conds.append(f"expected_salary <= {float(val)}")
        elif key == "min_salary":
            conds.append(f"expected_salary >= {float(val)}")
        elif isinstance(val, (list, tuple, set)):
            vals = list(val)
            if not vals:
                continue
            if key in ("skills", "domains", "tags", "hard_skills", "nice_skills"):
                conds.append(f"{key} in {vals}")
            else:
                in_str = ", ".join(chr(34) + str(v) + chr(34) for v in vals)
                conds.append(f"{key} in [{in_str}]")
        else:
            conds.append(f'{key} == "{val}"')
    return " and ".join(conds)


class MilvusStore(VectorStore):
    backend = "milvus"

    def __init__(self, s: Settings) -> None:
        from pymilvus import MilvusClient

        self._s = s
        self._client = MilvusClient(uri=s.milvus_uri, token=s.milvus_token or "", db_name=s.milvus_db)
        self._dim: int | None = None

    def _collection(self, collection: str) -> str:
        return (self._s.milvus_collection_candidates if collection == COLLECTION_CANDIDATES
                else self._s.milvus_collection_knowledge)

    def health(self) -> bool:
        try:
            self._client.list_collections()
            return True
        except Exception as e:  # noqa: BLE001
            log.debug("Milvus 不可用: %s", e)
            return False

    # ------------------------------------------------------------ schema
    def ensure_collection(self, collection: str, dim: int) -> None:
        name = self._collection(collection)
        if self._client.has_collection(name):
            return
        from pymilvus import DataType, Function, FunctionType

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("text", DataType.VARCHAR, max_length=8192, enable_analyzer=True)
        schema.add_field("candidate_id", DataType.VARCHAR, max_length=64)
        schema.add_field("city", DataType.VARCHAR, max_length=64)
        schema.add_field("education", DataType.VARCHAR, max_length=32)
        schema.add_field("job_intent", DataType.VARCHAR, max_length=32)
        schema.add_field("skills", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64)
        schema.add_field("domains", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=32)
        schema.add_field("tags", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=32)
        schema.add_field("years", DataType.FLOAT)
        schema.add_field("expected_salary", DataType.INT64)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)

        # 服务端 BM25 稀疏检索
        schema.add_function(Function(name="bm25", function_type=FunctionType.BM25, input_field_names=["text"],
                                     output_field_names=["sparse"]))
        idx = self._client.prepare_index_params()
        idx.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
        idx.add_index(field_name="sparse", index_type="AUTOINDEX", metric_type="BM25")
        self._client.create_collection(name, schema=schema, index_params=idx, consistency_level=self._s.milvus_consistency_level)
        log.info("创建 Milvus collection %s (dim=%d)", name, dim)
        self._dim = dim

    def count(self, collection: str) -> int:
        try:
            stats = self._client.get_collection_stats(self._collection(collection))
            return int(stats.get("row_count", 0))
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------ 写入
    def upsert(self, collection: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0
        dim = len(vectors[0])
        self.ensure_collection(collection, dim)
        rows: list[dict[str, Any]] = []
        for ch, vec in zip(chunks, vectors):
            md = ch.metadata
            rows.append({
                "chunk_id": ch.id,
                "text": ch.text,
                "dense": list(vec),
                "candidate_id": str(md.get("candidate_id", md.get("doc_id", ch.id))),
                "city": str(md.get("city", "")),
                "education": str(md.get("education", "")),
                "job_intent": str(md.get("job_intent", "")),
                "skills": list(md.get("skills", []))[:64],
                "domains": list(md.get("domains", []))[:32],
                "tags": list(md.get("tags", []))[:32],
                "years": float(md.get("years", 0) or 0),
                "expected_salary": int(md.get("expected_salary", 0) or 0),
                "meta_json": json.dumps(md, ensure_ascii=False)[:8000],
            })
        self._client.upsert(collection_name=self._collection(collection), data=rows)
        return len(rows)

    # ------------------------------------------------------------ 读取
    def _rows_to_chunks(self, rows: list[dict[str, Any]]) -> list[Chunk]:
        out: list[Chunk] = []
        for r in rows:
            raw = r.get("meta_json")
            meta = json.loads(raw) if raw else {}
            meta.setdefault("candidate_id", r.get("candidate_id"))
            out.append(Chunk(id=r["chunk_id"], text=r.get("text", ""), score=0.0, metadata=meta))
        return out

    def get_by_id(self, collection: str, doc_id: str) -> Chunk | None:
        try:
            rows = self._client.get(collection_name=self._collection(collection), ids=[doc_id])
        except Exception:  # noqa: BLE001
            return None
        return self._rows_to_chunks(rows)[0] if rows else None

    # ------------------------------------------------------------ 检索
    def _search(self, collection: str, vector: Sequence[float] | None, query: str | None,
                top_k: int, filters: dict[str, Any] | None) -> list[Chunk]:
        from pymilvus import AnnSearchRequest

        name = self._collection(collection)
        expr = _to_expr(filters)
        reqs = []
        if vector is not None:
            reqs.append(AnnSearchRequest(data=[list(vector)], anns_field="dense", param={"metric_type": "COSINE"},
                                         limit=top_k, expr=expr))
        if query:
            reqs.append(AnnSearchRequest(data=[query], anns_field="sparse", param={"metric_type": "BM25"},
                                         limit=top_k, expr=expr))
        if not reqs:
            return []
        hits = self._client.hybrid_search(
            collection_name=name, reqs=reqs,
            ranker=rrf_ranker(), limit=top_k, output_fields=["text", "meta_json"],
        )
        chunks = self._rows_to_chunks([dict(h) for h in hits])
        for rank, c in enumerate(chunks, start=1):
            c.score = round(1.0 / (60 + rank), 6)  # 与 RRF 尺度对齐
        return chunks

    def dense_search(self, collection: str, vector: Sequence[float], top_k: int,
                     filters: dict[str, Any] | None = None) -> list[Chunk]:
        chunks = self._search(collection, vector, None, top_k, filters)
        for rank, c in enumerate(chunks, start=1):
            c.dense_rank = rank
        return chunks

    def sparse_search(self, collection: str, query: str, top_k: int,
                      filters: dict[str, Any] | None = None) -> list[Chunk]:
        chunks = self._search(collection, None, query, top_k, filters)
        for rank, c in enumerate(chunks, start=1):
            c.sparse_rank = rank
        return chunks

    def hybrid_search(self, collection: str, query: str, vector: Sequence[float], top_k: int,
                      filters: dict[str, Any] | None = None, dense_weight: float = 0.6) -> list[Chunk]:
        pool = max(top_k * 3, top_k)
        dense = self.dense_search(collection, vector, pool, filters)
        sparse = self.sparse_search(collection, query, pool, filters)
        return rrf_fuse(dense, sparse, dense_weight)[:top_k]

    def stats(self) -> dict[str, int]:
        return {COLLECTION_CANDIDATES: self.count(COLLECTION_CANDIDATES),
                COLLECTION_KNOWLEDGE: self.count(COLLECTION_KNOWLEDGE)}


def rrf_ranker():
    from pymilvus import AnnSearchRequest, RRFRanker  # noqa: F401

    return RRFRanker(k=60)

"""内存向量后端：BM25 + 稠密点积 + RRF。

用途：
  * 本地无 ES/Milvus 时让项目零依赖跑通；
  * 单元测试中作为确定性的参照实现。
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Sequence

from ..embedding import cosine
from ..state import Chunk
from .base import COLLECTION_CANDIDATES, COLLECTION_KNOWLEDGE, VectorStore, rrf_fuse

log = logging.getLogger(__name__)


class MemoryVectorStore(VectorStore):
    """纯 Python 内存后端：BM25 + 稠密点积 + RRF。

    用途：
      * 本地无 ES/Milvus 时让项目零依赖跑通；
      * 单元测试里作为确定性的参照实现。
    """

    backend = "memory"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._data: dict[str, dict[str, Chunk]] = {c: {} for c in (COLLECTION_CANDIDATES, COLLECTION_KNOWLEDGE)}
        self._vecs: dict[str, dict[str, list[float]]] = {c: {} for c in (COLLECTION_CANDIDATES, COLLECTION_KNOWLEDGE)}
        self._postings: dict[str, dict[str, dict[str, int]]] = {c: defaultdict(dict) for c in self._data}
        self._df: dict[str, dict[str, int]] = {c: defaultdict(int) for c in self._data}
        self._len: dict[str, dict[str, int]] = {c: {} for c in self._data}
        self._avg_len: dict[str, float] = {c: 0.0 for c in self._data}
        self._k1, self._b = k1, b

    # -- 写入
    def upsert(self, collection: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        from ..embedding import tokenize

        for ch, vec in zip(chunks, vectors):
            self._data[collection][ch.id] = ch
            self._vecs[collection][ch.id] = list(vec)
            toks = tokenize(ch.text)
            self._len[collection][ch.id] = len(toks)
            for t in set(toks):
                self._postings[collection][t][ch.id] = toks.count(t)
                self._df[collection][t] += 1
        n = len(self._len[collection])
        self._avg_len[collection] = (sum(self._len[collection].values()) / n) if n else 0.0
        return len(chunks)

    def count(self, collection: str) -> int:
        return len(self._data.get(collection, {}))

    # -- 过滤
    @staticmethod
    def _match(chunk: Chunk, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        md = chunk.metadata
        for key, val in filters.items():
            if val is None:
                continue
            if key == "ids":
                if chunk.candidate_id not in val and chunk.id not in val:
                    return False
            elif key == "min_years":
                if float(md.get("years", 0)) < float(val):
                    return False
            elif key == "max_years":
                if float(md.get("years", 0)) > float(val):
                    return False
            elif key == "max_salary":
                if float(md.get("expected_salary", 1e9)) > float(val):
                    return False
            elif key == "min_salary":
                if float(md.get("expected_salary", 0)) < float(val):
                    return False
            elif isinstance(val, (list, tuple, set)):
                actual = md.get(key)
                if isinstance(actual, list):
                    if not set(val) & set(actual):
                        return False
                elif actual not in val:
                    return False
            else:
                if md.get(key) != val:
                    return False
        return True

    # -- BM25
    def _bm25(self, collection: str, query: str, top_k: int, filters: dict[str, Any] | None) -> list[Chunk]:
        from ..embedding import tokenize

        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        N = max(len(self._data[collection]), 1)
        avgdl = self._avg_len[collection] or 1.0
        scores: dict[str, float] = defaultdict(float)
        for t in q_tokens:
            df = self._df[collection].get(t, 0)
            if df == 0:
                continue
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            for doc_id, tf in self._postings[collection].get(t, {}).items():
                dl = self._len[collection].get(doc_id, 1)
                denom = tf + self._k1 * (1 - self._b + self._b * dl / avgdl)
                scores[doc_id] += idf * (tf * (self._k1 + 1)) / (denom or 1.0)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out: list[Chunk] = []
        for rank, (doc_id, sc) in enumerate(ranked, start=1):
            ch = self._data[collection].get(doc_id)
            if not ch or not self._match(ch, filters):
                continue
            c = _clone(ch)
            c.score = round(sc, 6)
            c.sparse_rank = rank
            out.append(c)
        return out

    # -- 检索
    def dense_search(self, collection: str, vector: Sequence[float], top_k: int,
                     filters: dict[str, Any] | None = None) -> list[Chunk]:
        scored = []
        for doc_id, vec in self._vecs[collection].items():
            ch = self._data[collection].get(doc_id)
            if not ch or not self._match(ch, filters):
                continue
            scored.append((cosine(list(vector), vec), ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for rank, (sim, ch) in enumerate(scored[:top_k], start=1):
            c = _clone(ch)
            c.score = round(sim, 6)
            c.dense_rank = rank
            out.append(c)
        return out

    def sparse_search(self, collection: str, query: str, top_k: int,
                      filters: dict[str, Any] | None = None) -> list[Chunk]:
        return self._bm25(collection, query, top_k, filters)

    def hybrid_search(self, collection: str, query: str, vector: Sequence[float], top_k: int,
                      filters: dict[str, Any] | None = None, dense_weight: float = 0.6) -> list[Chunk]:
        pool = max(top_k * 3, top_k)
        dense = self.dense_search(collection, vector, pool, filters)
        sparse = self.sparse_search(collection, query, pool, filters)
        return rrf_fuse(dense, sparse, dense_weight)[:top_k]

    def get_by_id(self, collection: str, doc_id: str) -> Chunk | None:
        ch = self._data.get(collection, {}).get(doc_id)
        return _clone(ch) if ch else None

    def health(self) -> bool:
        return True

    def stats(self) -> dict[str, int]:
        return {c: self.count(c) for c in self._data}


def _clone(ch: Chunk) -> Chunk:
    return Chunk(
        id=ch.id,
        text=ch.text,
        score=ch.score,
        metadata=dict(ch.metadata),
        dense_rank=ch.dense_rank,
        sparse_rank=ch.sparse_rank,
        rerank_score=ch.rerank_score,
        evidence=list(ch.evidence),
    )

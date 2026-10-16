"""统一向量存储抽象：ES / Milvus / 内存 三种后端。

上层检索逻辑只依赖 VectorStore 协议，因此可以无痛切换后端。
每个后端都实现 **混合检索**（BM25 稀疏 + Dense 向量），
这是招聘场景的关键：候选人姓名/学校/证书这类精确词 BM25 更强，
技能描述/项目语义 Dense 更强。
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Sequence

from ..embedding import cosine
from ..state import Chunk

log = logging.getLogger(__name__)

COLLECTION_CANDIDATES = "candidates"
COLLECTION_KNOWLEDGE = "knowledge"


# ================================================================== 协议
class VectorStore(ABC):
    backend: str = "base"

    # --- 写入 ---
    @abstractmethod
    def upsert(self, collection: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int: ...

    @abstractmethod
    def count(self, collection: str) -> int: ...

    # --- 检索 ---
    @abstractmethod
    def dense_search(
        self, collection: str, vector: Sequence[float], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[Chunk]: ...

    @abstractmethod
    def sparse_search(
        self, collection: str, query: str, top_k: int, filters: dict[str, Any] | None = None
    ) -> list[Chunk]: ...

    @abstractmethod
    def hybrid_search(
        self, collection: str, query: str, vector: Sequence[float], top_k: int,
        filters: dict[str, Any] | None = None, dense_weight: float = 0.6,
    ) -> list[Chunk]: ...

    @abstractmethod
    def get_by_id(self, collection: str, doc_id: str) -> Chunk | None: ...

    @abstractmethod
    def health(self) -> bool: ...


# ================================================================== RRF 融合
def rrf_fuse(
    dense: list[Chunk], sparse: list[Chunk], dense_weight: float = 0.6, k: int = 60
) -> list[Chunk]:
    """Reciprocal Rank Fusion —— 融合两路召回，天然鲁棒，无需分数归一化。"""
    if not dense and not sparse:
        return []
    scores: dict[str, float] = defaultdict(float)
    merged: dict[str, Chunk] = {}
    for rank, c in enumerate(dense, start=1):
        scores[c.id] += dense_weight * 1.0 / (k + rank)
        merged[c.id] = c
    for rank, c in enumerate(sparse, start=1):
        scores[c.id] += (1 - dense_weight) * 1.0 / (k + rank)
        if c.id in merged:
            m = merged[c.id]
            m.dense_rank = m.dense_rank or rank
            m.sparse_rank = rank
        else:
            c.sparse_rank = rank
            merged[c.id] = c
    out = list(merged.values())
    for c in out:
        c.score = round(scores[c.id], 6)
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def build_es_filters(filters: dict[str, Any] | None) -> dict[str, Any] | list | None:
    """把通用 filter dict 翻译为 ES bool query。"""
    if not filters:
        return None
    must, should = [], []
    for key, val in filters.items():
        if val is None:
            continue
        if key == "ids":
            must.append({"ids": {"values": val}})
        elif key in RANGE_FIELDS:
            # 逻辑字段名 → ES 实际字段名 + 操作符
            field, op = RANGE_FIELDS[key]
            must.append({"range": {field: {op: val}}})
        elif isinstance(val, (list, tuple, set)):
            if not val:
                continue
            # mapping 中 city/education 等已直接声明为 keyword，
            # 再加 .keyword 后缀会指向不存在的字段（ES 8 会直接报错）
            must.append({"terms": {key: list(val)}})
        else:
            must.append({"term": {key: val}})
    if must and should:
        return {"bool": {"must": must, "should": should}}
    if must:
        return {"bool": {"must": must}}
    if should:
        return {"bool": {"should": should}}
    return None


# 逻辑名 → (ES 字段名, 操作符)
# 注意 max_salary/min_salary 是「对 expected_salary 的上下界」，
# 不是 ES 里名为 max_salary 的字段——早期版本直接透传会导致过滤静默失效。
RANGE_FIELDS: dict[str, tuple[str, str]] = {
    "min_years": ("years", "gte"),
    "max_years": ("years", "lte"),
    "min_salary": ("expected_salary", "gte"),
    "max_salary": ("expected_salary", "lte"),
}

"""向量存储工厂：按配置选择后端，失败自动降级，保证 Agent 永不因基础设施挂掉。"""
from __future__ import annotations

import logging

from ..config import Settings, get_settings
from .base import COLLECTION_CANDIDATES, COLLECTION_KNOWLEDGE, VectorStore, rrf_fuse
from .es_store import ElasticsearchStore
from .memory_store import MemoryVectorStore
from .milvus_store import MilvusStore

log = logging.getLogger(__name__)

_store: VectorStore | None = None


def _probe(name: str, factory) -> VectorStore | None:
    try:
        st = factory()
        if st.health():
            log.info("向量库后端: %s ✅", name)
            return st
        log.warning("向量库后端 %s 健康检查失败", name)
    except Exception as e:  # noqa: BLE001
        log.warning("向量库后端 %s 不可用: %s", name, e)
    return None


def get_store(s: Settings | None = None, force: str | None = None) -> VectorStore:
    """获取全局单例。

    force 可指定后端（如 'memory'）。**无论是否 force，结果都会写入单例**，
    否则「用 memory 后端执行入库」这类调用会写进临时对象，
    后续 Agent 拿到的却是另一个空实例（表现为召回恒为 0）。
    """
    global _store
    if _store is not None and force is None:
        return _store
    s = s or get_settings()
    backend = force or s.vector_store_backend

    if backend == "memory":
        _store = MemoryVectorStore()
        return _store

    st: VectorStore | None = None
    if backend in ("auto", "elasticsearch"):
        st = _probe("elasticsearch", lambda: ElasticsearchStore(s))
    if st is None and backend in ("auto", "milvus"):
        st = _probe("milvus", lambda: MilvusStore(s))
    if st is None:
        if backend != "auto":
            log.warning("指定后端 %s 不可用，回退内存实现", backend)
        st = MemoryVectorStore()
    _store = st
    return st


def reset_store() -> None:
    global _store
    _store = None


__all__ = [
    "VectorStore",
    "MemoryVectorStore",
    "ElasticsearchStore",
    "MilvusStore",
    "get_store",
    "reset_store",
    "rrf_fuse",
    "COLLECTION_CANDIDATES",
    "COLLECTION_KNOWLEDGE",
]

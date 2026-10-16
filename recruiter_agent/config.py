"""全局配置。所有可调参数集中在此，支持环境变量覆盖（前缀 RECRUITER_）。"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RECRUITER_", env_file=".env", extra="ignore"
    )

    # ---------- LLM ----------
    llm_provider: Literal["openai", "mock"] = "mock"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1400

    # ---------- Embedding ----------
    embedding_provider: Literal["sentence_transformers", "openai", "hash"] = "hash"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 384

    # ---------- Vector Store ----------
    # auto: 优先 ES -> Milvus -> 内存回退；也可强制指定
    vector_store_backend: Literal["auto", "elasticsearch", "milvus", "memory"] = "auto"

    es_url: str = "http://localhost:9200"
    es_index_candidates: str = "recruit_candidates_v1"
    es_index_knowledge: str = "recruit_knowledge_v1"
    es_api_key: str | None = None
    es_verify_certs: bool = False
    es_bm25_weight: float = 0.4  # 混合检索中 BM25 权重（1-dim 归给向量）
    es_knn_candidates: int = 100

    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_db: str = "recruit"
    milvus_collection_candidates: str = "candidates_v1"
    milvus_collection_knowledge: str = "knowledge_v1"
    milvus_consistency_level: str = "Strong"

    # ---------- Rerank ----------
    rerank_provider: Literal["cross_encoder", "rrf", "llm"] = "rrf"
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = 5
    rrf_k: int = 60

    # ---------- Graph ----------
    max_reflection_rounds: int = 2
    max_tool_calls: int = 6
    retrieval_top_k: int = 30  # 召回数量（rerank 前）

    # ---------- Scoring ----------
    scoring_weights: dict[str, float] = {
        "hard_skill": 0.30,
        "domain_experience": 0.20,
        "seniority": 0.15,
        "interview": 0.15,
        "stability": 0.08,
        "salary_fit": 0.07,
        "job_intent": 0.05,
    }
    pass_score: float = 60.0

    @property
    def has_real_llm(self) -> bool:
        return self.llm_provider == "openai" and bool(self.llm_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # 允许无 .env 直接跑
    os.environ.setdefault("RECRUITER_VECTOR_STORE_BACKEND", "auto")
    return Settings()
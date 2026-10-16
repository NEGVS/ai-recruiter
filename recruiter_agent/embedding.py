"""Embedding 抽象层：真实向量模型 / 本地 hash embedding 双实现。

hash embedding 不是玩具——它是一个确定性的、基于 token 哈希 + 子线性 TF 的
稀疏投影向量，可正常参与余弦相似度计算。在没有 GPU/模型权重的环境下，
它保证检索链路（召回→融合→rerank）真实可运行；一旦配置真实模型即自动切换。
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Protocol

from .config import Settings, get_settings

_TOKEN_RE = re.compile(r"[a-zA-Z]+|[0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中英混合分词：英文按词，中文按字 + 2-gram。"""
    text = (text or "").lower()
    tokens = _TOKEN_RE.findall(text)
    cjk = [t for t in tokens if "\u4e00" <= t <= "\u9fff"]
    cjk_bigrams = ["".join(p) for p in zip(cjk, cjk[1:])]
    en = [t for t in tokens if not ("\u4e00" <= t <= "\u9fff")]
    return en + cjk + cjk_bigrams


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]: ...


# ------------------------------------------------------------------ hash 实现
class HashEmbedder:
    """确定性 hash embedding（signed hashing trick + IDF 加权 + L2 归一化）。

    相比朴素版本，加入了**语料级 IDF**：
      * 拟合阶段统计每个 token 的 document frequency；
      * 打向量时按 idf 加权，抑制「的/工程师/经验」这类高频词的主导作用。
    这让「LLM RAG 向量检索」与「RAG LLM 检索增强」的相似度显著高于
    与无关文本的相似度，检索行为更接近真实语义模型。

    若语料未 fit 过，则退化为无 IDF 的版本（仍然确定性且可运行）。
    """

    name = "hash-embedder"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self._df: Counter[str] = Counter()
        self._n_docs: int = 0
        self._fitted = False

    # ---------- 拟合 ----------
    def fit(self, texts: list[str]) -> "HashEmbedder":
        self._df.clear()
        self._n_docs = 0
        for t in texts:
            self._n_docs += 1
            for tok in set(tokenize(t)):
                self._df[tok] += 1
        self._fitted = self._n_docs > 0
        return self

    def _idf(self, tok: str) -> float:
        if not self._fitted:
            return 1.0
        df = self._df.get(tok, 0)
        # 平滑 IDF，范围约 (0, log(N+1)]
        return math.log((self._n_docs + 1) / (df + 1)) + 1.0

    # ---------- 向量化 ----------
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = tokenize(text)
        if not toks:
            return vec
        counts = Counter(toks)
        for tok, cnt in counts.items():
            weight = (1.0 + math.log(cnt)) * self._idf(tok)  # sublinear TF × IDF
            h = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# ------------------------------------------------------------------ 真实模型
class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(model_name)
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        arr = self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in arr]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OpenAIEmbedder:
    name = "openai-embedder"

    def __init__(self, s: Settings) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": s.llm_api_key}
        if s.llm_base_url:
            kwargs["base_url"] = s.llm_base_url
        self._c = OpenAI(**kwargs)
        self._model = "text-embedding-3-small"
        self.dim = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):
            batch = texts[i : i + 128]
            resp = self._c.embeddings.create(model=self._model, input=batch)
            out.extend([list(map(float, d.embedding)) for d in resp.data])
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ------------------------------------------------------------------ 工厂
_embedder: Embedder | None = None


def get_embedder(s: Settings | None = None) -> Embedder:
    global _embedder
    if _embedder is not None:
        return _embedder
    s = s or get_settings()
    if s.embedding_provider == "sentence_transformers":
        try:
            _embedder = SentenceTransformerEmbedder(s.embedding_model)
        except Exception as e:  # noqa: BLE001
            print(f"[embedder] 加载 {s.embedding_model} 失败({e})，回退 hash embedding")
            _embedder = HashEmbedder(s.embedding_dim)
    elif s.embedding_provider == "openai" and s.has_real_llm:
        _embedder = OpenAIEmbedder(s)
    else:
        _embedder = HashEmbedder(s.embedding_dim)
    return _embedder


def reset_embedder() -> None:
    global _embedder
    _embedder = None


# ------------------------------------------------------------------ 相似度
def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mmr(
    query_vec: list[float],
    doc_vecs: list[list[float]],
    top_k: int = 10,
    lambda_: float = 0.7,
) -> list[int]:
    """MMR 去冗余重排，返回 doc 索引。"""
    if not doc_vecs:
        return []
    sims = [cosine(query_vec, d) for d in doc_vecs]
    selected: list[int] = []
    candidates = set(range(len(doc_vecs)))
    while candidates and len(selected) < top_k:
        if not selected:
            best = max(candidates, key=lambda i: sims[i])
        else:
            def mmr_score(i: int) -> float:
                redundancy = max(cosine(doc_vecs[i], doc_vecs[j]) for j in selected)
                return lambda_ * sims[i] - (1 - lambda_) * redundancy

            best = max(candidates, key=mmr_score)
        selected.append(best)
        candidates.discard(best)
    return selected

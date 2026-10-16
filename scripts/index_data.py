"""向量化入库脚本：把候选人 + 知识文档切分成 Chunk 并写入向量库。

运行：
    python scripts/index_data.py              # 自动选后端
    python scripts/index_data.py --backend memory
    python scripts/index_data.py --backend elasticsearch
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recruiter_agent.embedding import get_embedder  # noqa: E402
from recruiter_agent.state import Chunk  # noqa: E402
from recruiter_agent.tools.business_tools import DB_PATH  # noqa: E402
from recruiter_agent.vectorstore import (  # noqa: E402
    COLLECTION_CANDIDATES, COLLECTION_KNOWLEDGE, get_store,
)

KNOWLEDGE_DIR = ROOT / "data" / "knowledge"

EDU_RANK = {"大专": 1, "本科": 2, "硕士": 3, "博士": 4}


# ---------------------------------------------------------------- 切分
def chunk_resume(text: str, max_len: int = 320) -> list[str]:
    """轻量切分：按句号切 + 超长硬切。中文简历文本规整，无需引入复杂 splitter。"""
    sents = re.split(r"(?<=[。；;.!?！？])", text)
    out, cur = [], ""
    for s in sents:
        if not s.strip():
            continue
        if len(cur) + len(s) <= max_len:
            cur += s
        else:
            if cur:
                out.append(cur.strip())
            cur = s if len(s) <= max_len else s[:max_len]
    if cur:
        out.append(cur.strip())
    return out or [text[:max_len]]


def chunk_markdown(text: str, max_len: int = 420) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= max_len:
            cur = f"{cur}\n{p}" if cur else p
        else:
            if cur:
                out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return out or [text[:max_len]]


# ---------------------------------------------------------------- 构建
def build_candidate_chunks() -> tuple[list[Chunk], dict[str, dict]]:
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute("SELECT * FROM candidates")]

    chunks: list[Chunk] = []
    profiles: dict[str, dict] = {}
    for r in rows:
        skills = json.loads(r.pop("skills_json", "[]") or "[]")
        domains = json.loads(r.pop("domains_json", "[]") or "[]")
        projects = json.loads(r.pop("projects_json", "[]") or "[]")
        highlights = json.loads(r.pop("highlights_json", "[]") or "[]")
        tags = json.loads(r.pop("tags_json", "[]") or "[]")
        interview = json.loads(r.pop("interview_json", "{}") or "{}")
        risks = json.loads(r.pop("risks_json", "[]") or "[]")
        r["edu_rank"] = EDU_RANK.get(r["education"], 2)

        base_meta = {
            "candidate_id": r["candidate_id"], "name": r["name"], "title": r["title"],
            "years": r["years"], "city": r["city"], "education": r["education"],
            "school": r.get("school", ""), "major": r.get("major", ""),
            "skills": skills, "domains": domains, "tags": tags,
            "job_intent": r["job_intent"], "expected_salary": r["expected_salary"],
            "current_salary": r.get("current_salary", 0),
            "availability_days": r.get("availability_days", 30),
            "stability_years_per_role": r.get("stability_years_per_role", 0),
            "job_changes_3y": r.get("job_changes_3y", 1),
            "interview_score": interview.get("score", 0),
            "interview_rounds": interview.get("rounds", 0),
            "known_risks": risks,
            "source": "candidate_profile",
        }
        profiles[r["candidate_id"]] = {**base_meta, "resume_text": r["resume_text"],
                                       "highlights": highlights, "projects": projects,
                                       "interview": interview}

        # chunk 0：画像摘要（强结构化字段，filter 友好）
        summary = (
            f"{r['name']}｜{r['title']}｜{r['years']}年经验｜{r['city']}｜{r['education']}（{r.get('school','')} {r.get('major','')}）\n"
            f"核心技能：{'、'.join(skills)}\n"
            f"领域：{'、'.join(domains)}\n"
            f"求职意向：{r['job_intent']}｜期望薪资 {r['expected_salary']//1000}K｜"
            f"可到岗 {r.get('availability_days',30)}天｜三年跳动 {r.get('job_changes_3y',1)}次\n"
            f"亮点：{'；'.join(highlights)}"
        )
        chunks.append(Chunk(id=f"{r['candidate_id']}#summary", text=summary,
                            metadata={**base_meta, "chunk_type": "summary", "index": 0}))

        # chunk 1+：简历正文分片
        for i, part in enumerate(chunk_resume(r["resume_text"])):
            chunks.append(Chunk(id=f"{r['candidate_id']}#resume{i}", text=part,
                                metadata={**base_meta, "chunk_type": "resume", "index": i + 1}))
    return chunks, profiles


def build_knowledge_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        m = re.search(r"\*\*文档ID\*\*:\s*(\S+)", text)
        doc_id = m.group(1) if m else p.stem
        m2 = re.search(r"\*\*类型\*\*:\s*(\S+)", text)
        ctype = m2.group(1) if m2 else "policy"
        m3 = re.search(r"\*\*标签\*\*:\s*(.+)", text)
        tags = [t.strip() for t in m3.group(1).split(",")] if m3 else []
        m4 = re.search(r"^#\s+(.+)$", text, re.M)
        title = m4.group(1).strip() if m4 else p.stem
        body = re.sub(r"^#.*$", "", text, count=1, flags=re.M)
        body = re.sub(r"\*\*.*?\*\*:.*", "", body)

        for i, part in enumerate(chunk_markdown(body)):
            chunks.append(Chunk(
                id=f"{doc_id}#{i}", text=f"【{title}】{part}".strip(),
                metadata={"doc_id": doc_id, "title": title, "chunk_type": ctype,
                          "tags": tags, "source": "knowledge_base", "index": i,
                          "skills": [], "domains": [], "city": ""},
            ))
    return chunks


# ---------------------------------------------------------------- 主流程
def index(backend: str = "auto", batch: int = 64) -> dict[str, int]:
    store = get_store(force=backend)
    emb = get_embedder()
    print(f"▶ 向量库后端: {store.backend} | Embedding: {emb.name}(dim={emb.dim})")

    c_chunks, profiles = build_candidate_chunks()
    k_chunks = build_knowledge_chunks()

    # hash embedding 需要先 fit 语料统计 IDF，否则高频词会主导相似度
    if hasattr(emb, "fit"):
        emb.fit([c.text for c in c_chunks] + [c.text for c in k_chunks])
        print(f"  ✓ embedding 语料拟合：{len(c_chunks) + len(k_chunks)} 段")

    total = 0
    for coll, chunks in ((COLLECTION_CANDIDATES, c_chunks), (COLLECTION_KNOWLEDGE, k_chunks)):
        n = 0
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            vecs = emb.embed([c.text for c in part])
            n += store.upsert(coll, part, vecs)
        total += n
        print(f"  ✓ {coll}: 写入 {n} chunks（候选库现有 {store.count(coll)}）")

    # 缓存画像，供评分节点使用
    (ROOT / "data" / "profiles_cache.json").write_text(
        json.dumps(profiles, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"✅ 入库完成，共 {total} chunks")
    return {"candidates": len(c_chunks), "knowledge": len(k_chunks), "backend": store.backend}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "elasticsearch", "milvus", "memory"])
    a = ap.parse_args()
    index(a.backend)

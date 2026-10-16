"""FastAPI 服务：把 Agent 暴露为 HTTP 接口。

启动： python -m uvicorn api:app --reload --port 8000
文档： http://localhost:8000/docs
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from recruiter_agent import RecruitAgent
from recruiter_agent.tools import all_tools
from recruiter_agent.vectorstore import get_store

app = FastAPI(
    title="AI 招聘 Agent API",
    description="基于 LangGraph 编排的 RAG + 多工具 + 自反思招聘助手",
    version="1.0.0",
)

_agent: RecruitAgent | None = None


def agent() -> RecruitAgent:
    global _agent
    if _agent is None:
        from pathlib import Path

        root = Path(__file__).resolve().parent
        if not (root / "data" / "recruit.db").exists():
            from scripts.init_data import build_db
            build_db()
        if get_store().count("candidates") == 0:
            from scripts.index_data import index
            index("memory")
        _agent = RecruitAgent()
    return _agent


# ---------------------------------------------------------------- 请求/响应
class AskRequest(BaseModel):
    query: str = Field(..., description="自然语言招聘需求", examples=["找 5 个北京 3 年以上 LLM 候选人"])
    session_id: str | None = Field(None, description="会话 ID，用于日志串联")
    with_trace: bool = Field(False, description="是否返回完整执行链路")


class AskResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    intent: str
    intent_confidence: float
    entities: dict[str, Any]
    tools_called: list[str]
    shortlist: list[dict[str, Any]]
    reflection: dict[str, Any] | None
    total_ms: int
    trace: list[dict[str, Any]] = []


# ---------------------------------------------------------------- 接口
@app.get("/health", summary="健康检查")
def health() -> dict[str, Any]:
    store = get_store()
    return {
        "status": "ok",
        "vector_store": store.backend,
        "candidates": store.count("candidates"),
        "knowledge_docs": store.count("knowledge"),
        "tools": sorted(all_tools().keys()),
    }


@app.post("/ask", response_model=AskResponse, summary="招聘需求问答")
def ask(req: AskRequest) -> AskResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")
    sid = req.session_id or uuid.uuid4().hex[:12]
    r = agent().run(req.query, session_id=sid)
    return AskResponse(
        session_id=sid,
        query=req.query,
        answer=r["answer"],
        intent=r.get("intent", ""),
        intent_confidence=r.get("intent_confidence", 0.0),
        entities=r.get("entities") or {},
        tools_called=[t.get("tool", "") for t in (r.get("tool_results") or [])],
        shortlist=[
            {
                "rank": s.get("rank"), "candidate_id": s.get("candidate_id"),
                "name": s.get("name"), "title": s.get("title"),
                "total": s.get("total"), "passed": s.get("passed"),
                "vetoed": s.get("vetoed"), "risks": s.get("risks", []),
            }
            for s in (r.get("shortlist") or [])
        ],
        reflection=r.get("reflection"),
        total_ms=r.get("total_ms", 0),
        trace=r.get("trace", []) if req.with_trace else [],
    )


@app.get("/graph", summary="导出 LangGraph 结构（Mermaid）")
def graph_mermaid() -> dict[str, str]:
    """返回编排图的 Mermaid 文本，便于文档/演示引用。"""
    from recruiter_agent.graph import get_graph

    try:
        return {"format": "mermaid", "content": get_graph().get_graph().draw_mermaid()}
    except Exception as e:  # noqa: BLE001
        return {"format": "mermaid", "content": f"# 导出失败: {e}"}


@app.get("/candidates", summary="候选人列表")
def list_candidates(limit: int = 20, city: str | None = None) -> dict[str, Any]:
    import sqlite3

    from recruiter_agent.tools.business_tools import DB_PATH

    sql = "SELECT candidate_id,name,title,years,city,education,skills_json,expected_salary,job_intent FROM candidates"
    params: list[Any] = []
    if city:
        sql += " WHERE city=?"
        params.append(city)
    sql += " ORDER BY years DESC LIMIT ?"
    params.append(limit)
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(sql, params)]
    import json as _json

    for r in rows:
        r["skills"] = _json.loads(r.pop("skills_json", "[]"))
    return {"count": len(rows), "candidates": rows}


@app.get("/jobs", summary="在招岗位")
def list_jobs() -> dict[str, Any]:
    import json as _json
    import sqlite3

    from recruiter_agent.tools.business_tools import DB_PATH

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute("SELECT * FROM jobs")]
    for r in rows:
        r["hard_skills"] = _json.loads(r.pop("hard_skills_json", "[]"))
        r.pop("nice_skills_json", None)
        r.pop("domains_json", None)
    return {"count": len(rows), "jobs": rows}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

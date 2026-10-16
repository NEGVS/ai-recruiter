"""Agent 状态定义 —— 整个 LangGraph 各节点共享的唯一状态契约。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import MessagesState

# ---------------------------------------------------------------- 领域数据结构


@dataclass
class Chunk:
    """向量库中的一条记录（候选人画像片段或知识库文档片段）。"""

    id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None
    evidence: list[str] = field(default_factory=list)  # 命中的证据片段，供 Reflection 校验

    @property
    def candidate_id(self) -> str:
        return str(self.metadata.get("candidate_id", self.metadata.get("doc_id", self.id)))


@dataclass
class ToolResult:
    name: str
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    cost_ms: int = 0
    observations: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    candidate_id: str
    name: str
    title: str
    years: float
    city: str
    education: str
    school: str = ""
    major: str = ""
    skills: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    projects: list[dict[str, Any]] = field(default_factory=list)
    job_intent: str = "active"  # active | passive | urgent
    expected_salary: int = 0
    current_salary: int = 0
    availability_days: int = 30
    stability_years_per_role: float = 2.0
    job_changes_3y: int = 1
    interview: dict[str, Any] = field(default_factory=dict)
    highlights: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    resume_text: str = ""
    tags: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "title": self.title,
            "years": self.years,
            "city": self.city,
            "education": self.education,
            "skills": self.skills,
            "domains": self.domains,
            "job_intent": self.job_intent,
            "expected_salary": self.expected_salary,
            "tags": self.tags,
        }


@dataclass
class JobRequisition:
    job_id: str
    title: str
    city: str
    level: str  # 校招/1-3/3-5/5-8/8+
    min_years: float
    max_years: float
    salary_range: tuple[int, int]
    hard_skills: list[str] = field(default_factory=list)
    nice_skills: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    education_req: str = "不限"
    headcount: int = 1
    urgency: str = "normal"  # low | normal | high
    description: str = ""


@dataclass
class ScoreBreakdown:
    candidate_id: str
    total: float
    dimensions: dict[str, dict[str, float]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    vetoed: bool = False  # 触发一票否决（硬技能零命中 / 经验严重不足）

    @property
    def passed(self) -> bool:
        return self.total > 0  # 由调用方结合 pass_score 与 vetoed 判断


@dataclass
class Reflection:
    """Reflection 节点的产物：自我批判 + 决策。"""

    round: int
    critique: str = ""
    issues: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)
    hallucination_flags: list[str] = field(default_factory=list)
    decision: Literal["accept", "revise"] = "accept"
    next_actions: list[str] = field(default_factory=list)
    score: float = 1.0

    @property
    def needs_revision(self) -> bool:
        return self.decision == "revise"


@dataclass
class TraceStep:
    node: str
    ms: int
    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


class RecruitState(MessagesState):
    """LangGraph 主状态。继承 MessagesState 自带 messages 归约器。"""

    # --- 输入 ---
    user_query: str
    session_id: str

    # --- 意图 ---
    intent: str
    intent_confidence: float
    entities: dict[str, Any]
    clarified: bool

    # --- 计划 ---
    plan: list[dict[str, Any]]

    # --- 中间产物 ---
    tool_results: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    knowledge_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]
    scores: list[dict[str, Any]]
    shortlist: list[dict[str, Any]]
    reflection: dict[str, Any] | None
    reflection_history: list[dict[str, Any]]

    # --- 各节点写入的元信息（必须在 State 中声明，否则会被 LangGraph 丢弃）---
    intent_reason: str
    rewritten_query: str
    expanded_terms: list[str]
    retrieval_stats: dict[str, Any]
    rerank_meta: dict[str, Any]
    score_meta: dict[str, Any]
    facts: str
    pending_interview: dict[str, Any] | None
    pending_input: dict[str, Any] | None

    # --- 控制 ---
    iteration: int
    revision_count: int
    tool_call_count: int
    errors: list[str]

    # --- 输出 ---
    final_answer: str
    answer_meta: dict[str, Any]
    trace: list[dict[str, Any]]
    total_ms: int
    started_at: float


def new_state(user_query: str, session_id: str = "cli") -> RecruitState:
    """构造初始状态。"""
    return RecruitState(
        user_query=user_query,
        session_id=session_id,
        intent="",
        intent_confidence=0.0,
        intent_reason="",
        entities={},
        clarified=False,
        plan=[],
        tool_results=[],
        retrieved_chunks=[],
        knowledge_chunks=[],
        reranked_chunks=[],
        scores=[],
        shortlist=[],
        reflection=None,
        reflection_history=[],
        rewritten_query="",
        expanded_terms=[],
        retrieval_stats={},
        rerank_meta={},
        score_meta={},
        facts="",
        pending_interview=None,
        pending_input=None,
        iteration=0,
        revision_count=0,
        tool_call_count=0,
        errors=[],
        final_answer="",
        answer_meta={},
        trace=[],
        total_ms=0,
        started_at=time.time(),
    )


def merge_dict(left: dict | None, right: dict | None) -> dict:
    """langgraph reducer：字典合并。"""
    out = dict(left or {})
    out.update(right or {})
    return out
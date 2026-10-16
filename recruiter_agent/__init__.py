"""AI 招聘 Agent：基于 LangGraph 编排的 RAG + 多工具 + 自反思检索系统。"""
from .config import Settings, get_settings
from .graph import RecruitAgent, build_graph, get_agent, get_graph
from .state import Candidate, Chunk, JobRequisition, RecruitState, new_state

__version__ = "1.0.0"

__all__ = [
    "RecruitAgent", "build_graph", "get_graph", "get_agent",
    "Settings", "get_settings", "RecruitState", "new_state",
    "Candidate", "Chunk", "JobRequisition", "__version__",
]

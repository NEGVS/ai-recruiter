"""工具（Tool）层。

每个工具遵循统一契约：
    name / description / run(**kwargs) -> ToolResult

设计要点：
* 工具是**确定性**的（查库、算薪、排期），语言模型只负责决定「调哪个、传什么参数」，
  从而把不可控的生成风险隔离在工具之外。
* 每个工具都带 `cost_ms` 与 `observations`，让 Trace 与 Reflection 有据可查。
* 工具失败不抛异常，统一返回 ok=False，由图路由决定「重试 / 降级 / 直接回答」。
"""
from __future__ import annotations

import time
from typing import Any, Callable, Protocol

from ..state import ToolResult


class Tool(Protocol):
    name: str
    description: str
    required: list[str]

    def run(self, **kwargs: Any) -> ToolResult: ...


class ToolSpec:
    """工具元信息，供 LLM 做 function calling 时的 schema。"""

    def __init__(self, name: str, description: str, required: list[str],
                 optional: dict[str, str] | None = None) -> None:
        self.name = name
        self.description = description
        self.required = required
        self.optional = optional or {}

    def to_schema(self) -> dict[str, Any]:
        props: dict[str, Any] = {k: {"type": "string", "description": v}
                                 for k, v in self.optional.items()}
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": self.required,
            },
        }

    def to_prompt_line(self) -> str:
        args = ", ".join(f"{k}:{v}" for k, v in self.optional.items())
        return f"- {self.name}({', '.join(self.required)}{'; ' + args if args else ''}) — {self.description}"


class BaseTool:
    name = "base"
    description = ""
    required: list[str] = []
    optional: dict[str, str] = {}

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.required, self.optional)

    def run(self, **kwargs: Any) -> ToolResult:  # pragma: no cover
        raise NotImplementedError

    # 供子类复用：自动计时 + 异常归一
    def safe_run(self, **kwargs: Any) -> ToolResult:
        t0 = time.time()
        missing = [k for k in self.required if k not in kwargs or kwargs[k] in (None, "")]
        if missing:
            return ToolResult(name=self.name, ok=False, error=f"缺少必填参数: {missing}",
                              cost_ms=int((time.time() - t0) * 1000))
        try:
            res = self.run(**kwargs)
        except Exception as e:  # noqa: BLE001
            return ToolResult(name=self.name, ok=False, error=f"{type(e).__name__}: {e}",
                              cost_ms=int((time.time() - t0) * 1000))
        res.cost_ms = int((time.time() - t0) * 1000)
        return res


# ---------------------------------------------------------------- 注册表
_REGISTRY: dict[str, BaseTool] = {}


def register(tool_cls: type[BaseTool]) -> type[BaseTool]:
    inst = tool_cls()
    _REGISTRY[inst.name] = inst
    return tool_cls


def get_tool(name: str) -> BaseTool | None:
    return _REGISTRY.get(name)


def all_tools() -> dict[str, BaseTool]:
    return dict(_REGISTRY)


def tool_schemas() -> list[dict[str, Any]]:
    return [t.spec().to_schema() for t in _REGISTRY.values()]


def tool_prompt() -> str:
    return "\n".join(t.spec().to_prompt_line() for t in _REGISTRY.values())

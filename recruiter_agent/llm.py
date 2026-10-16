"""LLM 抽象层。

设计目标：没有配置任何 API Key 时，用「确定性启发式 Mock」替代真实 LLM，
让整条 LangGraph 链路（含 Reflection 的批判与改写）依然可以完整跑通、可测试。

真实 LLM 通过 OpenAI 兼容协议接入（OpenAI / DeepSeek / 通义 / vLLM / Ollama 均可）。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .config import Settings, get_settings

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ 通用接口
class LLM(Protocol):
    name: str

    def chat(
        self, system: str, user: str, json_mode: bool = False, **kw: Any
    ) -> str: ...

    def chat_json(self, system: str, user: str, schema_hint: str = "") -> dict[str, Any]:
        ...


# ------------------------------------------------------------------ OpenAI 实现
class OpenAIChat:
    name = "openai"

    def __init__(self, s: Settings) -> None:
        from openai import OpenAI  # 延迟导入

        kwargs: dict[str, Any] = {"api_key": s.llm_api_key}
        if s.llm_base_url:
            kwargs["base_url"] = s.llm_base_url
        self._client = OpenAI(**kwargs)
        self._s = s

    def chat(self, system: str, user: str, json_mode: bool = False, **kw: Any) -> str:
        resp = self._client.chat.completions.create(
            model=kw.get("model", self._s.llm_model),
            temperature=kw.get("temperature", self._s.llm_temperature),
            max_tokens=kw.get("max_tokens", self._s.llm_max_tokens),
            response_format={"type": "json_object"} if json_mode else None,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def chat_json(self, system: str, user: str, schema_hint: str = "") -> dict[str, Any]:
        raw = self.chat(
            system + f"\n只输出严格 JSON。目标结构：{schema_hint}",
            user,
            json_mode=True,
        )
        return extract_json(raw)


# ------------------------------------------------------------------ 本地 Mock
@dataclass
class MockChat:
    """无 API Key 时的确定性回退实现。

    不是「假装智能」——它用规则匹配产出结构合法的结果，保证：
      1. 编排图（Graph / 条件边 / 循环）可端到端测试；
      2. Embedding / 检索 / 评分这些真实逻辑照样被真实执行；
      3. 换成真 LLM 时只有「语言表达」这一步变化。
    """

    name: str = "mock"

    def chat(self, system: str, user: str, json_mode: bool = False, **kw: Any) -> str:
        # 依据 system 提示词判断任务类型
        if "意图识别" in system or "intent" in system.lower():
            return json.dumps(_mock_intent(user), ensure_ascii=False)
        if "反思" in system or "critique" in system.lower():
            return json.dumps(_mock_reflection(user), ensure_ascii=False)
        if "rerank" in system.lower():
            return json.dumps(_mock_rerank(user), ensure_ascii=False)
        return _mock_answer(system, user)

    def chat_json(self, system: str, user: str, schema_hint: str = "") -> dict[str, Any]:
        return json.loads(self.chat(system, user, json_mode=True))


# ------------------------------------------------------------------ 工具函数
def extract_json(raw: str) -> dict[str, Any]:
    """从可能带 markdown 围栏的自然语言里抠出 JSON。"""
    if not raw:
        return {}
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 退化：抓第一个 {...} 平衡块
    start = text.find("{")
    if start < 0:
        return {}
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def _mock_intent(user: str) -> dict[str, Any]:
    from .intent_rules import rule_based_intent

    return rule_based_intent(user)


def _mock_reflection(user: str) -> dict[str, Any]:
    """基于上下文中的实际短板做批判，而不是空喊。"""
    ctx = _parse_ctx(user)
    missing, issues, hal = [], [], []

    if ctx.get("tool_calls", 0) == 0:
        issues.append("未调用任何工具，结论缺少数据支撑")
    if ctx.get("candidate_count", 0) == 0:
        issues.append("候选人召回为 0，检索条件可能过窄")
    if ctx.get("top_score", 0) < ctx.get("pass_score", 60):
        issues.append("最高分低于门槛线，需显式说明人才市场结论")
    if ctx.get("has_evidence", False) is False:
        hal.append("回答中出现无证据支撑的断言")
    if not ctx.get("has_salary_info", False):
        missing.append("薪酬带宽/期望薪资未核实")

    decision = "revise" if (missing or issues) and ctx.get("round", 0) < 2 else "accept"
    return {
        "critique": "；".join(issues + hal) or "证据链完整，结论与召回数据一致。",
        "issues": issues,
        "missing_aspects": missing,
        "hallucination_flags": hal,
        "decision": decision,
        "next_actions": ["补充薪酬数据查询", "放宽技能硬约束重新召回"] if decision == "revise" else [],
        "score": max(0.0, 1.0 - 0.2 * len(issues) - 0.15 * len(hal)),
    }


def _mock_rerank(user: str) -> dict[str, Any]:
    return {"order": [], "reason": "mock 模式使用规则 RRF 排序"}


def _parse_ctx(user: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in user.splitlines():
        if line.startswith("#"):
            k, _, v = line[1:].partition(":")
            k, v = k.strip(), v.strip()
            if v.isdigit():
                out[k] = int(v)
            elif v in ("True", "False"):
                out[k] = v == "True"
            elif v.replace(".", "", 1).isdigit():
                out[k] = float(v)
            else:
                out[k] = v
    return out


def _mock_answer(system: str, user: str) -> str:
    """把已获得的结构化结果渲染成人话（真实 LLM 在这里会更好）。"""
    payload = _parse_ctx(user)
    return (
        f"[Mock 渲染] 针对「{payload.get('query', user[:40])}」，"
        f"共召回 {payload.get('candidate_count', 0)} 位候选人，"
        f"综合排名第一 {payload.get('top_name', 'N/A')}，得分 {payload.get('top_score', 0)}。"
        " 配置 RECRUITER_LLM_PROVIDER=openai 与 API Key 后此处将由真实模型生成。"
    )


# ------------------------------------------------------------------ 工厂
_llm_singleton: LLM | None = None


def get_llm(s: Settings | None = None) -> LLM:
    global _llm_singleton
    if _llm_singleton is not None:
        return _llm_singleton
    s = s or get_settings()
    if s.has_real_llm:
        try:
            _llm_singleton = OpenAIChat(s)
            log.info("LLM: OpenAI 兼容模式 model=%s", s.llm_model)
        except Exception as e:  # noqa: BLE001
            log.warning("LLM 初始化失败，回退 mock: %s", e)
            _llm_singleton = MockChat()
    else:
        _llm_singleton = MockChat()
        log.info("LLM: mock 模式（未配置 API Key，编排可完整跑通）")
    return _llm_singleton


def reset_llm() -> None:
    """测试用：清空单例。"""
    global _llm_singleton
    _llm_singleton = None

"""约束策略：区分「硬约束」与「软约束」，并集中管理放宽逻辑。

为什么需要单独一层：
    招聘场景里，用户说的「上海」「薪资 60k 以内」是**商业约束**，系统无权
    为了凑数而自动放宽——放宽意味着推荐了用户明确不要的人，这比「没有结果」
    伤害大得多（面试里这也是最容易被追问穿的一点）。

策略：
    * 硬约束（城市 / 薪资 / 学历 / 求职意向 / 标签）：**永不自动放宽**；
    * 软约束（年限）：允许放宽，但必须：
        1. 只放宽一次（不叠加）；
        2. 原始值快照进 `_original`，答案层必须同时展示「原条件」与「已放宽」；
        3. 放宽不会突破硬约束。

这样「推荐结果」永远可以用一句话解释：它满足了什么、为了让步放宽了什么。
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# 系统无权替用户放开的条件
HARD_FIELDS = ("cities", "max_salary", "min_salary", "min_salary_kw", "education",
               "job_intent", "tags")
# 允许放宽的软约束（仅年限，且有上限）
SOFT_FIELDS = ("min_years", "max_years")

# 年限放宽幅度：一次到位，不叠加
MIN_YEARS_RELAX = 2.0
MAX_YEARS_RELAX = 3.0
MAX_YEARS_CEILING = 20.0


def snapshot_original(entities: dict[str, Any]) -> dict[str, Any]:
    """记录用户原始约束（只记可判定的那几个字段）。"""
    keep = set(HARD_FIELDS) | set(SOFT_FIELDS)
    return {k: entities[k] for k in keep if entities.get(k) not in (None, "", [], {})}


def relax_soft_constraints(entities: dict[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[Any, Any]]]:
    """放宽软约束一次。返回 (新 entities, {字段: (放宽前, 放宽后)})。

    已经放宽过（存在 `_relaxed`）则不再放宽——避免多轮反思把条件越滚越松。
    """
    if entities.get("_relaxed"):
        return entities, {}

    original = entities.get("_original") or snapshot_original(entities)
    e = dict(entities)
    changed: dict[str, tuple[Any, Any]] = {}

    if e.get("min_years") is not None:
        before = float(e["min_years"])
        after = max(0.0, before - MIN_YEARS_RELAX)
        if after < before:
            e["min_years"] = after
            changed["min_years"] = (before, after)

    if e.get("max_years") is not None:
        before = float(e["max_years"])
        after = min(MAX_YEARS_CEILING, before + MAX_YEARS_RELAX)
        if after > before:
            e["max_years"] = after
            changed["max_years"] = (before, after)

    if changed:
        e["_original"] = original
        e["_relaxed"] = [{"round": 1, "changes": changed}]
        log.info("放宽软约束（仅年限）：%s", changed)

    return e, changed


def verify_hard_constraints(entities: dict[str, Any], original: dict[str, Any] | None) -> list[str]:
    """核对硬约束是否被改动过 —— 用于 Reflection 的硬性检查。"""
    if not original:
        return []
    violations: list[str] = []
    for k in HARD_FIELDS:
        if k in original and entities.get(k) != original[k]:
            violations.append(f"硬约束 {k} 被改动：{original[k]} → {entities.get(k)}")
    return violations

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from trajectory2gym.trajectory.tool_semantics import ToolSemantics


def rejected_side_effect_failures(
    failure_infos: Iterable[dict[str, Any]],
    *,
    tool_semantics: ToolSemantics,
) -> list[dict[str, Any]]:
    """Select rejected actions that could already have changed live state.

    A simulator can reject an action and expose a recovery continuation. In a
    live system the same action may be irreversible, so later recovery cannot
    restore task-success eligibility. This policy consumes normalized evidence
    and declarative tool semantics; it contains no benchmark workflow rules.
    """

    selected: list[dict[str, Any]] = []
    for info in failure_infos:
        if not isinstance(info, dict):
            continue
        actions = info.get("actual_actions")
        if not isinstance(actions, list):
            continue
        if any(
            tool_semantics.is_side_effecting(str(action.get("name") or ""))
            for action in actions
            if isinstance(action, dict)
        ):
            selected.append(info)
    return selected


def merge_unique_failures(*groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge policy selections without counting a recorded transition twice."""

    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    for group in groups:
        for info in group:
            marker = id(info)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(info)
    return merged

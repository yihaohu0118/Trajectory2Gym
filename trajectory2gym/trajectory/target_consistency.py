from __future__ import annotations

import copy
from typing import Any, Callable


def flatten_target_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        flattened: list[str] = []
        for item in value:
            flattened.extend(flatten_target_values(item))
        return flattened
    if isinstance(value, dict):
        flattened = []
        for item in value.values():
            flattened.extend(flatten_target_values(item))
        return flattened
    text = str(value).strip()
    return [text] if text else []


def target_value_signature(value: Any) -> Any:
    values = flatten_target_values(value)
    if isinstance(value, list):
        return tuple(sorted(values))
    if len(values) == 1:
        return values[0]
    return tuple(values)


def state_change_target_mismatches(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    *,
    target_keys: set[str] | frozenset[str],
    is_state_changing_tool: Callable[[str], bool],
    require_same_tool: bool = True,
) -> dict[str, dict[str, Any]]:
    """Compare verified state-changing action targets without executing them.

    This is a benchmark-neutral verifier primitive: adapters provide a tool
    taxonomy and target-key set, while the comparison only checks whether a
    proposed mutation targets the same observable entity as the verified step.
    """
    if len(actual) != 1 or len(expected) != 1:
        return {}
    actual_action = actual[0]
    expected_action = expected[0]
    actual_name = str(actual_action.get("name") or "")
    expected_name = str(expected_action.get("name") or "")
    if require_same_tool and actual_name != expected_name:
        return {}
    if not is_state_changing_tool(actual_name):
        return {}
    actual_args = actual_action.get("arguments") or {}
    expected_args = expected_action.get("arguments") or {}
    if not isinstance(actual_args, dict) or not isinstance(expected_args, dict):
        return {}

    mismatches: dict[str, dict[str, Any]] = {}
    for key in sorted(set(target_keys) & (set(actual_args) | set(expected_args))):
        actual_signature = target_value_signature(actual_args.get(key))
        expected_signature = target_value_signature(expected_args.get(key))
        if actual_signature != expected_signature:
            mismatches[key] = {
                "actual": copy.deepcopy(actual_args.get(key)),
                "expected": copy.deepcopy(expected_args.get(key)),
            }
    return mismatches

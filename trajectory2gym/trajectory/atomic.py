from __future__ import annotations

import copy
import json
from typing import Any


def unroll_multi_action_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split an evidenced multi-call message into atomic transitions.

    Splitting occurs only when ordered actions have an equal number of leading
    compiled tool observations.  Ambiguous records are preserved unchanged.
    """

    unrolled: list[dict[str, Any]] = []
    for source_step in steps:
        actions = list(source_step.get("gold_actions") or [])
        action_metadata = list(source_step.get("action_metadata") or [])
        observations = list(source_step.get("observations") or [])
        kind = str((source_step.get("metadata") or {}).get("kind") or "")
        leading_tool_observations: list[dict[str, Any]] = []
        for observation in observations:
            if not _is_compiled_tool_observation(observation):
                break
            leading_tool_observations.append(observation)

        if (
            kind != "official_dialogue_tool_call"
            or len(actions) <= 1
            or len(action_metadata) != len(actions)
            or len(leading_tool_observations) != len(actions)
        ):
            copied = copy.deepcopy(source_step)
            copied["step_index"] = len(unrolled)
            unrolled.append(copied)
            continue

        trailing_observations = observations[len(actions):]
        original_index = source_step.get("step_index")
        for action_index, (action, metadata, observation) in enumerate(
            zip(actions, action_metadata, leading_tool_observations)
        ):
            step_metadata = copy.deepcopy(source_step.get("metadata") or {})
            step_metadata.update(
                {
                    "atomic_action_index": action_index,
                    "atomic_action_count": len(actions),
                    "unrolled_from_step_index": original_index,
                }
            )
            atomic_observations = [copy.deepcopy(observation)]
            if action_index == len(actions) - 1:
                atomic_observations.extend(copy.deepcopy(trailing_observations))
            unrolled.append(
                {
                    "step_index": len(unrolled),
                    "gold_actions": [copy.deepcopy(action)],
                    "observations": atomic_observations,
                    "action_metadata": [copy.deepcopy(metadata)],
                    "metadata": step_metadata,
                }
            )
    return unrolled


def _is_compiled_tool_observation(observation: Any) -> bool:
    if not isinstance(observation, dict):
        return False
    if observation.get("role") == "tool":
        return True
    content = observation.get("content")
    if not isinstance(content, str):
        return False
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(payload, dict)
        and {"name", "response", "tool_reward"}.issubset(payload)
    )

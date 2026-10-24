from __future__ import annotations

import json
from typing import Any

from .schema import TrajectoryEpisode


def _nonempty_error_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return bool(value)


def _mapping_has_explicit_error(value: dict[str, Any]) -> bool:
    return any(
        str(key).strip().lower() in {"error", "errors", "exception"}
        and _nonempty_error_value(item)
        for key, item in value.items()
    )


def observation_is_explicit_error(observation: dict[str, Any]) -> bool:
    """Recognize explicit environment errors without interpreting semantics."""

    if _mapping_has_explicit_error(observation):
        return True
    content = observation.get("content")
    if observation.get("role") == "tool" and isinstance(content, str):
        if content.strip().lower().startswith("error"):
            return True
    payload: Any = content
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return False
    if not isinstance(payload, dict):
        return False
    if _mapping_has_explicit_error(payload):
        return True
    response = payload.get("response")
    if isinstance(response, str) and response.strip().lower().startswith("error"):
        return True
    if isinstance(response, dict) and _mapping_has_explicit_error(response):
        return True
    return False


def episode_explicit_error_count(episode: TrajectoryEpisode) -> int:
    # Compiler-created decision windows move the recorded trajectory prefix
    # into ``initial_messages``.  Errors in that prefix still belong to the
    # source episode: later verified transitions remain usable evidence, but a
    # recovered slice must not regain positive completion credit merely because
    # the failing transition sits before the window boundary.
    observations = (
        *episode.initial_messages,
        *(observation for step in episode.steps for observation in step.observations),
    )
    return sum(observation_is_explicit_error(observation) for observation in observations)

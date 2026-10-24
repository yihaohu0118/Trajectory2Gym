from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolAction:
    """Canonical tool action used by trajectory2gym adapters."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, value: dict[str, Any]) -> "ToolAction":
        name = str(value.get("name") or "")
        arguments = value.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        return cls(name=name, arguments=copy.deepcopy(arguments))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": copy.deepcopy(self.arguments)}


@dataclass(frozen=True)
class TrajectoryStep:
    """One expected decision point in a trajectory-backed gym episode."""

    index: int
    expected_actions: tuple[ToolAction, ...] = ()
    observations: tuple[dict[str, Any], ...] = ()
    kind: str = "tool"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def requires_tool(self) -> bool:
        return bool(self.expected_actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "expected_actions": [action.to_dict() for action in self.expected_actions],
            "observations": copy.deepcopy(list(self.observations)),
            "kind": self.kind,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class TrajectoryEpisode:
    """Unified parsed form shared by all thin dataset adapters."""

    id: str
    source: str
    tools: tuple[dict[str, Any], ...]
    initial_messages: tuple[dict[str, Any], ...]
    steps: tuple[TrajectoryStep, ...]
    domain: str | None = None
    task: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "domain": self.domain,
            "task": self.task,
            "tools": copy.deepcopy(list(self.tools)),
            "initial_messages": copy.deepcopy(list(self.initial_messages)),
            "steps": [step.to_dict() for step in self.steps],
            "metadata": copy.deepcopy(self.metadata),
        }

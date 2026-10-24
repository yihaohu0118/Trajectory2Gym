from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .schema import ToolAction, TrajectoryEpisode, TrajectoryStep


@dataclass(frozen=True)
class EvidencePointer:
    """Pointer to evidence used by a verifier or compiler decision."""

    source: str
    kind: str
    identifier: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "kind": self.kind,
            "identifier": self.identifier,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class TrajectoryIRStep:
    index: int
    assistant_tool_actions: tuple[ToolAction, ...] = ()
    tool_observations: tuple[dict[str, Any], ...] = ()
    kind: str = "tool"
    evidence_pointers: tuple[EvidencePointer, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_step(cls, step: TrajectoryStep, *, episode_id: str, benchmark: str) -> "TrajectoryIRStep":
        pointers = (
            EvidencePointer(
                source=benchmark,
                kind="trajectory_step",
                identifier=f"{episode_id}:{step.index}",
                metadata={"step_kind": step.kind},
            ),
        )
        return cls(
            index=step.index,
            assistant_tool_actions=tuple(step.expected_actions),
            tool_observations=tuple(copy.deepcopy(list(step.observations))),
            kind=step.kind,
            evidence_pointers=pointers,
            metadata=copy.deepcopy(step.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "assistant_tool_actions": [action.to_dict() for action in self.assistant_tool_actions],
            "tool_observations": copy.deepcopy(list(self.tool_observations)),
            "kind": self.kind,
            "evidence_pointers": [pointer.to_dict() for pointer in self.evidence_pointers],
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class TrajectoryIR:
    """Benchmark-neutral compiler input for Trajectory2Gym."""

    task_id: str
    benchmark: str
    domain: str | None
    tool_schemas: tuple[dict[str, Any], ...]
    visible_dialogue_context: tuple[dict[str, Any], ...]
    steps: tuple[TrajectoryIRStep, ...]
    trajectory_metadata: dict[str, Any] = field(default_factory=dict)
    benchmark_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_episode(cls, episode: TrajectoryEpisode) -> "TrajectoryIR":
        return cls(
            task_id=episode.id,
            benchmark=episode.source,
            domain=episode.domain,
            tool_schemas=tuple(copy.deepcopy(list(episode.tools))),
            visible_dialogue_context=tuple(copy.deepcopy(list(episode.initial_messages))),
            steps=tuple(
                TrajectoryIRStep.from_step(step, episode_id=episode.id, benchmark=episode.source)
                for step in episode.steps
            ),
            trajectory_metadata={"task": episode.task},
            benchmark_metadata=copy.deepcopy(episode.metadata),
        )

    def to_episode(self) -> TrajectoryEpisode:
        return TrajectoryEpisode(
            id=self.task_id,
            source=self.benchmark,
            domain=self.domain,
            task=str(self.trajectory_metadata.get("task") or ""),
            tools=tuple(copy.deepcopy(list(self.tool_schemas))),
            initial_messages=tuple(copy.deepcopy(list(self.visible_dialogue_context))),
            steps=tuple(
                TrajectoryStep(
                    index=step.index,
                    expected_actions=step.assistant_tool_actions,
                    observations=step.tool_observations,
                    kind=step.kind,
                    metadata=copy.deepcopy(step.metadata),
                )
                for step in self.steps
            ),
            metadata=copy.deepcopy(self.benchmark_metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "benchmark": self.benchmark,
            "domain": self.domain,
            "tool_schemas": copy.deepcopy(list(self.tool_schemas)),
            "visible_dialogue_context": copy.deepcopy(list(self.visible_dialogue_context)),
            "steps": [step.to_dict() for step in self.steps],
            "trajectory_metadata": copy.deepcopy(self.trajectory_metadata),
            "benchmark_metadata": copy.deepcopy(self.benchmark_metadata),
        }


def episode_to_ir(episode: TrajectoryEpisode) -> TrajectoryIR:
    return TrajectoryIR.from_episode(episode)

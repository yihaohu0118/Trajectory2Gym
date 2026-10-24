from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .schema import TrajectoryEpisode, TrajectoryStep


@dataclass(frozen=True)
class ExecutionNode:
    """A node in the recovered local execution graph."""

    id: str
    kind: str
    step_index: int
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "step_index": self.step_index,
            "label": self.label,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class ExecutionEdge:
    """A dependency edge recovered from trajectory structure."""

    source: str
    target: str
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class ExecutionGraph:
    """Recovered partial execution graph for one trajectory episode."""

    episode_id: str
    nodes: tuple[ExecutionNode, ...]
    edges: tuple[ExecutionEdge, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": copy.deepcopy(self.metadata),
        }

    def to_dot(self) -> str:
        lines = [f'digraph "{_dot_escape(self.episode_id)}" {{']
        lines.append("  rankdir=LR;")
        for node in self.nodes:
            label = f"{node.kind}:{node.label}"
            lines.append(f'  "{_dot_escape(node.id)}" [label="{_dot_escape(label)}"];')
        for edge in self.edges:
            lines.append(
                f'  "{_dot_escape(edge.source)}" -> "{_dot_escape(edge.target)}" '
                f'[label="{_dot_escape(edge.kind)}"];'
            )
        lines.append("}")
        return "\n".join(lines)


def build_execution_graph(episode: TrajectoryEpisode) -> ExecutionGraph:
    nodes: list[ExecutionNode] = []
    edges: list[ExecutionEdge] = []
    action_node_by_step: dict[int, str] = {}
    observation_node_by_step: dict[int, str] = {}

    for position, step in enumerate(episode.steps):
        action_id = f"step_{step.index}_action"
        observation_id = f"step_{step.index}_observation"
        action_label = _step_action_label(step)
        nodes.append(
            ExecutionNode(
                id=action_id,
                kind="action",
                step_index=step.index,
                label=action_label,
                metadata={"step_kind": step.kind, "position": position},
            )
        )
        action_node_by_step[step.index] = action_id

        if step.observations:
            nodes.append(
                ExecutionNode(
                    id=observation_id,
                    kind="observation",
                    step_index=step.index,
                    label=f"obs[{len(step.observations)}]",
                    metadata={"step_kind": step.kind, "position": position},
                )
            )
            observation_node_by_step[step.index] = observation_id
            edges.append(ExecutionEdge(source=action_id, target=observation_id, kind="yields_observation"))

        for dependency in _explicit_dependencies(step):
            source = observation_node_by_step.get(dependency) or action_node_by_step.get(dependency)
            if source is not None:
                edges.append(
                    ExecutionEdge(
                        source=source,
                        target=action_id,
                        kind="explicit_dependency",
                        metadata={"dependency_step": dependency},
                    )
                )

        if not _explicit_dependencies(step) and position > 0:
            previous_step = episode.steps[position - 1]
            source = observation_node_by_step.get(previous_step.index) or action_node_by_step.get(previous_step.index)
            if source is not None:
                edges.append(ExecutionEdge(source=source, target=action_id, kind="sequence_dependency"))

    return ExecutionGraph(
        episode_id=episode.id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        metadata={
            "source": episode.source,
            "domain": episode.domain,
            "step_count": episode.step_count,
        },
    )


def _step_action_label(step: TrajectoryStep) -> str:
    if not step.expected_actions:
        return step.kind
    return "+".join(action.name for action in step.expected_actions)


def _explicit_dependencies(step: TrajectoryStep) -> tuple[int, ...]:
    raw = step.metadata.get("dependency_list")
    if not isinstance(raw, list):
        return ()
    dependencies: list[int] = []
    for item in raw:
        try:
            dependencies.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(dependencies)


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')

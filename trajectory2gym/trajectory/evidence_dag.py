from __future__ import annotations

"""A small, benchmark-neutral runtime for static trajectory DAG evidence."""

import copy
import itertools
import json
from dataclasses import dataclass
from typing import Any

from .matcher import check_action_schema, match_actions, parse_assistant_actions
from .schema import ToolAction, TrajectoryEpisode, TrajectoryStep


TERMINAL_ACTIONS = {
    "ask_user_for_required_parameters",
    "prepare_to_answer",
}


@dataclass(frozen=True)
class DAGDecision:
    label: str
    decision: str
    reward: float
    observations: tuple[dict[str, Any], ...] = ()
    accepted_node_ids: tuple[int, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "decision": self.decision,
            "reward": self.reward,
            "observations": copy.deepcopy(list(self.observations)),
            "accepted_node_ids": list(self.accepted_node_ids),
            "reason": self.reason,
        }


class StaticEvidenceDAGGym:
    """Execute only transitions explicitly represented by a static DAG.

    Nodes become available when all declared dependencies are satisfied.
    Exact actions and controlled schema-guided equivalents return the recorded
    observation. Invalid actions receive a typed error. Schema-valid actions
    without DAG evidence abstain and never advance the state.
    """

    def __init__(
        self,
        episode: TrajectoryEpisode,
        *,
        max_rejected_attempts: int = 2,
    ) -> None:
        self.episode = episode
        self.tools = tuple(copy.deepcopy(list(episode.tools)))
        self.max_rejected_attempts = max_rejected_attempts
        self.nodes = {step.index: step for step in episode.steps}
        self.dependencies = {
            step.index: frozenset(
                int(value)
                for value in step.metadata.get("dependency_list", [])
                if isinstance(value, int) or str(value).isdigit()
            )
            for step in episode.steps
        }
        unknown_dependencies = {
            dependency
            for values in self.dependencies.values()
            for dependency in values
            if dependency not in self.nodes
        }
        if unknown_dependencies:
            raise ValueError(
                f"{episode.id}: unknown DAG dependencies "
                f"{sorted(unknown_dependencies)}"
            )
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.completed: set[int] = set()
        self.messages = copy.deepcopy(list(self.episode.initial_messages))
        self.rejected_attempts = 0
        self.done = False
        self.success = False
        self.trace: list[dict[str, Any]] = []
        return self.observe()

    def available_steps(self) -> tuple[TrajectoryStep, ...]:
        return tuple(
            self.nodes[index]
            for index in sorted(self.nodes)
            if index not in self.completed
            and self.dependencies[index].issubset(self.completed)
        )

    def observe(self) -> dict[str, Any]:
        available = self.available_steps() if not self.done else ()
        return {
            "messages": copy.deepcopy(self.messages),
            "tools": copy.deepcopy(list(self.tools)),
            "available_node_ids": [step.index for step in available],
            "completed_node_ids": sorted(self.completed),
            "done": self.done,
            "success": self.success,
        }

    def step(
        self,
        assistant_message: dict[str, Any] | str,
    ) -> tuple[dict[str, Any], DAGDecision]:
        if self.done:
            decision = DAGDecision(
                label="already_done",
                decision="reject",
                reward=0.0,
                reason="episode already terminated",
            )
            return self.observe(), decision

        available = self.available_steps()
        expected_names = {
            action.name
            for step in available
            for action in step.expected_actions
        }
        actions = parse_assistant_actions(
            assistant_message,
            expected_action_names=expected_names,
        )
        if actions:
            decision = self._step_actions(assistant_message, actions, available)
        else:
            decision = self._step_text(assistant_message, available)
        self.trace.append(
            {
                "assistant": copy.deepcopy(assistant_message),
                "decision": decision.to_dict(),
                "completed_node_ids": sorted(self.completed),
            }
        )
        return self.observe(), decision

    def _step_actions(
        self,
        assistant_message: dict[str, Any] | str,
        actions: tuple[ToolAction, ...],
        available: tuple[TrajectoryStep, ...],
    ) -> DAGDecision:
        available_tools = tuple(step for step in available if step.expected_actions)
        matched = self._match_available_actions(actions, available_tools)
        if matched is not None:
            matched_steps, label = matched
            self.messages.append(self._assistant_message(assistant_message))
            tool_call_ids = self._tool_call_ids(assistant_message, len(actions))
            observations: list[dict[str, Any]] = []
            for position, step in enumerate(matched_steps):
                self.completed.add(step.index)
                for observation in step.observations:
                    emitted = {
                        "role": "tool",
                        "content": self._content(observation.get("content")),
                        "tool_call_id": tool_call_ids[position],
                    }
                    self.messages.append(emitted)
                    observations.append(emitted)
            self.rejected_attempts = 0
            self._finish_if_exhausted()
            return DAGDecision(
                label=label,
                decision="accept",
                reward=1.0 if label == "exact_dag" else 0.75,
                observations=tuple(observations),
                accepted_node_ids=tuple(step.index for step in matched_steps),
                reason="recorded DAG transition",
            )

        known_names = {
            str((tool.get("function") or tool).get("name") or "")
            for tool in self.tools
        }
        if any(action.name not in known_names for action in actions):
            return self._reject(
                assistant_message,
                label="invalid_tool",
                reward=-0.45,
                error="unknown_tool",
                reason="tool is absent from the declared schema",
            )
        schema_checks = tuple(
            check_action_schema(action, self.tools) for action in actions
        )
        if any(not check.valid for check in schema_checks):
            return self._reject(
                assistant_message,
                label="invalid_schema",
                reward=-0.35,
                error="invalid_tool_arguments",
                reason="; ".join(
                    check.reason for check in schema_checks if not check.valid
                ),
            )
        return self._reject(
            assistant_message,
            label="abstain_unsupported",
            reward=-0.05,
            error="transition_unverified",
            reason="schema-valid action has no evidence at this DAG state",
        )

    def _step_text(
        self,
        assistant_message: dict[str, Any] | str,
        available: tuple[TrajectoryStep, ...],
    ) -> DAGDecision:
        text = self._assistant_text(assistant_message).strip()
        terminal_steps = [
            step
            for step in available
            if str((step.metadata.get("raw_action") or {}).get("name") or "")
            in TERMINAL_ACTIONS
            and self._normalise_text(
                step.metadata.get("expected_assistant_message")
            )
            == self._normalise_text(text)
        ]
        if text and terminal_steps:
            step = terminal_steps[0]
            terminal_name = str(step.metadata["raw_action"]["name"])
            self.messages.append(self._assistant_message(assistant_message))
            self.completed.add(step.index)
            observations = [
                {
                    "role": str(observation.get("role") or "user"),
                    "content": self._content(observation.get("content")),
                }
                for observation in step.observations
            ]
            if terminal_name == "ask_user_for_required_parameters":
                for emitted in observations:
                    self.messages.append(emitted)
            self._finish_if_exhausted()
            self.rejected_attempts = 0
            return DAGDecision(
                label=f"exact_dag_{terminal_name}",
                decision="accept",
                reward=1.0,
                observations=tuple(observations),
                accepted_node_ids=(step.index,),
                reason="recorded DAG terminal transition",
            )
        return self._reject(
            assistant_message,
            label="abstain_unverified_text",
            reward=-0.05,
            error="transition_unverified",
            reason="text response has no evidence at this DAG state",
        )

    @staticmethod
    def _normalise_text(value: Any) -> str:
        """Apply only representation-level whitespace normalisation."""

        return " ".join(str(value or "").split())

    def _match_available_actions(
        self,
        actions: tuple[ToolAction, ...],
        available: tuple[TrajectoryStep, ...],
    ) -> tuple[tuple[TrajectoryStep, ...], str] | None:
        if not actions or len(actions) > len(available):
            return None
        # WTB nodes contain one tool call each. Permutations preserve the
        # model's call order while allowing any dependency-valid topology.
        for candidate_steps in itertools.permutations(available, len(actions)):
            expected = tuple(
                step.expected_actions[0] for step in candidate_steps
            )
            result = match_actions(
                actual_actions=actions,
                expected_actions=expected,
                tools=self.tools,
            )
            if result.label == "gold":
                return tuple(candidate_steps), "exact_dag"
            if result.label == "equivalent_args":
                return tuple(candidate_steps), "equivalent_dag"
        return None

    def _reject(
        self,
        assistant_message: dict[str, Any] | str,
        *,
        label: str,
        reward: float,
        error: str,
        reason: str,
    ) -> DAGDecision:
        self.messages.append(self._assistant_message(assistant_message))
        actions = parse_assistant_actions(assistant_message)
        observations: list[dict[str, Any]] = []
        if actions:
            call_ids = self._tool_call_ids(assistant_message, len(actions))
            for call_id in call_ids:
                emitted = {
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "status_code": 400,
                            "error": error,
                            "message": (
                                "The environment cannot verify this action at "
                                "the current evidence-bounded state."
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    "tool_call_id": call_id,
                }
                self.messages.append(emitted)
                observations.append(emitted)
        else:
            emitted = {
                "role": "user",
                "content": (
                    "The previous response is not supported at the current "
                    "evidence-bounded state. Reconsider the task and tools."
                ),
            }
            self.messages.append(emitted)
            observations.append(emitted)
        self.rejected_attempts += 1
        if self.rejected_attempts > self.max_rejected_attempts:
            self.done = True
        return DAGDecision(
            label=label,
            decision="abstain" if label.startswith("abstain") else "reject",
            reward=reward,
            observations=tuple(observations),
            reason=reason,
        )

    def _finish_if_exhausted(self) -> None:
        if len(self.completed) == len(self.nodes):
            self.done = True
            self.success = True

    @staticmethod
    def _assistant_text(message: dict[str, Any] | str) -> str:
        if isinstance(message, str):
            return message
        return str(message.get("content") or "")

    @staticmethod
    def _assistant_message(
        message: dict[str, Any] | str,
    ) -> dict[str, Any]:
        if isinstance(message, str):
            return {"role": "assistant", "content": message}
        normalized = copy.deepcopy(message)
        normalized["role"] = "assistant"
        return normalized

    @staticmethod
    def _tool_call_ids(
        message: dict[str, Any] | str,
        count: int,
    ) -> list[str]:
        if isinstance(message, dict):
            calls = message.get("tool_calls") or []
            ids = [
                str(call.get("id") or f"dag_call_{index}")
                for index, call in enumerate(calls)
                if isinstance(call, dict)
            ]
            if len(ids) == count:
                return ids
        return [f"dag_call_{index}" for index in range(count)]

    @staticmethod
    def _content(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

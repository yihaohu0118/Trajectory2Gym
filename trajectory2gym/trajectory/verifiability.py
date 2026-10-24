from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .matcher import ActionMatchResult, match_step_action
from .observation_db import (
    ObservationDatabase,
    ObservationLookup,
    observation_context_key,
    prefix_has_state_change,
)
from .replay import replay_gold_observation
from .schema import TrajectoryEpisode, TrajectoryStep
from .boundary import AMBIGUOUS_LABELS, INVALID_LABELS, VALID_LABELS
from .tool_semantics import DEFAULT_TOOL_SEMANTICS, ToolSemantics
from .external_request import external_action_catalog_from_episodes


@dataclass(frozen=True)
class VerifiabilityDecision:
    """Sound partial-environment decision for one candidate action."""

    label: str
    action_class: str
    accepted: bool
    observation_messages: tuple[dict[str, Any], ...] = ()
    reason: str = ""
    match: ActionMatchResult | None = None
    observation_lookup: ObservationLookup | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.action_class == "valid"

    @property
    def is_invalid(self) -> bool:
        return self.action_class == "invalid"

    @property
    def is_ambiguous(self) -> bool:
        return self.action_class == "ambiguous"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "action_class": self.action_class,
            "accepted": self.accepted,
            "observation_messages": copy.deepcopy(list(self.observation_messages)),
            "reason": self.reason,
            "match": self.match.to_dict() if self.match is not None else None,
            "observation_lookup": self.observation_lookup.to_dict()
            if self.observation_lookup is not None
            else None,
            "metadata": copy.deepcopy(self.metadata),
        }


class VerifiabilityAnalyzer:
    """Classify candidate tool actions as valid, invalid, or ambiguous.

    This analyzer is intentionally stricter than the training reward code. Its
    job is to expose partial-environment boundaries for compiler analysis.
    """

    def __init__(
        self,
        *,
        observation_db: ObservationDatabase | None = None,
        accept_equivalent: bool = True,
        allow_step_replay: bool = True,
        tool_semantics: ToolSemantics = DEFAULT_TOOL_SEMANTICS,
        external_action_catalog: tuple[dict[str, Any], ...] = (),
    ):
        self.observation_db = observation_db
        self.accept_equivalent = accept_equivalent
        self.allow_step_replay = allow_step_replay
        self.tool_semantics = tool_semantics
        self.external_action_catalog = external_action_catalog

    @classmethod
    def from_episodes(
        cls,
        episodes: list[TrajectoryEpisode] | tuple[TrajectoryEpisode, ...],
        *,
        accept_equivalent: bool = True,
        allow_step_replay: bool = True,
        tool_semantics: ToolSemantics = DEFAULT_TOOL_SEMANTICS,
    ) -> "VerifiabilityAnalyzer":
        return cls(
            observation_db=ObservationDatabase.from_episodes(
                episodes, tool_semantics=tool_semantics
            ),
            accept_equivalent=accept_equivalent,
            allow_step_replay=allow_step_replay,
            tool_semantics=tool_semantics,
            external_action_catalog=external_action_catalog_from_episodes(episodes),
        )

    def classify(
        self,
        episode: TrajectoryEpisode,
        step: TrajectoryStep,
        action: dict[str, Any] | list[dict[str, Any]] | str,
    ) -> VerifiabilityDecision:
        match = match_step_action(
            step,
            _normalize_action(action),
            tools=episode.tools,
            external_action_catalog=self.external_action_catalog,
        )
        metadata = {"episode_id": episode.id, "step_index": step.index, "step_kind": step.kind}
        if step.kind == "terminal_stop":
            source_eligible = _terminal_stop_source_eligible(episode, step)
            metadata.update(
                {
                    "terminal_stop_decision": True,
                    "terminal_stop_source_eligible": source_eligible,
                    "terminal_stop_semantics_verified": False,
                    "terminal_stop_provenance": str(
                        step.metadata.get("terminal_stop_provenance") or ""
                    ),
                    "terminal_stop_synthetic": bool(
                        step.metadata.get("terminal_stop_synthetic", False)
                    ),
                }
            )
            if match.label == "terminal_stop" and not source_eligible:
                return _decision(
                    "ambiguous_terminal_stop_without_success_evidence",
                    "ambiguous",
                    match,
                    metadata,
                    reason=(
                        "terminal stop lacks explicit training-eligible and "
                        "terminal-success source evidence"
                    ),
                )
            if match.label == "terminal_stop":
                return VerifiabilityDecision(
                    label="valid_terminal_stop",
                    action_class="valid",
                    accepted=True,
                    observation_messages=(),
                    reason=(
                        "recorded successful episode boundary supports the stop "
                        "decision; response semantics are not verified"
                    ),
                    match=match,
                    metadata=metadata,
                )
        step_position = next(
            (position for position, candidate in enumerate(episode.steps) if candidate is step),
            next(
                (
                    position
                    for position, candidate in enumerate(episode.steps)
                    if candidate.index == step.index
                ),
                0,
            ),
        )

        if self.allow_step_replay:
            replay = replay_gold_observation(step, match, accept_equivalent=self.accept_equivalent)
        else:
            replay = None

        if replay is not None and replay.accepted:
            if match.label == "external_request_grounded":
                label = "valid_external_request_grounded"
            else:
                label = "valid_replay_exact" if match.exact else "valid_replay_equivalent"
            return VerifiabilityDecision(
                label=label,
                action_class="valid",
                accepted=True,
                observation_messages=replay.observation_messages,
                reason="trajectory evidence supports replay",
                match=match,
                metadata=metadata,
            )

        if match.label == "unknown_tool":
            return _decision("invalid_unknown_tool", "invalid", match, metadata, reason=match.reason)
        if match.label == "schema_invalid":
            return _decision("invalid_schema", "invalid", match, metadata, reason=match.reason)
        if match.label == "missing_tool_call":
            return _decision("invalid_missing_tool_call", "invalid", match, metadata, reason=match.reason)
        if match.label == "spurious_tool_call":
            return _decision("invalid_spurious_tool_call", "invalid", match, metadata, reason=match.reason)
        if match.label == "unverified_text":
            return _decision(
                "ambiguous_unverified_text",
                "ambiguous",
                match,
                metadata,
                reason=match.reason,
            )

        if (
            self.observation_db is not None
            and len(match.actual_actions) == 1
            and self.tool_semantics.allows_observation_db_replay(match.actual_actions[0].name)
        ):
            lookup = self.observation_db.lookup(
                match.actual_actions[0],
                context_key=observation_context_key(episode, step_position),
                allow_context_free=not prefix_has_state_change(
                    episode, step_position, tool_semantics=self.tool_semantics
                ),
            )
            if lookup.status == "hit":
                return VerifiabilityDecision(
                    label="valid_replay_database",
                    action_class="valid",
                    accepted=True,
                    observation_messages=lookup.observations,
                    reason="canonical action appears in compiled observation database",
                    match=match,
                    observation_lookup=lookup,
                    metadata=metadata,
                )
            if lookup.status == "conflict":
                return VerifiabilityDecision(
                    label="ambiguous_conflicting_observation",
                    action_class="ambiguous",
                    accepted=False,
                    reason=lookup.reason,
                    match=match,
                    observation_lookup=lookup,
                    metadata=metadata,
                )
            if lookup.status == "context_required":
                return VerifiabilityDecision(
                    label="ambiguous_context_dependent_observation",
                    action_class="ambiguous",
                    accepted=False,
                    reason=lookup.reason,
                    match=match,
                    observation_lookup=lookup,
                    metadata=metadata,
                )

        if match.same_tool_names and match.schema_valid:
            return _decision(
                "ambiguous_unseen_query",
                "ambiguous",
                match,
                metadata,
                reason="schema-valid same-tool action has no supported observation",
            )
        if match.schema_valid:
            return _decision(
                "ambiguous_wrong_tool",
                "ambiguous",
                match,
                metadata,
                reason="schema-valid action uses a different known tool than the trajectory step",
            )
        return _decision(
            "ambiguous_observation_missing",
            "ambiguous",
            match,
            metadata,
            reason="no reliable observation can be recovered",
        )


def _decision(
    label: str,
    action_class: str,
    match: ActionMatchResult,
    metadata: dict[str, Any],
    *,
    reason: str,
) -> VerifiabilityDecision:
    return VerifiabilityDecision(
        label=label,
        action_class=action_class,
        accepted=False,
        reason=reason,
        match=match,
        metadata=metadata,
    )


def _normalize_action(action: dict[str, Any] | list[dict[str, Any]] | str) -> dict[str, Any] | str:
    if isinstance(action, list):
        if action and all(isinstance(item, dict) and "name" in item and "role" not in item for item in action):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": item.get("name"),
                            "arguments": item.get("arguments", {}),
                        },
                    }
                    for item in action
                ],
            }
        return copy.deepcopy(action[-1]) if action else {"role": "assistant", "content": ""}
    if isinstance(action, dict) and "name" in action:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": action.get("name"),
                        "arguments": action.get("arguments", {}),
                    },
                }
            ],
        }
    return copy.deepcopy(action)


def _terminal_stop_source_eligible(
    episode: TrajectoryEpisode, step: TrajectoryStep
) -> bool:
    """Validate a stop certificate against explicit generic source outcome fields."""

    metadata = episode.metadata if isinstance(episode.metadata, dict) else {}
    return bool(
        step.kind == "terminal_stop"
        and not step.expected_actions
        and not step.observations
        and step.metadata.get("verified_terminal_stop") is True
        and step.metadata.get("terminal_stop_provenance")
        == "recorded_successful_episode_boundary"
        and step.metadata.get("terminal_stop_semantics_verified") is False
        and metadata.get("source_training_eligible") is True
        and metadata.get("source_terminal_success_eligible") is True
    )

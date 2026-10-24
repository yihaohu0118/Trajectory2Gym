from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .matcher import ActionMatchResult
from .schema import TrajectoryStep


@dataclass(frozen=True)
class ReplayDecision:
    """Policy-free replay decision for a parsed trajectory step."""

    accepted: bool
    observation_messages: tuple[dict[str, Any], ...] = ()
    label: str = ""
    reward_hint: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "observation_messages": copy.deepcopy(list(self.observation_messages)),
            "label": self.label,
            "reward_hint": self.reward_hint,
            "metadata": copy.deepcopy(self.metadata),
        }


def replay_gold_observation(
    step: TrajectoryStep,
    match: ActionMatchResult,
    *,
    accept_equivalent: bool = True,
) -> ReplayDecision:
    """Return stored observations for neutral accepted matches.

    Domain adapters can override the reward and acceptance policy. This helper
    only centralizes the common "gold/equivalent action replays trace output"
    behavior shared by trajectory2gym environments.
    """

    accepted_labels = {
        "gold",
        "gold_text",
        "external_request_grounded",
        "terminal_stop",
    }
    if accept_equivalent:
        accepted_labels.add("equivalent_args")
    accepted = match.label in accepted_labels
    observation_messages = (
        ()
        if match.label == "terminal_stop"
        else tuple(copy.deepcopy(list(step.observations)))
    )
    return ReplayDecision(
        accepted=accepted,
        observation_messages=observation_messages if accepted else (),
        label=match.label,
        reward_hint=_reward_hint(match.label),
        metadata={"step_index": step.index, "step_kind": step.kind},
    )


def _reward_hint(label: str) -> float:
    if label in {"gold", "gold_text", "external_request_grounded", "terminal_stop"}:
        return 1.0
    if label == "equivalent_args":
        return 0.5
    return 0.0

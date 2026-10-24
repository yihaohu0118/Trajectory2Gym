from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .boundary import boundary_class, boundary_decision, boundary_definition, boundary_runtime_rule


@dataclass(frozen=True)
class Decision:
    """Shared accept / reject / abstain API for Trajectory2Gym compilers."""

    status: str
    action: str
    subtype: str
    runtime_label: str = ""
    grounded_observation: tuple[dict[str, Any], ...] = ()
    reward_signal: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    hallucinated_obs: bool = False
    verifier_name: str = "trajectory2gym"
    typed_feedback: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_subtype(
        cls,
        subtype: str,
        *,
        grounded_observation: tuple[dict[str, Any], ...] = (),
        reward_signal: float = 0.0,
        evidence: dict[str, Any] | None = None,
        hallucinated_obs: bool = False,
        verifier_name: str = "trajectory2gym",
        typed_feedback: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "Decision":
        runtime_label = subtype
        status = boundary_class(runtime_label)
        return cls(
            status=status,
            action=boundary_decision(runtime_label),
            subtype=canonical_subtype(runtime_label),
            runtime_label=runtime_label,
            grounded_observation=tuple(copy.deepcopy(list(grounded_observation))),
            reward_signal=float(reward_signal),
            evidence=copy.deepcopy(dict(evidence or {})),
            hallucinated_obs=bool(hallucinated_obs),
            verifier_name=verifier_name,
            typed_feedback=typed_feedback,
            metadata=copy.deepcopy(dict(metadata or {})),
        )

    @property
    def accepted(self) -> bool:
        return self.status == "valid" and self.action == "accept"

    def assert_sound(self, *, allow_unsafe: bool = False) -> None:
        if self.status == "ambiguous" and self.grounded_observation and not allow_unsafe:
            raise ValueError("ambiguous decisions must abstain without observation")
        if self.status == "invalid" and self.grounded_observation and not allow_unsafe:
            raise ValueError("invalid decisions must not return grounded observations")
        if self.hallucinated_obs and not allow_unsafe:
            raise ValueError("hallucinated observations are not allowed in strict/partial modes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "action": self.action,
            "subtype": self.subtype,
            "runtime_label": self.runtime_label,
            "grounded_observation": copy.deepcopy(list(self.grounded_observation)),
            "reward_signal": self.reward_signal,
            "evidence": copy.deepcopy(self.evidence),
            "hallucinated_obs": self.hallucinated_obs,
            "verifier_name": self.verifier_name,
            "typed_feedback": self.typed_feedback,
            "metadata": copy.deepcopy(self.metadata),
            "definition": boundary_definition(self.runtime_label or self.subtype),
            "runtime_rule": boundary_runtime_rule(self.runtime_label or self.subtype),
        }


def canonical_subtype(label: str) -> str:
    mapping = {
        "valid_replay_exact": "exact_replay",
        "valid_replay_equivalent": "equivalent_args",
        "canonical_equivalent_args": "equivalent_args",
        "equivalent_args": "equivalent_args",
        "valid_replay_database": "valid_replay_database",
        "valid_terminal_stop": "terminal_stop",
        "valid_topology_verified": "topology_valid",
        "valid_topology_prefix": "prefix_valid",
        "invalid_schema": "schema_invalid",
        "invalid_unknown_tool": "wrong_tool",
        "invalid_missing_tool_call": "wrong_tool",
        "invalid_spurious_tool_call": "wrong_tool",
        "invalid_topology_wrong_tool": "wrong_tool",
        "invalid_topology_wrong_args": "wrong_args",
        "invalid_topology_dependency": "dependency_error",
        "invalid_topology_premature_text": "dependency_error",
        "ambiguous_conflicting_observation": "conflict",
        "ambiguous_topology_conflict": "conflict",
        "ambiguous_hidden_state": "hidden_state_required",
        "ambiguous_unseen_query": "unverifiable",
        "ambiguous_wrong_tool": "unverifiable",
        "ambiguous_observation_missing": "unverifiable",
        "ambiguous_repeated_action": "unverifiable",
        "ambiguous_topology_unsupported": "unverifiable",
        "ambiguous_topology_review": "unverifiable",
        "ambiguous_terminal_stop_without_success_evidence": "unverifiable",
    }
    return mapping.get(label, label)


def decision_from_boundary_payload(payload: dict[str, Any]) -> Decision:
    return Decision.from_subtype(
        str(payload.get("label") or payload.get("subtype") or ""),
        evidence={
            "source": payload.get("evidence"),
            "reason": payload.get("reason"),
            "boundary_payload": copy.deepcopy(payload),
        },
        hallucinated_obs=bool(payload.get("hallucinated_obs", False)),
        verifier_name=str(payload.get("verifier_name") or "trajectory2gym_boundary"),
        metadata=copy.deepcopy(dict(payload.get("metadata") or {})),
    )

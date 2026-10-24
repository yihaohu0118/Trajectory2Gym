from __future__ import annotations

from typing import Any


VALID_LABELS = {
    "valid_replay_exact",
    "valid_replay_equivalent",
    "valid_replay_database",
    "valid_topology_verified",
    "valid_topology_prefix",
    "valid_external_request_grounded",
    "valid_terminal_stop",
    "valid_replay_recorded_state_graph",
}

# Strict replay accepts only the transition currently exposed by one recorded
# trajectory cursor (plus the two structural forms of that same transition).
# Keeping this set in the generic boundary module makes the Partial-vs-Strict
# counterfactual auditable without importing a benchmark adapter.
STRICT_REPLAY_ACCEPTED_LABELS = frozenset(
    {
        "valid_replay_exact",
        "valid_external_request_grounded",
        "valid_terminal_stop",
    }
)

# These branches are the strong, evidence-reuse part of Partial Gym.  They are
# deliberately separated from controlled representation equivalence and the
# legacy trajectory-trie experiment so a successful rollout cannot claim a
# causal Partial gain merely because JSON formatting was normalized.
PARTIAL_EVIDENCE_BRANCH_LABELS = frozenset(
    {
        "valid_replay_database",
        "valid_replay_recorded_state_graph",
    }
)

PARTIAL_EXTENSION_LABELS = frozenset(
    {
        *PARTIAL_EVIDENCE_BRANCH_LABELS,
        "valid_replay_equivalent",
        "valid_replay_alternative",
    }
)

INVALID_LABELS = {
    "invalid_schema",
    "invalid_unknown_tool",
    "invalid_missing_tool_call",
    "invalid_spurious_tool_call",
    "invalid_topology_wrong_args",
    "invalid_topology_wrong_tool",
    "invalid_topology_dependency",
    "invalid_topology_premature_text",
}

AMBIGUOUS_LABELS = {
    "ambiguous_unseen_query",
    "ambiguous_conflicting_observation",
    "ambiguous_wrong_tool",
    "ambiguous_observation_missing",
    "ambiguous_repeated_action",
    "ambiguous_topology_unsupported",
    "ambiguous_topology_conflict",
    "ambiguous_topology_review",
    "ambiguous_terminal_stop_without_success_evidence",
}


BOUNDARY_DEFINITIONS = {
    "valid_replay_exact": "candidate action exactly matches the recorded trajectory step",
    "valid_replay_equivalent": "candidate action is schema-valid and equivalent to the recorded step under local matching",
    "valid_replay_database": "candidate action has a conflict-free canonical action key in the compiled observation database",
    "valid_topology_verified": "candidate action is accepted by a dataset-provided topology or dependency verifier",
    "valid_topology_prefix": "a verifier accepts a sound prefix while rejecting or ignoring unsupported extra actions",
    "valid_external_request_grounded": "assistant text explicitly grounds every recorded external action before the stored event is replayed",
    "valid_terminal_stop": "a non-empty assistant response stops at an explicitly successful recorded episode boundary; response semantics are not verified",
    "valid_replay_recorded_state_graph": "candidate action follows a context-certified unique authentic edge from the exact current recorded state to an explicit successor state with a successful recorded continuation or complete local block",
    "invalid_schema": "candidate uses a known tool but violates its schema",
    "invalid_unknown_tool": "candidate references no known tool schema",
    "invalid_missing_tool_call": "candidate omits a required tool call",
    "invalid_spurious_tool_call": "candidate emits a tool call where the trajectory step is terminal text",
    "invalid_topology_wrong_args": "candidate is schema-valid but violates topology/task-bound argument constraints",
    "invalid_topology_wrong_tool": "candidate uses a known tool rejected by the topology verifier for the current transition",
    "invalid_topology_dependency": "candidate violates explicit dependency-order constraints",
    "invalid_topology_premature_text": "candidate emits final text before the topology permits termination",
    "ambiguous_unseen_query": "candidate is schema-valid and semantically plausible but has no recorded observation evidence",
    "ambiguous_conflicting_observation": "candidate key appears in trajectories with conflicting observations",
    "ambiguous_wrong_tool": "candidate uses a known but unsupported tool for this transition",
    "ambiguous_observation_missing": "candidate cannot be paired with a reliable observation",
    "ambiguous_repeated_action": "candidate repeats an evidence-backed action that returns no new observation or trajectory progress",
    "ambiguous_topology_unsupported": "candidate is schema-valid but has no topology or observation support",
    "ambiguous_topology_conflict": "topology and schema/argument audits disagree, so the branch is not promoted",
    "ambiguous_topology_review": "candidate requires dataset-specific review before a sound boundary decision",
    "ambiguous_terminal_stop_without_success_evidence": "candidate attempts to use a terminal-stop certificate without explicit generic source-success eligibility",
}


def boundary_class(label: str) -> str:
    if label in VALID_LABELS or label.startswith("valid_"):
        return "valid"
    if label in INVALID_LABELS or label.startswith("invalid_"):
        return "invalid"
    if label in AMBIGUOUS_LABELS or label.startswith("ambiguous_"):
        return "ambiguous"
    return "unknown"


def boundary_decision(label: str) -> str:
    action_class = boundary_class(label)
    if action_class == "valid":
        return "accept"
    if action_class == "invalid":
        return "typed_error"
    if action_class == "ambiguous":
        return "abstain"
    return "reject"


def boundary_runtime_rule(label: str) -> str:
    if label == "valid_terminal_stop":
        return "accept the stop decision without generating an observation"
    decision = boundary_decision(label)
    if decision == "accept":
        return "return verified observation or topology-approved feedback"
    if decision == "typed_error":
        return "return typed error"
    if decision == "abstain":
        return "abstain without observation"
    return "reject"


def boundary_definition(label: str) -> str:
    return BOUNDARY_DEFINITIONS.get(label, "compiler boundary label")


def make_boundary_payload(
    label: str,
    *,
    evidence: str,
    reason: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action_class = boundary_class(label)
    return {
        "label": label,
        "action_class": action_class,
        "decision": boundary_decision(label),
        "runtime_rule": boundary_runtime_rule(label),
        "definition": boundary_definition(label),
        "accepted": action_class == "valid",
        "evidence": evidence,
        "reason": reason,
        "metadata": dict(metadata or {}),
    }

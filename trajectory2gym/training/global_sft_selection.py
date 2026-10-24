"""Benchmark-neutral global selection for environment-certified SFT rows."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


JsonObject = dict[str, Any]
FORMAT = "trajectory2gym_global_sft_selection_v1"

_READ_PREFIXES = (
    "get", "list", "search", "find", "check", "fetch", "lookup", "retrieve",
    "view", "read", "calculate", "compute",
)
_IRREVERSIBLE_PREFIXES = (
    "delete", "cancel", "close", "terminate", "transfer", "pay", "purchase",
    "book", "submit", "send", "refund", "return",
)
_HANDOFF_MARKERS = ("handoff", "transfer_to_human", "escalat")
_ERROR_PREFIXES = ("error:", "error ", "failed:", "failure:", "invalid:")
_OBJECT_ONTOLOGY = {
    "identity": {"user", "customer", "profile", "account", "auth", "member"},
    "commerce": {"order", "item", "product", "refund", "return", "payment", "purchase"},
    "travel": {"flight", "reservation", "booking", "baggage", "airport", "trip"},
    "finance": {"card", "bank", "cash", "transfer", "balance", "credit", "debit", "transaction"},
    "connectivity": {"data", "wifi", "mms", "network", "sim", "phone", "app", "roaming"},
    "communication": {"message", "email", "sms", "notification"},
    "location": {"address", "location", "city", "country", "state", "zip"},
    "time": {"date", "time", "schedule", "deadline"},
}
_ARGUMENT_ONTOLOGY = {
    "identifier": {"id", "identifier", "number", "code", "reference"},
    "amount": {"amount", "price", "cost", "fee", "balance", "total"},
    "temporal": {"date", "time", "day", "month", "year", "duration"},
    "location": {"address", "city", "country", "state", "zip", "origin", "destination"},
    "contact": {"email", "phone", "name"},
    "status": {"status", "state", "reason", "mode", "type"},
    "quantity": {"count", "quantity", "number", "limit"},
    "content": {"text", "message", "query", "description", "expression"},
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _bucket(value: int, bounds: Sequence[int]) -> str:
    for bound in bounds:
        if value <= bound:
            return f"le{bound}"
    return f"gt{bounds[-1]}"


def _call_function(call: Mapping[str, Any]) -> tuple[str, JsonObject]:
    function = call.get("function")
    if isinstance(function, Mapping):
        name = str(function.get("name") or "")
        raw_arguments = function.get("arguments")
    else:
        name = str(call.get("name") or "")
        raw_arguments = call.get("arguments")
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {"_unparsed": raw_arguments}
    elif isinstance(raw_arguments, Mapping):
        arguments = dict(raw_arguments)
    else:
        arguments = {}
    return name, arguments


def _access_class(name: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if any(marker in lowered for marker in _HANDOFF_MARKERS):
        return "handoff"
    if lowered.startswith(_READ_PREFIXES):
        return "read"
    if lowered.startswith(_IRREVERSIBLE_PREFIXES):
        return "irreversible_write"
    return "write"


def _name_tokens(name: str) -> set[str]:
    return {
        token
        for token in re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        .lower()
        .replace(".", "_")
        .split("_")
        if token
    }


def _object_class(name: str) -> str:
    tokens = _name_tokens(name)
    matches = [label for label, vocabulary in _OBJECT_ONTOLOGY.items() if tokens & vocabulary]
    return "+".join(sorted(matches)) if matches else "generic"


def _argument_role(name: str) -> str:
    tokens = _name_tokens(name)
    matches = [label for label, vocabulary in _ARGUMENT_ONTOLOGY.items() if tokens & vocabulary]
    return "+".join(sorted(matches)) if matches else "generic"


def _value_shape(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, Mapping):
        return "object"
    return "other"


def _content_value(content: Any) -> Any:
    if isinstance(content, (Mapping, list)):
        return content
    text = str(content or "").strip()
    if text.startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


def _is_error_observation(message: Mapping[str, Any]) -> bool:
    if message.get("error") is True:
        return True
    value = _content_value(message.get("content"))
    if isinstance(value, Mapping):
        status = str(value.get("status") or "").lower()
        return bool(
            value.get("error") not in (None, "", False)
            or value.get("success") is False
            or status in {"error", "failed", "failure", "invalid"}
        )
    return str(value).lower().startswith(_ERROR_PREFIXES)


def _observation_shape(message: Mapping[str, Any]) -> str:
    if _is_error_observation(message):
        return "error"
    value = _content_value(message.get("content"))
    if value in (None, "", [], {}):
        return "empty"
    return _value_shape(value)


def _assistant_positions(messages: Sequence[Mapping[str, Any]]) -> list[int]:
    return [index for index, message in enumerate(messages) if message.get("role") == "assistant"]


def _decision_kind(message: Mapping[str, Any]) -> str:
    calls = list(message.get("tool_calls") or [])
    if len(calls) > 1:
        return "multi_call"
    if calls:
        names = [_call_function(call)[0] for call in calls if isinstance(call, Mapping)]
        if names and all(_access_class(name) == "handoff" for name in names):
            return "handoff"
        return "tool_call"
    text = str(message.get("content") or "").strip().lower()
    if any(marker in text for marker in ("transfer", "human agent", "specialist")):
        return "handoff_text"
    return "text"


def _prior_visible_text(messages: Sequence[Mapping[str, Any]], stop: int) -> str:
    return "\n".join(
        _canonical(message.get("content"))
        for message in messages[:stop]
        if message.get("role") in {"user", "tool", "external"}
    )


def capability_atoms(row: Mapping[str, Any]) -> frozenset[str]:
    """Return structural atoms without benchmark/category or exact tool names."""

    messages = [message for message in row.get("messages") or [] if isinstance(message, Mapping)]
    assistant_positions = _assistant_positions(messages)
    supervised = {int(value) for value in row.get("supervised_assistant_turn_indices") or []}
    atoms: set[str] = set()
    access_path: list[str] = []
    object_path: list[str] = []
    previous_assistant_position = -1
    for ordinal, position in enumerate(assistant_positions):
        if ordinal not in supervised:
            previous_assistant_position = position
            continue
        message = messages[position]
        kind = _decision_kind(message)
        calls = [call for call in message.get("tool_calls") or [] if isinstance(call, Mapping)]
        atoms.add(f"decision:{kind}")
        atoms.add(f"horizon:{_bucket(ordinal + 1, (1, 2, 4, 8, 16, 32))}")
        atoms.add(f"call_count:{_bucket(len(calls), (0, 1, 2, 4, 8))}")
        recent_tool_messages = [
            item
            for item in messages[previous_assistant_position + 1 : position]
            if item.get("role") == "tool"
        ]
        if any(_is_error_observation(item) for item in recent_tool_messages):
            atoms.add("recovery:after_error")
        prior = _prior_visible_text(messages, position)
        for call in calls:
            name, arguments = _call_function(call)
            access = _access_class(name)
            object_class = _object_class(name)
            access_path.append(access)
            object_path.append(object_class)
            atoms.add(f"access:{access}")
            atoms.add(f"object:{object_class}")
            atoms.add(f"argument_count:{_bucket(len(arguments), (0, 1, 2, 4, 8))}")
            shapes = sorted(_value_shape(value) for value in arguments.values())
            atoms.add("argument_shapes:" + ("+".join(shapes) if shapes else "none"))
            roles = sorted(_argument_role(key) for key in arguments)
            atoms.add("argument_roles:" + ("+".join(roles) if roles else "none"))
            grounded = sum(
                isinstance(value, str) and bool(value) and value in prior
                for value in arguments.values()
            )
            atoms.add(f"grounded_arguments:{_bucket(grounded, (0, 1, 2, 4))}")
        observations: list[Mapping[str, Any]] = []
        cursor = position + 1
        while cursor < len(messages) and messages[cursor].get("role") == "tool":
            observations.append(messages[cursor])
            cursor += 1
        if calls:
            atoms.add(f"observation_count:{_bucket(len(observations), (0, 1, 2, 4, 8))}")
            for observation in observations:
                atoms.add(f"observation:{_observation_shape(observation)}")
                value = _content_value(observation.get("content"))
                if isinstance(value, Mapping):
                    roles = sorted(_argument_role(str(key)) for key in value)
                    atoms.add("observation_roles:" + ("+".join(roles) if roles else "none"))
        later = [
            messages[next_position]
            for next_position in assistant_positions
            if next_position > position
        ]
        next_kind = _decision_kind(later[0]) if later else "terminal"
        atoms.add(f"transition:{kind}->{next_kind}")
        previous_assistant_position = position
    for left, right in zip(access_path, access_path[1:]):
        atoms.add(f"access_transition:{left}->{right}")
    for left, right in zip(object_path, object_path[1:]):
        atoms.add(f"object_transition:{left}->{right}")
    if access_path:
        atoms.add("access_path_prefix:" + ">".join(access_path[:4]))
    if object_path:
        atoms.add("object_path_prefix:" + ">".join(object_path[:4]))
    atoms.add(f"trajectory_decisions:{_bucket(len(supervised), (1, 2, 4, 8, 16, 32))}")
    return frozenset(atoms)


def _nested_feature(row: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    curation = row.get("curation") or {}
    learning = curation.get("learning_value") or {}
    features = learning.get("features") or {}
    value = features.get(name, default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _teacher_family(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    value = (
        row.get("teacher_family")
        or metadata.get("teacher_family")
        or metadata.get("teacher_model")
        or row.get("source")
        or "unknown"
    )
    return _digest(str(value))[:16]


def _task_identity(row: Mapping[str, Any]) -> str:
    value = str(row.get("source_task_id") or row.get("task_id") or "")
    if not value:
        raise ValueError("candidate row lacks task identity")
    return value


def _trace_identity(row: Mapping[str, Any]) -> str:
    value = str(row.get("trace_hash") or "")
    if not value:
        raise ValueError("candidate row lacks trace hash")
    return value


def _semantic_identity(row: Mapping[str, Any]) -> str:
    return _digest(
        {
            "messages": row.get("messages") or [],
            "tools": row.get("tools") or [],
            "supervised_assistant_turn_indices": row.get(
                "supervised_assistant_turn_indices"
            )
            or [],
            "verified_assistant_turn_indices": row.get(
                "verified_assistant_turn_indices"
            )
            or [],
        }
    )


def _declared_family_identity(row: Mapping[str, Any]) -> str:
    return str(
        row.get("trajectory_family_id")
        or row.get("counterfactual_family_id")
        or _task_identity(row)
    )


def _capability_path_family_identity(row: Mapping[str, Any]) -> str:
    """Identify a policy-visible structural path without provenance labels.

    Candidate corpora frequently use the task id as ``trajectory_family_id``.
    That is useful provenance, but it makes every task look path-diverse even
    when thousands of trajectories express the same one-decision behavior.
    The selector needs a structural family for diversity accounting and caps;
    the ordered capability path deliberately excludes benchmark, domain, task,
    exact tool names, and argument values.
    """

    return _digest(capability_path_signature(row))


def _decision_signatures(row: Mapping[str, Any]) -> list[str]:
    messages = [message for message in row.get("messages") or [] if isinstance(message, Mapping)]
    assistant = [message for message in messages if message.get("role") == "assistant"]
    supervised = {int(value) for value in row.get("supervised_assistant_turn_indices") or []}
    signatures: list[str] = []
    for ordinal, message in enumerate(assistant):
        if ordinal not in supervised:
            continue
        calls = []
        for call in message.get("tool_calls") or []:
            if not isinstance(call, Mapping):
                continue
            name, arguments = _call_function(call)
            calls.append((_access_class(name), sorted(_value_shape(v) for v in arguments.values())))
        signatures.append(_canonical((_decision_kind(message), calls)))
    return signatures


def capability_path_signature(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return an ordered, benchmark-neutral policy-visible decision path.

    ``capability_atoms`` deliberately describes set-valued coverage.  This
    companion signature retains the order of structurally distinct decisions
    and authentic observation outcomes without admitting exact tool names,
    argument values, task identity, benchmark, domain, or evaluation labels.
    It is therefore suitable for measuring within-capability path support.
    """

    messages = [message for message in row.get("messages") or [] if isinstance(message, Mapping)]
    assistant_positions = _assistant_positions(messages)
    supervised = {int(value) for value in row.get("supervised_assistant_turn_indices") or []}
    result: list[str] = []
    previous_assistant_position = -1
    for ordinal, position in enumerate(assistant_positions):
        if ordinal not in supervised:
            previous_assistant_position = position
            continue
        message = messages[position]
        prior = _prior_visible_text(messages, position)
        calls = []
        for call in message.get("tool_calls") or []:
            if not isinstance(call, Mapping):
                continue
            name, arguments = _call_function(call)
            calls.append(
                {
                    "access": _access_class(name),
                    "object": _object_class(name),
                    "arguments": sorted(
                        (_argument_role(str(key)), _value_shape(value))
                        for key, value in arguments.items()
                    ),
                    "grounded_arguments": _bucket(
                        sum(
                            isinstance(value, str) and bool(value) and value in prior
                            for value in arguments.values()
                        ),
                        (0, 1, 2, 4),
                    ),
                }
            )
        observations = []
        cursor = position + 1
        while cursor < len(messages) and messages[cursor].get("role") == "tool":
            observation = messages[cursor]
            value = _content_value(observation.get("content"))
            roles = (
                sorted(_argument_role(str(key)) for key in value)
                if isinstance(value, Mapping)
                else []
            )
            observations.append(
                {"shape": _observation_shape(observation), "roles": roles}
            )
            cursor += 1
        recent = messages[previous_assistant_position + 1 : position]
        result.append(
            _canonical(
                {
                    "decision": _decision_kind(message),
                    "calls": calls,
                    "observations": observations,
                    "after_error": any(
                        item.get("role") == "tool" and _is_error_observation(item)
                        for item in recent
                    ),
                }
            )
        )
        previous_assistant_position = position
    return tuple(result)


def _certified_exact_adjacent_loop_redundancy(row: Mapping[str, Any]) -> float:
    """Count only evidence-equivalent immediate repeated policy decisions.

    Coarse capability signatures are intentionally unsuitable here: two reads
    or writes of the same structural class may occur at distinct states in a
    valid workflow. Exact action plus authentic successor observation must
    repeat on adjacent assistant decisions. An official action-only
    certificate licenses imitation but provides no observation edge, so it is
    never treated as evidence of a redundant transition.
    """
    messages = [message for message in row.get("messages") or [] if isinstance(message, Mapping)]
    assistant_positions = _assistant_positions(messages)
    supervised = {int(value) for value in row.get("supervised_assistant_turn_indices") or []}
    action_only = {int(value) for value in row.get("verified_assistant_turn_indices") or []}
    previous_ordinal: int | None = None
    previous_key: str | None = None
    repeats = 0
    decisions = 0
    for ordinal in sorted(supervised):
        position = assistant_positions[ordinal]
        message = messages[position]
        calls = [call for call in message.get("tool_calls") or [] if isinstance(call, Mapping)]
        decisions += 1
        key = None
        if calls and ordinal not in action_only:
            observations = []
            cursor = position + 1
            while cursor < len(messages) and messages[cursor].get("role") == "tool":
                observations.append(messages[cursor])
                cursor += 1
            if observations:
                key = _canonical(
                    {
                        "calls": [
                            {"name": _call_function(call)[0], "arguments": _call_function(call)[1]}
                            for call in calls
                        ],
                        "observations": [
                            {
                                "content": observation.get("content"),
                                "error": observation.get("error"),
                            }
                            for observation in observations
                        ],
                    }
                )
        elif not calls:
            key = _canonical({"text": str(message.get("content") or "").strip()})
        if (
            key is not None
            and previous_key == key
            and previous_ordinal is not None
            and ordinal == previous_ordinal + 1
        ):
            repeats += 1
        previous_ordinal, previous_key = ordinal, key
    return repeats / max(1, decisions)


def _validate_candidate(row: Mapping[str, Any]) -> None:
    if row.get("success") is not True:
        raise ValueError("global SFT candidate must be a successful trajectory")
    if row.get("enable_thinking") is not False:
        raise ValueError("global SFT candidate must be non-thinking")
    if not row.get("supervised_assistant_turn_indices"):
        raise ValueError("global SFT candidate has no supervised assistant decision")
    termination = str((row.get("metadata") or {}).get("termination_reason") or "").lower()
    if any(marker in termination for marker in ("infrastructure", "overflow", "exception")):
        raise ValueError("infrastructure-invalid trajectory entered candidate pool")
    messages = [message for message in row.get("messages") or [] if isinstance(message, Mapping)]
    assistant_positions = _assistant_positions(messages)
    supervised = {int(value) for value in row.get("supervised_assistant_turn_indices") or []}
    action_only_certified = {
        int(value) for value in row.get("verified_assistant_turn_indices") or []
    }
    if any(value < 0 or value >= len(assistant_positions) for value in supervised):
        raise ValueError("supervised assistant index is out of range")
    for ordinal in supervised:
        position = assistant_positions[ordinal]
        calls = [
            call
            for call in messages[position].get("tool_calls") or []
            if isinstance(call, Mapping)
        ]
        if not calls:
            continue
        observations: list[Mapping[str, Any]] = []
        cursor = position + 1
        while cursor < len(messages) and messages[cursor].get("role") == "tool":
            observations.append(messages[cursor])
            cursor += 1
        if any(_is_error_observation(item) for item in observations):
            raise ValueError("error-producing tool action is a supervised target")
        if len(observations) < len(calls) and ordinal not in action_only_certified:
            raise ValueError("tool target lacks official observation or action-only certificate")
    capability_atoms(row)


@dataclass(frozen=True)
class Candidate:
    index: int
    row: JsonObject
    trace_hash: str
    task: str
    teacher: str
    teacher_provenance_explicit: bool
    family: str
    tokens: int
    atoms: frozenset[str]
    recovery: float
    branch: float
    cost_efficiency: float
    loop_redundancy: float
    unsupported_reference_risk: float
    policy_gap: float


def prepare_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    supervised_tokens: Mapping[str, int],
    require_multiscale_policy_gap: bool = False,
    loop_redundancy_mode: str = "coarse_signature_v1",
    path_family_mode: str = "declared_or_task_v1",
) -> tuple[list[Candidate], JsonObject]:
    if loop_redundancy_mode not in {"coarse_signature_v1", "certified_exact_adjacent_v2"}:
        raise ValueError("unsupported loop redundancy mode")
    if path_family_mode not in {"declared_or_task_v1", "capability_path_v2"}:
        raise ValueError("unsupported path family mode")
    unique: dict[str, JsonObject] = {}
    duplicate_rows = 0
    semantic_duplicates = 0
    semantic_to_trace: dict[str, str] = {}
    stripped_weight_fields: Counter[str] = Counter()
    for source in rows:
        _validate_candidate(source)
        trace = _trace_identity(source)
        if trace in unique:
            duplicate_rows += 1
            continue
        semantic = _semantic_identity(source)
        if semantic in semantic_to_trace:
            semantic_duplicates += 1
            continue
        copied = dict(source)
        for key in list(copied):
            if "loss_weight" in key or key == "sft_target_token_mass":
                stripped_weight_fields[key] += 1
                copied.pop(key, None)
        copied["split"] = "train"
        unique[trace] = copied
        semantic_to_trace[semantic] = trace

    raw: list[dict[str, Any]] = []
    missing_gap = 0
    for index, trace in enumerate(sorted(unique)):
        row = unique[trace]
        tokens = int(supervised_tokens.get(trace, 0))
        if tokens <= 0:
            raise ValueError(f"candidate {trace} lacks exact supervised token count")
        signatures = _decision_signatures(row)
        repeated = len(signatures) - len(set(signatures))
        gap_value = row.get("multi_scale_policy_gap")
        if gap_value is None:
            missing_gap += 1
            gap = 0.0
        else:
            gap = float(gap_value)
            if not math.isfinite(gap) or gap < 0:
                raise ValueError("multi_scale_policy_gap must be finite and non-negative")
        atoms = capability_atoms(row)
        recovery = max(
            float("recovery:after_error" in atoms),
            _nested_feature(row, "authentic_recovery"),
        )
        transition_atoms = sum(atom.startswith("transition:") for atom in atoms)
        raw.append(
            {
                "index": index,
                "row": row,
                "trace_hash": trace,
                "task": _task_identity(row),
                "teacher": _teacher_family(row),
                "teacher_provenance_explicit": bool(
                    row.get("teacher_family")
                    or (row.get("metadata") or {}).get("teacher_family")
                    or (row.get("metadata") or {}).get("teacher_model")
                ),
                "family": (
                    _capability_path_family_identity(row)
                    if path_family_mode == "capability_path_v2"
                    else _declared_family_identity(row)
                ),
                "tokens": tokens,
                "atoms": atoms,
                "recovery": recovery,
                "branch": max(
                    float(transition_atoms),
                    _nested_feature(row, "branch_coverage"),
                ),
                "cost_efficiency": max(
                    len(signatures) / tokens,
                    _nested_feature(row, "cost_efficiency"),
                ),
                "loop_redundancy": (
                    max(repeated / max(1, len(signatures)), _nested_feature(row, "loop_redundancy"))
                    if loop_redundancy_mode == "coarse_signature_v1"
                    else _certified_exact_adjacent_loop_redundancy(row)
                ),
                "unsupported_reference_risk": max(
                    0.0, _nested_feature(row, "unsupported_reference_risk")
                ),
                "policy_gap": gap,
            }
        )
    if require_multiscale_policy_gap and missing_gap:
        raise ValueError(
            f"{missing_gap} candidates lack predeclared multi_scale_policy_gap"
        )

    normalized_keys = ("recovery", "branch", "cost_efficiency", "policy_gap")
    ranges: dict[str, tuple[float, float]] = {}
    for key in normalized_keys:
        values = [float(item[key]) for item in raw]
        ranges[key] = (min(values, default=0.0), max(values, default=0.0))

    def normalized(key: str, value: float) -> float:
        low, high = ranges[key]
        if high <= low:
            return 0.0 if high == 0 else 1.0
        return (value - low) / (high - low)

    candidates = [
        Candidate(
            index=item["index"],
            row=item["row"],
            trace_hash=item["trace_hash"],
            task=item["task"],
            teacher=item["teacher"],
            teacher_provenance_explicit=item["teacher_provenance_explicit"],
            family=item["family"],
            tokens=item["tokens"],
            atoms=item["atoms"],
            recovery=normalized("recovery", item["recovery"]),
            branch=normalized("branch", item["branch"]),
            cost_efficiency=normalized("cost_efficiency", item["cost_efficiency"]),
            loop_redundancy=item["loop_redundancy"],
            unsupported_reference_risk=item["unsupported_reference_risk"],
            policy_gap=normalized("policy_gap", item["policy_gap"]),
        )
        for item in raw
    ]
    return candidates, {
        "candidate_rows": len(candidates),
        "duplicate_trace_rows_removed": duplicate_rows,
        "duplicate_semantic_rows_removed": semantic_duplicates,
        "missing_multiscale_policy_gap": missing_gap,
        "stripped_weight_fields": dict(sorted(stripped_weight_fields.items())),
        "normalization_ranges": {key: list(value) for key, value in ranges.items()},
        "loop_redundancy_mode": loop_redundancy_mode,
        "path_family_mode": path_family_mode,
        "unique_path_families": len({candidate.family for candidate in candidates}),
    }


def select_global_candidates(
    candidates: Sequence[Candidate],
    *,
    token_budget: int,
    max_per_task: int = 8,
    max_per_task_teacher: int = 4,
    policy_gap_enabled: bool = False,
    coverage_closure: bool = False,
    minimum_per_task: int = 0,
    fill_remaining: bool = True,
    teacher_cap_requires_explicit_provenance: bool = False,
    max_selected_trajectories: int | None = None,
    max_per_capability_path: int | None = None,
) -> tuple[list[JsonObject], JsonObject, list[JsonObject]]:
    if token_budget <= 0 or max_per_task <= 0 or max_per_task_teacher <= 0:
        raise ValueError("budgets and caps must be positive")
    if minimum_per_task < 0 or minimum_per_task > max_per_task:
        raise ValueError("minimum_per_task must be between zero and max_per_task")
    if max_selected_trajectories is not None and max_selected_trajectories <= 0:
        raise ValueError("maximum selected trajectories must be positive")
    if max_per_capability_path is not None and max_per_capability_path <= 0:
        raise ValueError("maximum per capability path must be positive")
    atom_universe = set().union(*(candidate.atoms for candidate in candidates))
    selected_atoms: set[str] = set()
    selected_families: set[str] = set()
    task_counts: Counter[str] = Counter()
    teacher_counts: Counter[tuple[str, str]] = Counter()
    path_counts: Counter[str] = Counter()
    selected: list[JsonObject] = []
    ledger: list[JsonObject] = []
    rejected: dict[int, str] = {}
    selected_indices: set[int] = set()
    used_tokens = 0

    def eligible(
        candidate: Candidate, *, ignore_path_cap: bool = False
    ) -> str | None:
        if max_selected_trajectories is not None and len(selected) >= max_selected_trajectories:
            return "trajectory_cap"
        if task_counts[candidate.task] >= max_per_task:
            return "task_cap"
        teacher_cap_applies = (
            not teacher_cap_requires_explicit_provenance
            or candidate.teacher_provenance_explicit
        )
        if teacher_cap_applies and teacher_counts[(candidate.task, candidate.teacher)] >= max_per_task_teacher:
            return "task_teacher_cap"
        if (
            not ignore_path_cap
            and
            max_per_capability_path is not None
            and path_counts[candidate.family] >= max_per_capability_path
        ):
            return "capability_path_cap"
        if used_tokens + candidate.tokens > token_budget:
            return "token_budget"
        return None

    def admit(candidate: Candidate, value: float, components: JsonObject, phase: str) -> None:
        nonlocal used_tokens
        rank = len(selected) + 1
        copied = dict(candidate.row)
        copied["global_selection"] = {
            "format": FORMAT,
            "rank": rank,
            "score": value,
            "phase": phase,
            "components": components,
            "supervised_tokens": candidate.tokens,
        }
        selected.append(copied)
        ledger.append(
            {
                "trace_hash": candidate.trace_hash,
                "selected": True,
                "rank": rank,
                "score": value,
                "phase": phase,
                "components": components,
                "tokens": candidate.tokens,
                "task_hash": _digest(candidate.task),
                "teacher_hash": candidate.teacher,
            }
        )
        selected_indices.add(candidate.index)
        used_tokens += candidate.tokens
        selected_atoms.update(candidate.atoms)
        selected_families.add(candidate.family)
        task_counts[candidate.task] += 1
        if not teacher_cap_requires_explicit_provenance or candidate.teacher_provenance_explicit:
            teacher_counts[(candidate.task, candidate.teacher)] += 1
        path_counts[candidate.family] += 1

    def score(candidate: Candidate) -> tuple[float, JsonObject]:
        new_atoms = candidate.atoms - selected_atoms
        marginal = len(new_atoms) / max(1, len(candidate.atoms))
        path_diversity = float(candidate.family not in selected_families)
        value = (
            4.0 * marginal
            + (4.0 * candidate.policy_gap if policy_gap_enabled else 0.0)
            + 2.0 * candidate.recovery
            + 1.5 * candidate.branch
            + 1.0 * path_diversity
            + 1.0 * candidate.cost_efficiency
            - 2.0 * candidate.loop_redundancy
            - 2.0 * candidate.unsupported_reference_risk
        )
        return value, {
            "marginal_certified_coverage": marginal,
            "new_atoms": len(new_atoms),
            "policy_gap": candidate.policy_gap if policy_gap_enabled else 0.0,
            "authentic_recovery": candidate.recovery,
            "branch_coverage": candidate.branch,
            "path_diversity": path_diversity,
            "cost_efficiency": candidate.cost_efficiency,
            "loop_redundancy": candidate.loop_redundancy,
            "unsupported_reference_risk": candidate.unsupported_reference_risk,
        }

    task_closure_rows = 0
    task_closure_tokens = 0
    task_universe = {candidate.task for candidate in candidates}
    if minimum_per_task:
        # Every executable training task competes under the same rule. This
        # prevents a large source from consuming the budget while a smaller
        # environment loses all representative behavior, without consulting
        # benchmark identity or assigning source-specific quotas.
        candidates_by_task: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in candidates:
            candidates_by_task[candidate.task].append(candidate)
        for _ in range(minimum_per_task):
            choices = []
            for task in sorted(task_universe):
                if task_counts[task] >= minimum_per_task:
                    continue
                for candidate in candidates_by_task[task]:
                    if (
                        candidate.index in selected_indices
                        or eligible(candidate, ignore_path_cap=True) is not None
                    ):
                        continue
                    reliability_value = (
                        (candidate.policy_gap if policy_gap_enabled else 0.0)
                        + candidate.recovery
                        + candidate.cost_efficiency
                        - candidate.loop_redundancy
                        - candidate.unsupported_reference_risk
                    )
                    choices.append((reliability_value,-candidate.tokens,candidate.trace_hash,candidate))
            if not choices:
                break
            while choices:
                _,_,_,candidate=max(choices,key=lambda item:item[:3])
                choices=[item for item in choices if item[3].task!=candidate.task]
                if eligible(candidate, ignore_path_cap=True) is not None:
                    continue
                components={"task_coverage":1.0,"policy_gap":candidate.policy_gap,
                    "authentic_recovery":candidate.recovery,"cost_efficiency":candidate.cost_efficiency,
                    "loop_redundancy":candidate.loop_redundancy,
                    "unsupported_reference_risk":candidate.unsupported_reference_risk,
                    "marginal_certified_coverage":len(candidate.atoms-selected_atoms)/max(1,len(candidate.atoms)),
                    "new_atoms":len(candidate.atoms-selected_atoms),"branch_coverage":candidate.branch,
                    "path_diversity":float(candidate.family not in selected_families)}
                admit(candidate,reliability_value,components,"task_representative_closure")
                task_closure_rows+=1;task_closure_tokens+=candidate.tokens

    closure_rows = 0
    closure_tokens = 0
    closure_atoms = 0
    if coverage_closure:
        remaining = {candidate.index: candidate for candidate in candidates}
        while remaining:
            choices = []
            for candidate in remaining.values():
                if eligible(candidate) is not None:
                    continue
                new_atoms = candidate.atoms - selected_atoms
                if not new_atoms:
                    continue
                choices.append(
                    (
                        len(new_atoms) / candidate.tokens,
                        len(new_atoms),
                        -candidate.tokens,
                        candidate.trace_hash,
                        candidate,
                    )
                )
            if not choices:
                break
            _, count, _, _, candidate = max(choices, key=lambda item: item[:4])
            new_atoms = candidate.atoms - selected_atoms
            components = {
                "marginal_certified_coverage": len(new_atoms) / max(1, len(candidate.atoms)),
                "new_atoms": len(new_atoms),
                "coverage_atoms_per_token": len(new_atoms) / candidate.tokens,
                "policy_gap": 0.0,
                "authentic_recovery": candidate.recovery,
                "branch_coverage": candidate.branch,
                "path_diversity": float(candidate.family not in selected_families),
                "cost_efficiency": candidate.cost_efficiency,
                "loop_redundancy": candidate.loop_redundancy,
                "unsupported_reference_risk": candidate.unsupported_reference_risk,
            }
            admit(candidate, float(count), components, "capability_coverage_closure")
            closure_rows += 1
            closure_tokens += candidate.tokens
            remaining.pop(candidate.index)
        closure_atoms = len(selected_atoms)

    heap: list[tuple[float, str, int]] = []
    by_index = {candidate.index: candidate for candidate in candidates}
    if fill_remaining:
        for candidate in candidates:
            if candidate.index in selected_indices:
                continue
            value, _ = score(candidate)
            heapq.heappush(heap, (-value, candidate.trace_hash, candidate.index))

    while heap:
        negative_bound, trace_hash, index = heapq.heappop(heap)
        if index in rejected:
            continue
        candidate = by_index[index]
        reason = eligible(candidate)
        if reason is not None:
            rejected[index] = reason
            continue
        value, components = score(candidate)
        next_bound = -heap[0][0] if heap else float("-inf")
        if value + 1e-12 < next_bound:
            heapq.heappush(heap, (-value, trace_hash, index))
            continue
        if value <= 0:
            rejected[index] = "non_positive_marginal_value"
            continue
        admit(candidate, value, components, "learning_value")

    selected_hashes = {str(row["trace_hash"]) for row in selected}
    for candidate in candidates:
        if candidate.trace_hash in selected_hashes:
            continue
        ledger.append(
            {
                "trace_hash": candidate.trace_hash,
                "selected": False,
                "reason": rejected.get(candidate.index, "not_reached_before_budget"),
                "tokens": candidate.tokens,
                "task_hash": _digest(candidate.task),
                "teacher_hash": candidate.teacher,
            }
        )
    audit = {
        "format": FORMAT,
        "candidate_rows": len(candidates),
        "selected_rows": len(selected),
        "token_budget": token_budget,
        "selected_supervised_tokens": used_tokens,
        "unused_token_budget": token_budget - used_tokens,
        "max_per_task": max_per_task,
        "max_per_task_teacher": max_per_task_teacher,
        "policy_gap_mode": "required_multiscale" if policy_gap_enabled else "disabled",
        "capability_atoms_total": len(atom_universe),
        "capability_atoms_selected": len(selected_atoms),
        "capability_coverage": len(selected_atoms) / max(1, len(atom_universe)),
        "selected_trace_hashes_sha256": _digest(sorted(selected_hashes)),
        "rejection_counts": dict(sorted(Counter(rejected.values()).items())),
        "benchmark_or_category_feature_used": False,
        "domain_specific_loss_weights": False,
        "coverage_closure": coverage_closure,
        "coverage_closure_selected_rows": closure_rows,
        "coverage_closure_selected_tokens": closure_tokens,
        "coverage_closure_atoms": closure_atoms,
        "minimum_per_task": minimum_per_task,
        "task_universe": len(task_universe),
        "task_closure_selected_rows": task_closure_rows,
        "task_closure_selected_tokens": task_closure_tokens,
        "tasks_represented": len(task_counts),
        "fill_remaining": fill_remaining,
        "teacher_cap_requires_explicit_provenance": teacher_cap_requires_explicit_provenance,
        "candidates_with_explicit_teacher_provenance": sum(
            candidate.teacher_provenance_explicit for candidate in candidates
        ),
        "max_selected_trajectories": max_selected_trajectories,
        "max_per_capability_path": max_per_capability_path,
        "capability_path_cap_applies_after_task_closure": True,
        "selected_capability_paths": len(path_counts),
    }
    return selected, audit, ledger


__all__ = [
    "Candidate",
    "FORMAT",
    "capability_atoms",
    "prepare_candidates",
    "select_global_candidates",
]

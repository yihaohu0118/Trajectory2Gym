"""Compile verified trajectories into native full-trace SFT examples.

The compiler intentionally has a small contract:

* one source trajectory becomes at most one SFT example;
* tool schemas and tool calls stay structured;
* supervision is explicit: every assistant turn (the baseline), only
  evidence-backed tool decisions plus the terminal response, or the agentic
  combination of verified tool decisions and all conversational decisions;
* train/validation assignment is task-disjoint;
* missing schemas and unknown tool calls fail closed.

Benchmark repositories remain responsible for collecting trajectories and
official evaluation.  This module only consumes their exported records.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from trajectory2gym.training.learning_value import (
    LEARNING_VALUE_PROFILES,
    LearningValueFeatures,
    evaluate_learning_value,
    learning_value_contract,
)
from trajectory2gym.training.policy_view import evidence_message_history


JsonObject = dict[str, Any]


class UnknownToolCallError(ValueError):
    """A trajectory invoked a tool absent from its declared native schema."""


class NoTrainableAssistantDecisionError(ValueError):
    """A trajectory contains no assistant decision eligible for policy loss."""


@dataclass(frozen=True)
class SFTBuildConfig:
    """Generic selection policy for a full-trajectory cold start."""

    seed: int = 0
    validation_fraction: float = 0.05
    max_traces_per_task: int = 2
    success_only: bool = True
    assume_success_when_missing: bool = False
    enable_thinking: bool = False
    supervision_mode: str = "verified_agentic"
    selection_mode: str = "learning_value"
    include_successful_policy_anchors: bool = False
    policy_frontier_only: bool = False
    policy_frontier_loss_only: bool = False
    require_supported_references: bool = False
    source_reservation_contains: str | None = None
    source_reservation_per_eligible_task: int = 0
    require_source_reservation: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.max_traces_per_task <= 0:
            raise ValueError("max_traces_per_task must be positive")
        if self.supervision_mode not in {
            "all_assistant",
            "verified_actions",
            "verified_agentic",
        }:
            raise ValueError(
                "supervision_mode must be all_assistant, verified_actions, "
                "or verified_agentic"
            )
        if self.selection_mode not in {
            "path_diverse",
            "learning_value",
            "candidate_pool",
        }:
            raise ValueError(
                "selection_mode must be path_diverse, learning_value, or candidate_pool"
            )
        if self.selection_mode in {"learning_value", "candidate_pool"} and not self.success_only:
            raise ValueError(
                "learning_value/candidate-pool compilation uses failed trajectories "
                "as evidence but never as SFT targets; success_only must remain enabled"
            )
        if self.policy_frontier_only and self.selection_mode != "learning_value":
            raise ValueError(
                "policy_frontier_only requires learning_value selection"
            )
        if self.policy_frontier_loss_only and not self.policy_frontier_only:
            raise ValueError(
                "policy_frontier_loss_only requires policy_frontier_only"
            )
        reservation_enabled = self.source_reservation_contains is not None
        if reservation_enabled != (self.source_reservation_per_eligible_task > 0):
            raise ValueError(
                "source reservation requires both a non-empty source match and "
                "a positive per-eligible-task reservation"
            )
        if (
            self.source_reservation_contains is not None
            and not self.source_reservation_contains.strip()
        ):
            raise ValueError("source reservation match must not be blank")
        if self.source_reservation_per_eligible_task < 0:
            raise ValueError("source reservation per task must be non-negative")
        if self.source_reservation_per_eligible_task > self.max_traces_per_task:
            raise ValueError("source reservation cannot exceed the per-task cap")
        if reservation_enabled and self.selection_mode != "learning_value":
            raise ValueError("source reservation requires learning_value selection")
        if self.require_source_reservation and not reservation_enabled:
            raise ValueError(
                "require_source_reservation requires an enabled source reservation"
            )


def iter_records(path: Path) -> Iterator[JsonObject]:
    """Read either a JSON array/object or JSONL without format guessing."""

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for record in payload:
                if not isinstance(record, dict):
                    raise ValueError(f"{path}: JSON array contains a non-object")
                yield record
            return
        if isinstance(payload, dict):
            yield payload
            return
        raise ValueError(f"{path}: expected a JSON object or array")

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield record


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_digest(values: Iterable[str]) -> str:
    return hashlib.sha256(
        _canonical_json(sorted(set(values))).encode("utf-8")
    ).hexdigest()


def _tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name") or "").strip()


def build_schema_bank(paths: Sequence[Path]) -> dict[str, list[JsonObject]]:
    """Merge declared schemas by domain and reject conflicting definitions."""

    definitions: dict[str, dict[str, JsonObject]] = defaultdict(dict)
    for path in paths:
        for record in iter_records(path):
            domain = str(record.get("domain") or "").strip()
            tools = record.get("tools")
            if not domain or not isinstance(tools, list):
                continue
            for tool in tools:
                if not isinstance(tool, dict):
                    raise ValueError(f"{path}: tool schema must be an object")
                name = _tool_name(tool)
                if not name:
                    raise ValueError(f"{path}: tool schema has no function name")
                previous = definitions[domain].get(name)
                if previous is not None and _canonical_json(previous) != _canonical_json(tool):
                    raise ValueError(
                        f"conflicting schema for domain={domain!r} tool={name!r}"
                    )
                definitions[domain][name] = copy.deepcopy(tool)
    return {
        domain: [tools[name] for name in sorted(tools)]
        for domain, tools in sorted(definitions.items())
    }


def _normalise_tool_call(call: Mapping[str, Any]) -> JsonObject:
    function = call.get("function")
    if not isinstance(function, Mapping):
        raise ValueError("assistant tool_call has no function object")
    name = str(function.get("name") or "").strip()
    if not name:
        raise ValueError("assistant tool_call has no function name")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool {name!r} has non-JSON arguments") from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"tool {name!r} arguments must decode to an object")
    result: JsonObject = {
        "type": "function",
        "function": {"name": name, "arguments": copy.deepcopy(arguments)},
    }
    if call.get("id") is not None:
        result["id"] = str(call["id"])
    return result


def _normalise_message(message: Mapping[str, Any]) -> JsonObject:
    role = str(message.get("role") or "").strip()
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"unsupported message role {role!r}")
    result: JsonObject = {"role": role, "content": message.get("content") or ""}
    calls = message.get("tool_calls")
    if calls:
        if role not in {"assistant", "user"} or not isinstance(calls, list):
            raise ValueError("tool_calls must be a list on an actor message")
        result["tool_calls"] = [_normalise_tool_call(call) for call in calls]
    for key in ("name", "tool_call_id"):
        if message.get(key) is not None:
            result[key] = str(message[key])
    if role in {"user", "tool"} and str(message.get("requestor") or "") == "user":
        result["requestor"] = "user"
    return result


def _step_actions(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    actions = step.get("expected_actions")
    if actions is None:
        actions = step.get("gold_actions")
    if actions is None:
        return []
    if not isinstance(actions, list):
        raise ValueError("trajectory step actions must be a list")
    return actions


def _messages_from_steps(record: Mapping[str, Any]) -> list[JsonObject]:
    initial = record.get("initial_messages")
    if initial is None:
        initial = record.get("prompt")
    if not isinstance(initial, list):
        raise ValueError("trajectory has neither messages nor initial_messages/prompt")
    messages = [_normalise_message(message) for message in initial]
    steps = record.get("steps")
    if not isinstance(steps, list):
        raise ValueError("trajectory steps must be a list")
    trajectory_id = str(record.get("id") or record.get("task_id") or "trajectory")
    for step_index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError("trajectory step must be an object")
        actions = _step_actions(step)
        if actions:
            calls: list[JsonObject] = []
            for action_index, action in enumerate(actions):
                if not isinstance(action, Mapping):
                    raise ValueError("trajectory action must be an object")
                calls.append(
                    {
                        "id": f"call_{trajectory_id}_{step_index}_{action_index}",
                        "type": "function",
                        "function": {
                            "name": str(action.get("name") or ""),
                            "arguments": copy.deepcopy(action.get("arguments") or {}),
                        },
                    }
                )
            messages.append(_normalise_message({"role": "assistant", "content": "", "tool_calls": calls}))
        else:
            metadata = step.get("metadata")
            expected_text = (
                str(metadata.get("expected_assistant_message") or "").strip()
                if isinstance(metadata, Mapping)
                else ""
            )
            if expected_text:
                messages.append({"role": "assistant", "content": expected_text})
        observations = step.get("observations") or []
        if not isinstance(observations, list):
            raise ValueError("trajectory observations must be a list")
        messages.extend(_normalise_message(observation) for observation in observations)
    return messages


def _record_messages(record: Mapping[str, Any]) -> list[JsonObject]:
    messages = record.get("traj")
    if messages is None:
        messages = record.get("messages")
    if messages is not None:
        if not isinstance(messages, list):
            raise ValueError("trajectory messages must be a list")
        return [_normalise_message(message) for message in messages]
    return _messages_from_steps(record)


def _record_success(record: Mapping[str, Any], *, assume_missing: bool) -> bool:
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        role = str(
            record.get("trajectory_role")
            or metadata.get("trajectory_role")
            or ""
        ).strip()
        if (
            role == "frontier_anchor"
            and metadata.get("context_frontier_anchor") is True
            and metadata.get("verified_bounded_frontier_anchor") is True
            and metadata.get("training_eligible") is True
            and metadata.get("decision_window_truncated") is not True
        ):
            # This is an explicit, finite local training target emitted by the
            # generic state-graph compiler.  It does not claim that the source
            # benchmark session was successful.
            return True
    for key in ("success", "source_session_success"):
        value = record.get(key)
        if isinstance(value, bool):
            return value
    reward = record.get("reward")
    if isinstance(reward, (int, float)) and not isinstance(reward, bool):
        return float(reward) >= 1.0
    if isinstance(metadata, Mapping):
        for key in ("source_session_success", "terminal_success_eligible"):
            value = metadata.get(key)
            if isinstance(value, bool):
                return value
        reward = metadata.get("official_reward")
        if isinstance(reward, (int, float)) and not isinstance(reward, bool):
            return float(reward) >= 1.0
    return assume_missing


def _trajectory_role(record: Mapping[str, Any]) -> str:
    """Return an explicitly declared data role without benchmark inference."""

    metadata = record.get("metadata")
    candidates = [record.get("trajectory_role")]
    if isinstance(metadata, Mapping):
        candidates.append(metadata.get("trajectory_role"))
    for candidate in candidates:
        role = str(candidate or "").strip()
        if role:
            return role
    return "demonstration"


def _context_assistant_turns(record: Mapping[str, Any]) -> int:
    """Count immutable historical policy turns for a rooted local target."""

    metadata = record.get("metadata")
    explicit = record.get("context_assistant_turns")
    if explicit is None and isinstance(metadata, Mapping):
        explicit = metadata.get("context_assistant_turns")
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 0:
            raise ValueError("context_assistant_turns must be a non-negative integer")
        return explicit
    if _trajectory_role(record) != "frontier_anchor":
        return 0
    initial = record.get("initial_messages")
    if initial is None:
        initial = record.get("prompt")
    if not isinstance(initial, list):
        raise ValueError("frontier anchor requires an explicit initial context")
    return sum(
        str(message.get("role") or "") == "assistant"
        for message in initial
        if isinstance(message, Mapping)
    )


def _state_certificates_by_assistant_turn(
    record: Mapping[str, Any],
    *,
    domain: str,
    tools: Sequence[JsonObject],
) -> dict[str, JsonObject]:
    """Preserve explicit CCSG entry states through the SFT serialization."""

    metadata = record.get("metadata")
    explicit = record.get("state_certificates_by_assistant_turn")
    if explicit is None and isinstance(metadata, Mapping):
        explicit = metadata.get("state_certificates_by_assistant_turn")
    if isinstance(explicit, Mapping):
        result: dict[str, JsonObject] = {}
        for raw_turn, raw_certificate in explicit.items():
            if not isinstance(raw_certificate, Mapping):
                continue
            pre_state_id = str(
                raw_certificate.get("pre_state_id") or ""
            ).strip()
            if pre_state_id:
                result[str(int(raw_turn))] = copy.deepcopy(
                    dict(raw_certificate)
                )
        return result

    steps = record.get("steps")
    if not isinstance(steps, list):
        return {}
    source = str(record.get("source") or "trajectory")
    trajectory_group_id = ""
    if isinstance(metadata, Mapping):
        trajectory_group_id = str(
            metadata.get("trajectory_group_id")
            or metadata.get("source_task_id")
            or ""
        )
    tool_signature = _stable_hash(list(tools))
    result: dict[str, JsonObject] = {}
    assistant_turn = 0
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        step_metadata = step.get("metadata")
        step_metadata = (
            step_metadata if isinstance(step_metadata, Mapping) else {}
        )
        actions = _step_actions(step)
        expected_text = str(
            step_metadata.get("expected_assistant_message") or ""
        ).strip()
        if not actions and not expected_text:
            continue
        pre_state_id = str(step_metadata.get("pre_state_id") or "").strip()
        if pre_state_id:
            result[str(assistant_turn)] = {
                "source": source,
                "domain": domain,
                "trajectory_group_id": trajectory_group_id,
                "tool_signature": tool_signature,
                "pre_state_id": pre_state_id,
            }
        assistant_turn += 1
    return result


def _source_task_id(record: Mapping[str, Any], domain: str) -> str:
    metadata = record.get("metadata")
    candidates = [record.get("source_task_id")]
    if isinstance(metadata, Mapping):
        candidates.append(metadata.get("source_task_id"))
    candidates.extend((record.get("task_id"), record.get("id")))
    for candidate in candidates:
        if candidate is not None and str(candidate).strip():
            value = str(candidate).strip()
            return value if value.startswith(f"{domain}:") else f"{domain}:{value}"
    raise ValueError("trajectory has no task identity")


def _action_signature(messages: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            names.append(str(call["function"]["name"]))
    return tuple(names)


def _observation_is_explicit_error(message: Mapping[str, Any]) -> bool:
    """Recognize only explicit tool failures; ambiguous text stays evidence."""

    content = message.get("content")
    if isinstance(content, Mapping):
        if content.get("error") not in (None, "", False):
            return True
        if content.get("success") is False:
            return True
        return str(content.get("status") or "").strip().lower() in {
            "error",
            "failed",
            "failure",
            "invalid",
        }
    text = str(content or "").strip()
    if text.lower().startswith(("error:", "error ", "failed:", "failure:")):
        return True
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return False
        if isinstance(parsed, Mapping):
            return _observation_is_explicit_error({"content": parsed})
    return False


def _canonical_call_key(call: Mapping[str, Any]) -> str:
    function = call.get("function")
    return _canonical_json(function if isinstance(function, Mapping) else call)


def _supervised_assistant_turns(
    messages: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    context_assistant_turns: int = 0,
    allow_context_certified_repeats: bool = False,
    decision_certified_turns: frozenset[int] = frozenset(),
    state_certificates: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[int]:
    assistant_message_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if mode == "all_assistant":
        return list(range(len(assistant_message_indices)))

    selected: list[int] = []
    seen_calls: set[str] = set()
    seen_call_state_ids: dict[str, set[str]] = defaultdict(set)
    seen_call_observation_keys: dict[str, set[str]] = defaultdict(set)
    state_certificates = state_certificates or {}
    for assistant_turn, message_index in enumerate(assistant_message_indices):
        message = messages[message_index]
        calls = message.get("tool_calls") or []
        if calls:
            is_decision_certified_target = (
                assistant_turn in decision_certified_turns
            )
            is_context_certified_target = bool(
                allow_context_certified_repeats
                and assistant_turn >= context_assistant_turns
            )
            observations: list[Mapping[str, Any]] = []
            cursor = message_index + 1
            while cursor < len(messages) and messages[cursor].get("role") == "tool":
                observations.append(messages[cursor])
                cursor += 1
            call_keys = [_canonical_call_key(call) for call in calls]
            # A bounded frontier anchor already carries a graph certificate
            # over its exact action and authentic observation.  Some native
            # trajectory formats serialize that observation as a user/external
            # event rather than a ``tool`` role, so the certificate is the
            # stronger portable source of truth for this target only.
            # A decision checker can certify the policy target without
            # executing the tool.  This permission is deliberately narrower
            # than an environment transition certificate: it enables SFT loss
            # on this assistant action, but ``_certified_decisions`` still
            # requires authentic tool observations before adding an
            # action-observation edge to the evidence graph.
            has_complete_evidence = bool(
                len(observations) >= len(calls)
                or is_context_certified_target
                or is_decision_certified_target
            )
            has_error = any(_observation_is_explicit_error(item) for item in observations)
            repeated_call_keys = [key for key in call_keys if key in seen_calls]
            repeats_exact_action = bool(repeated_call_keys)
            state_certificate = state_certificates.get(str(assistant_turn)) or {}
            current_state_id = str(state_certificate.get("pre_state_id") or "")
            observation_keys = [
                _canonical_observation(observation)
                for observation in observations[: len(calls)]
            ]
            repeated_positions = [
                index for index, key in enumerate(call_keys) if key in seen_calls
            ]
            repeat_has_distinct_authentic_observation = bool(
                repeated_call_keys
                and all(
                    index < len(observation_keys)
                    and observation_keys[index]
                    not in seen_call_observation_keys.get(call_keys[index], set())
                    for index in repeated_positions
                )
            )
            seen_calls.update(call_keys)
            if current_state_id:
                for key in call_keys:
                    seen_call_state_ids[key].add(current_state_id)
            for index, key in enumerate(call_keys):
                if index < len(observation_keys):
                    seen_call_observation_keys[key].add(observation_keys[index])
            repeat_is_certified_target = bool(
                repeats_exact_action
                and (
                    is_context_certified_target
                    or is_decision_certified_target
                    or repeat_has_distinct_authentic_observation
                )
            )
            if has_complete_evidence and not has_error and (
                not repeats_exact_action or repeat_is_certified_target
            ):
                selected.append(assistant_turn)
        elif mode == "verified_agentic":
            # Asking for missing information, requesting confirmation, giving
            # a user-side instruction, declining an unsafe request, and
            # handing off are policy decisions too.  They do not fabricate an
            # environment observation and are essential in datasets where the
            # user or an external actor performs part of the workflow.  Keep
            # them while retaining the stricter verifier above for tool calls.
            selected.append(assistant_turn)

    # The last assistant text is the demonstrated stopping/hand-off decision.
    # Preserve it even when intermediate conversational prose is left to the
    # base model. If the trajectory ends with a tool call, that call was
    # already handled above.
    if assistant_message_indices:
        final_turn = len(assistant_message_indices) - 1
        final_message = messages[assistant_message_indices[-1]]
        if not final_message.get("tool_calls") and final_turn not in selected:
            selected.append(final_turn)
    if not selected:
        raise NoTrainableAssistantDecisionError(
            "verified_actions found no supervised assistant turn"
        )
    return sorted(selected)


def _decision_certified_assistant_turns(
    record: Mapping[str, Any],
    *,
    success: bool,
    assistant_turns: int,
) -> tuple[frozenset[int], str]:
    """Read an explicit action-only supervision certificate.

    Some tool-use datasets have an authoritative action checker but no tool
    executor.  Their verified calls are legitimate policy targets even though
    they are not environment transitions.  The two permissions must remain
    separate: the caller may use these indices for SFT masking only, while the
    evidence graph continues to require recorded observations.
    """

    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    raw_turns = record.get("verified_assistant_turn_indices")
    if raw_turns is None:
        raw_turns = metadata.get("verified_assistant_turn_indices")
    if raw_turns is None:
        return frozenset(), ""
    certificate = str(
        record.get("supervision_certificate")
        or metadata.get("supervision_certificate")
        or ""
    ).strip()
    if not certificate:
        raise ValueError(
            "verified_assistant_turn_indices require a non-empty "
            "supervision_certificate"
        )
    if not success:
        raise ValueError(
            "decision-certified assistant turns require a successful record"
        )
    if not isinstance(raw_turns, list):
        raise ValueError("verified_assistant_turn_indices must be a list")
    turns: set[int] = set()
    for value in raw_turns:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "verified_assistant_turn_indices must contain non-negative "
                "integers"
            )
        if value >= assistant_turns:
            raise ValueError(
                "verified_assistant_turn_indices contains an out-of-range turn"
            )
        turns.add(value)
    return frozenset(turns), certificate


def _validate_example(example: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Validate a compiled example and classify out-of-schema calls.

    A failed policy rollout is evidence about what the policy attempted, not
    an imitation target.  Such a rollout may therefore contain a hallucinated
    tool name.  Keeping that decision is necessary to construct a same-state
    policy gap, but it must not relax validation for demonstrations or
    successful trajectories.

    A successful real-environment session can also contain a rejected action
    and later recover.  When the rejected assistant turn has a directly
    observed error *and* is excluded from the SFT loss mask, retaining the
    rest of that session is safe and useful.  This is a turn-level recovery
    rule, not a schema relaxation: the rejected call remains context only and
    can never become a target.
    """

    tools = example.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"{example.get('task_id')}: native tool schemas are required")
    schema_names = {_tool_name(tool) for tool in tools}
    if "" in schema_names:
        raise ValueError(f"{example.get('task_id')}: malformed tool schema")
    # Every policy probe is evidence-only, including a probe that eventually
    # recovers and completes the task.  Its intermediate invalid decisions
    # remain negative evidence and must never be converted into supervision.
    evidence_role = str(example.get("trajectory_role") or "demonstration")
    allow_invalid_evidence_calls = evidence_role in {
        "policy_probe",
        "evidence_only",
    }
    invalid_evidence_calls: list[str] = []
    rejected_recovery_calls: list[str] = []
    supervised_turns = {
        int(index)
        for index in example.get("supervised_assistant_turn_indices") or []
    }
    success = bool(example.get("success"))
    assistant_turns = 0
    messages = list(example.get("messages") or [])
    for message_index, message in enumerate(messages):
        assistant_turn = assistant_turns
        is_assistant = message.get("role") == "assistant"
        if is_assistant:
            assistant_turns += 1
        observations: list[Mapping[str, Any]] = []
        if is_assistant and message.get("tool_calls"):
            cursor = message_index + 1
            while cursor < len(messages) and messages[cursor].get("role") == "tool":
                observations.append(messages[cursor])
                cursor += 1
        observations_by_id = {
            str(observation.get("tool_call_id") or observation.get("id") or ""):
            observation
            for observation in observations
        }
        calls = (message.get("tool_calls") or []) if is_assistant else []
        for call_index, call in enumerate(calls):
            name = str(call["function"]["name"])
            if name not in schema_names:
                if allow_invalid_evidence_calls:
                    invalid_evidence_calls.append(name)
                    continue
                call_id = str(call.get("id") or "")
                observation = observations_by_id.get(call_id)
                if observation is None and call_index < len(observations):
                    # Some trajectory formats omit call IDs but preserve the
                    # native order of batched tool observations.
                    observation = observations[call_index]
                rejected_and_masked = (
                    success
                    and assistant_turn not in supervised_turns
                    and observation is not None
                    and _observation_is_explicit_error(observation)
                )
                if rejected_and_masked:
                    rejected_recovery_calls.append(name)
                    continue
                raise UnknownToolCallError(
                    f"{example.get('task_id')}: tool call {name!r} is absent from schema"
                )
    if assistant_turns == 0:
        raise NoTrainableAssistantDecisionError(
            f"{example.get('task_id')}: trajectory has no assistant turn"
        )
    return invalid_evidence_calls, rejected_recovery_calls


def compile_record(
    record: Mapping[str, Any],
    *,
    domain_override: str | None,
    schemas: Mapping[str, Sequence[JsonObject]],
    config: SFTBuildConfig,
) -> JsonObject | None:
    domain = str(domain_override or record.get("domain") or "").strip()
    if not domain:
        raise ValueError("trajectory domain is required")
    success = _record_success(record, assume_missing=config.assume_success_when_missing)
    if config.success_only and not success:
        return None
    messages = _record_messages(record)
    tools = record.get("tools")
    if not isinstance(tools, list) or not tools:
        tools = list(schemas.get(domain) or ())
    source_task_id = _source_task_id(record, domain)
    trace_hash = _stable_hash({"messages": messages, "tools": tools})
    assistant_indices = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    decision_certified_turns, decision_certificate = (
        _decision_certified_assistant_turns(
            record,
            success=success,
            assistant_turns=len(assistant_indices),
        )
    )
    # Indices consumed by the collator count assistant turns, not messages.
    context_assistant_turns = _context_assistant_turns(record)
    metadata = record.get("metadata")
    context_certified_anchor = bool(
        _trajectory_role(record) == "frontier_anchor"
        and isinstance(metadata, Mapping)
        and metadata.get("verified_bounded_frontier_anchor") is True
    )
    state_certificates = _state_certificates_by_assistant_turn(
        record,
        domain=domain,
        tools=tools,
    )
    supervised_turns = _supervised_assistant_turns(
        messages,
        mode=config.supervision_mode,
        context_assistant_turns=context_assistant_turns,
        allow_context_certified_repeats=context_certified_anchor,
        decision_certified_turns=decision_certified_turns,
        state_certificates=state_certificates,
    )
    if context_assistant_turns >= len(assistant_indices):
        raise NoTrainableAssistantDecisionError(
            "rooted trajectory context leaves no trainable assistant decision"
        )
    if context_assistant_turns:
        supervised_turns = [
            index for index in supervised_turns if index >= context_assistant_turns
        ]
        if not supervised_turns:
            raise NoTrainableAssistantDecisionError(
                "rooted trajectory has no verified target after context"
            )
    example: JsonObject = {
        "format": "trajectory2gym_sft_v0",
        "split": "train",
        "source": str(record.get("source") or "trajectory"),
        "trajectory_role": _trajectory_role(record),
        "task_id": f"{source_task_id}:{trace_hash[:12]}",
        "source_task_id": source_task_id,
        "domain": domain,
        "success": success,
        "enable_thinking": config.enable_thinking,
        "context_assistant_turns": context_assistant_turns,
        "supervised_assistant_turn_indices": supervised_turns,
        "assistant_turns": len(assistant_indices),
        "supervised_assistant_turns": len(supervised_turns),
        "tools": copy.deepcopy(tools),
        "messages": messages,
        "trace_hash": trace_hash,
    }
    verified_task_anchor = bool(
        record.get("verified_task_anchor")
        or (
            isinstance(metadata, Mapping)
            and metadata.get("verified_task_anchor") is True
        )
    )
    if verified_task_anchor:
        if example["trajectory_role"] != "retention_anchor":
            raise ValueError(
                "verified_task_anchor requires trajectory_role=retention_anchor"
            )
        # Keep the task-level evidence contract visible after compilation so
        # capped selectors can reserve one certified anchor before ranking the
        # remaining candidates by learning value.
        example["verified_task_anchor"] = True
    if decision_certified_turns:
        example["verified_assistant_turn_indices"] = sorted(
            decision_certified_turns
        )
        example["supervision_certificate"] = decision_certificate
    if state_certificates:
        example["state_certificates_by_assistant_turn"] = state_certificates
    probe_target = record.get("policy_probe_target_assistant_turn")
    if probe_target is None and isinstance(metadata, Mapping):
        probe_target = metadata.get("policy_probe_target_assistant_turn")
    if probe_target is not None:
        example["policy_probe_target_assistant_turn"] = int(probe_target)
    probe_outcome = record.get("policy_probe_outcome")
    if probe_outcome is None and isinstance(metadata, Mapping):
        probe_outcome = metadata.get("policy_probe_outcome")
        if probe_outcome is None and metadata.get("observation_generated") is False:
            probe_outcome = "unknown"
    if example["trajectory_role"] == "policy_probe":
        if probe_outcome is None:
            probe_outcome = "success" if success else "failure"
        probe_outcome = str(probe_outcome)
        if probe_outcome not in {"success", "failure", "unknown"}:
            raise ValueError(f"unsupported policy_probe_outcome: {probe_outcome!r}")
        example["policy_probe_outcome"] = probe_outcome
    invalid_evidence_calls, rejected_recovery_calls = _validate_example(example)
    if invalid_evidence_calls:
        example["invalid_evidence_tool_calls"] = invalid_evidence_calls
    if rejected_recovery_calls:
        example["rejected_recovery_tool_calls"] = rejected_recovery_calls
    return example


def _select_per_task(
    examples: Iterable[JsonObject], *, max_traces_per_task: int
) -> list[JsonObject]:
    grouped: dict[str, list[JsonObject]] = defaultdict(list)
    for example in examples:
        grouped[str(example["source_task_id"])].append(example)
    selected: list[JsonObject] = []
    for task_id in sorted(grouped):
        # Prefer a short representative from each distinct tool path, then
        # fill the remaining per-task budget with other unique full traces.
        # Repeated attempts can share the final tool sequence while differing
        # in clarification, confirmation, recovery, and stopping decisions.
        # Exact duplicate traces were already removed above.
        by_signature: dict[tuple[str, ...], list[JsonObject]] = defaultdict(list)
        for example in grouped[task_id]:
            signature = _action_signature(example["messages"])
            by_signature[signature].append(example)
        for candidates in by_signature.values():
            candidates.sort(
                key=lambda row: (
                    len(_canonical_json(row["messages"])),
                    row["trace_hash"],
                )
            )
        representatives = sorted(
            (candidates[0] for candidates in by_signature.values()),
            key=lambda row: (
                len(_canonical_json(row["messages"])),
                row["trace_hash"],
            ),
        )
        chosen = representatives[:max_traces_per_task]
        if len(chosen) < max_traces_per_task:
            chosen_hashes = {str(row["trace_hash"]) for row in chosen}
            remaining = sorted(
                (
                    row
                    for candidates in by_signature.values()
                    for row in candidates
                    if str(row["trace_hash"]) not in chosen_hashes
                ),
                key=lambda row: (
                    len(_canonical_json(row["messages"])),
                    row["trace_hash"],
                ),
            )
            chosen.extend(remaining[: max_traces_per_task - len(chosen)])
        selected.extend(chosen)
    return selected


# These weights define a small, benchmark-neutral data objective. Semantic
# similarity is intentionally absent: coarse interaction diversity may rank
# authentic targets later, but it never becomes environment evidence.
LEARNING_VALUE_WEIGHTS: dict[str, float] = dict(
    LEARNING_VALUE_PROFILES["sft"]
)


@dataclass(frozen=True)
class _CertifiedDecision:
    """One exact state-action-observation edge from a full trajectory."""

    assistant_turn: int
    state_key: str
    action_key: str
    observation_key: str
    edge_key: str
    explicit_error: bool
    structured_references: int
    unsupported_references: int
    state_certified: bool


@dataclass(frozen=True)
class _PolicyDecision:
    """A frozen-policy action at an exact observable trajectory state."""

    state_key: str
    action_key: str
    explicit_error: bool
    state_certified: bool


@dataclass(frozen=True)
class _CertifiedTextDecision:
    """A supervised text/no-tool policy decision at an exact state.

    This is action-only supervision.  It deliberately carries no observation
    or evidence edge and therefore cannot expand the Partial Gym executable
    domain.
    """

    assistant_turn: int
    state_key: str
    action_key: str


def _structured_reference_values(value: Any) -> list[str]:
    """Extract identifier-like values whose provenance can be audited.

    Mixed alpha-numeric values, dates, e-mails, order numbers, and similar
    references occur across tool-use benchmarks. Free-form prose, enums, and
    small numeric quantities are deliberately left unjudged.
    """

    references: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            # Search text and handoff summaries are policy decisions, not
            # identifiers copied into an environment mutation. Digits in
            # free-form prose (tool suffixes, amounts, dates) must not be
            # mistaken for unsupported state references. Structured sibling
            # fields such as account_id and agent_tool_name remain audited.
            if str(key).casefold() in {"query", "search_query", "summary"} and isinstance(
                child, str
            ):
                continue
            references.extend(_structured_reference_values(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            references.extend(_structured_reference_values(child))
    elif isinstance(value, str):
        text = value.strip()
        # Some generic wrapper tools carry their inner argument object as a
        # JSON string (for example ``{"arguments": "{...}"}``).  Audit the
        # identifier values inside that object instead of treating the entire
        # serialized blob as one impossible-to-match reference.
        if text.startswith(("{", "[")):
            try:
                nested = json.loads(text)
            except json.JSONDecodeError:
                nested = None
            if isinstance(nested, (Mapping, list, tuple)):
                return _structured_reference_values(nested)
        has_digit = any(character.isdigit() for character in text)
        has_alpha = any(character.isalpha() for character in text)
        if len(text) >= 4 and (
            "@" in text or has_digit and (has_alpha or len(text) >= 6)
        ):
            references.append(text)
    return references


def _decision_reference_support(
    calls: Sequence[Mapping[str, Any]],
    observable_context: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    context = _canonical_json(observable_context).casefold()
    references: list[str] = []
    for call in calls:
        function = call.get("function")
        if isinstance(function, Mapping):
            references.extend(
                _structured_reference_values(function.get("arguments") or {})
            )
    unsupported = sum(
        not _reference_is_context_supported(reference, context)
        for reference in references
    )
    return len(references), unsupported


def _reference_is_context_supported(reference: str, context: str) -> bool:
    """Accept exact provenance or a narrowly certified calendar derivation.

    Stateful banking policies sometimes require an expiration exactly one
    calendar year after a timestamp returned by the live backend.  Treating
    that date as an unsupported identifier discards an otherwise fully
    grounded transition.  We admit only the deterministic same-month/day,
    year-plus-one transformation when the visible request explicitly says
    ``one year``; arbitrary date arithmetic remains unsupported.
    """

    folded = reference.casefold()
    if folded in context:
        return True
    match = re.fullmatch(r"(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(\d{4})", reference)
    if match is None or not re.search(r"\bone[- ]year\b", context):
        return False
    month, day, year = map(int, match.groups())
    observed: set[tuple[int, int, int]] = set()
    observed.update(
        (int(raw_month), int(raw_day), int(raw_year))
        for raw_month, raw_day, raw_year in re.findall(
            r"\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(\d{4})\b",
            context,
        )
    )
    observed.update(
        (int(raw_month), int(raw_day), int(raw_year))
        for raw_year, raw_month, raw_day in re.findall(
            r"\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])\b",
            context,
        )
    )
    return any(
        observed_month == month
        and observed_day == day
        and observed_year + 1 == year
        for observed_month, observed_day, observed_year in observed
    )


def _canonical_observation(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        content = content.strip()
        if content.startswith(("{", "[")):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                pass
    return _canonical_json(
        {
            "name": str(message.get("name") or ""),
            "content": content,
        }
    )


def _certified_decisions(example: Mapping[str, Any]) -> list[_CertifiedDecision]:
    """Extract exact context-certified decisions without benchmark rules.

    The certificate contains the task identity, every system/user observation,
    and every prior exact action-observation edge.  Assistant prose is omitted:
    it is an agent decision rather than external state; any subsequent user
    response is included and therefore still separates observable contexts.
    Incomplete tool calls are not evidence and contribute no certified edge.
    """

    messages = example.get("messages") or []
    state_certificates = (
        example.get("state_certificates_by_assistant_turn") or {}
    )
    observable_context: list[JsonObject] = []
    decisions: list[_CertifiedDecision] = []
    assistant_turn = -1
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        calls = message.get("tool_calls") or []
        if role == "assistant":
            assistant_turn += 1
        if role == "assistant" and calls:
            observations: list[Mapping[str, Any]] = []
            cursor = index + 1
            while cursor < len(messages) and messages[cursor].get("role") == "tool":
                observations.append(messages[cursor])
                cursor += 1
            action_key = _canonical_json(
                [
                    call.get("function")
                    if isinstance(call.get("function"), Mapping)
                    else call
                    for call in calls
                ]
            )
            if len(observations) >= len(calls):
                state_certificate = state_certificates.get(
                    str(assistant_turn)
                )
                state_certified = isinstance(state_certificate, Mapping)
                state_key = _stable_hash(
                    {
                        "task": str(example["source_task_id"]),
                        "observable_context": observable_context,
                        "ccsg_state": (
                            state_certificate if state_certified else None
                        ),
                    }
                )
                observation_key = _canonical_json(
                    [_canonical_observation(item) for item in observations[: len(calls)]]
                )
                structured_references, unsupported_references = (
                    _decision_reference_support(calls, observable_context)
                )
                edge_key = _stable_hash(
                    {
                        "state": state_key,
                        "action": action_key,
                        "observation": observation_key,
                    }
                )
                decisions.append(
                    _CertifiedDecision(
                        assistant_turn=assistant_turn,
                        state_key=state_key,
                        action_key=action_key,
                        observation_key=observation_key,
                        edge_key=edge_key,
                        explicit_error=any(
                            _observation_is_explicit_error(item)
                            for item in observations[: len(calls)]
                        ),
                        structured_references=structured_references,
                        unsupported_references=unsupported_references,
                        state_certified=state_certified,
                    )
                )
                observable_context.append(
                    {"action": action_key, "observation": observation_key}
                )
            else:
                observable_context.append(
                    {"action": action_key, "observation": "__missing__"}
                )
            index = cursor
            continue
        if role in {"system", "user"}:
            observable_context.append(
                {"role": str(role), "content": message.get("content")}
            )
        elif role == "tool":
            # Preserve orphan external observations in the next certificate.
            observable_context.append(
                {"role": "tool", "observation": _canonical_observation(message)}
            )
        index += 1
    return decisions


def _policy_probe_decisions(example: Mapping[str, Any]) -> list[_PolicyDecision]:
    """Extract policy actions without claiming an environment transition.

    Full environment probes may contain observations.  Offline prefix probes
    intentionally stop immediately after the frozen policy's next action and
    therefore contain none.  Both are valid evidence about policy behavior at
    the exact preceding context, but only the former can contribute a
    certified action-observation edge through ``_certified_decisions``.
    """

    messages = example.get("messages") or []
    state_certificates = (
        example.get("state_certificates_by_assistant_turn") or {}
    )
    target_turn = example.get("policy_probe_target_assistant_turn")
    if target_turn is not None:
        target_turn = int(target_turn)
    observable_context: list[JsonObject] = []
    decisions: list[_PolicyDecision] = []
    assistant_turn = -1
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        calls = message.get("tool_calls") or []
        if role == "assistant":
            assistant_turn += 1
            observations: list[Mapping[str, Any]] = []
            cursor = index + 1
            while cursor < len(messages) and messages[cursor].get("role") == "tool":
                observations.append(messages[cursor])
                cursor += 1
            if calls:
                action_key = _canonical_json(
                    [
                        call.get("function")
                        if isinstance(call.get("function"), Mapping)
                        else call
                        for call in calls
                    ]
                )
            else:
                # The exact prose is deliberately excluded.  For tool-use
                # selection, responding with text instead of taking the
                # demonstrated tool action is the policy decision of interest.
                action_key = _canonical_json({"type": "assistant_text"})
            if target_turn is None or assistant_turn == target_turn:
                decisions.append(
                    _PolicyDecision(
                        state_key=_stable_hash(
                            {
                                "task": str(example["source_task_id"]),
                                "observable_context": observable_context,
                                "ccsg_state": state_certificates.get(
                                    str(assistant_turn)
                                ),
                            }
                        ),
                        action_key=action_key,
                        explicit_error=bool(calls)
                        and any(
                            _observation_is_explicit_error(item)
                            for item in observations[: len(calls)]
                        ),
                        state_certified=isinstance(
                            state_certificates.get(str(assistant_turn)),
                            Mapping,
                        ),
                    )
                )
            if calls:
                if len(observations) >= len(calls):
                    observation_key = _canonical_json(
                        [
                            _canonical_observation(item)
                            for item in observations[: len(calls)]
                        ]
                    )
                    observable_context.append(
                        {"action": action_key, "observation": observation_key}
                    )
                else:
                    observable_context.append(
                        {"action": action_key, "observation": "__missing__"}
                    )
            index = cursor
            continue
        if role in {"system", "user"}:
            observable_context.append(
                {"role": str(role), "content": message.get("content")}
            )
        elif role == "tool":
            observable_context.append(
                {"role": "tool", "observation": _canonical_observation(message)}
            )
        index += 1
    return decisions


def _certified_text_policy_decisions(
    example: Mapping[str, Any],
) -> list[_CertifiedTextDecision]:
    """Extract supervised no-tool choices without inventing transitions."""

    messages = example.get("messages") or []
    supervised_turns = {
        int(index)
        for index in example.get("supervised_assistant_turn_indices") or []
    }
    state_certificates = (
        example.get("state_certificates_by_assistant_turn") or {}
    )
    observable_context: list[JsonObject] = []
    decisions: list[_CertifiedTextDecision] = []
    assistant_turn = -1
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        calls = message.get("tool_calls") or []
        if role == "assistant":
            assistant_turn += 1
            observations: list[Mapping[str, Any]] = []
            cursor = index + 1
            while cursor < len(messages) and messages[cursor].get("role") == "tool":
                observations.append(messages[cursor])
                cursor += 1
            if not calls:
                if (
                    assistant_turn in supervised_turns
                    and str(message.get("content") or "").strip()
                ):
                    decisions.append(
                        _CertifiedTextDecision(
                            assistant_turn=assistant_turn,
                            state_key=_stable_hash(
                                {
                                    "task": str(example["source_task_id"]),
                                    "observable_context": observable_context,
                                    "ccsg_state": state_certificates.get(
                                        str(assistant_turn)
                                    ),
                                }
                            ),
                            action_key=_canonical_json(
                                {"type": "assistant_text"}
                            ),
                        )
                    )
            else:
                action_key = _canonical_json(
                    [
                        call.get("function")
                        if isinstance(call.get("function"), Mapping)
                        else call
                        for call in calls
                    ]
                )
                if len(observations) >= len(calls):
                    observation_key = _canonical_json(
                        [
                            _canonical_observation(item)
                            for item in observations[: len(calls)]
                        ]
                    )
                    observable_context.append(
                        {"action": action_key, "observation": observation_key}
                    )
                else:
                    observable_context.append(
                        {"action": action_key, "observation": "__missing__"}
                    )
            index = cursor
            continue
        if role in {"system", "user"}:
            observable_context.append(
                {"role": str(role), "content": message.get("content")}
            )
        elif role == "tool":
            observable_context.append(
                {"role": "tool", "observation": _canonical_observation(message)}
            )
        index += 1
    return decisions


def _external_event_instruction_turns(
    example: Mapping[str, Any],
) -> list[int]:
    """Return text turns followed by a fully observed external-actor event.

    Some agent trajectories include actions that are performed by the user or
    another external actor rather than by the assistant tool runtime.  The
    policy-visible view hides that actor's tool-call payload, but the compiler
    retains it as provenance.  A text turn is an evidence-backed instruction
    target only when every structured external call before the next assistant
    turn has a matching external observation.  No text semantics or
    benchmark-specific tool names are used.
    """

    messages = evidence_message_history(example)
    selected: list[int] = []
    assistant_turn = -1
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        assistant_turn += 1
        if message.get("tool_calls") or not str(message.get("content") or "").strip():
            continue
        cursor = index + 1
        certified = False
        while cursor < len(messages) and messages[cursor].get("role") != "assistant":
            actor_message = messages[cursor]
            calls = actor_message.get("tool_calls") or []
            if (
                actor_message.get("role") == "user"
                and str(actor_message.get("requestor") or "") == "user"
                and calls
            ):
                call_ids = {
                    str(call.get("id"))
                    for call in calls
                    if call.get("id") is not None
                }
                observed_ids: set[str] = set()
                observation_cursor = cursor + 1
                while (
                    observation_cursor < len(messages)
                    and messages[observation_cursor].get("role") != "assistant"
                ):
                    observation = messages[observation_cursor]
                    if (
                        observation.get("role") == "tool"
                        and str(observation.get("requestor") or "") == "user"
                        and observation.get("tool_call_id") is not None
                    ):
                        observed_ids.add(str(observation["tool_call_id"]))
                    observation_cursor += 1
                if call_ids and call_ids <= observed_ids:
                    certified = True
                    break
            cursor += 1
        if certified:
            selected.append(assistant_turn)
    return selected


def _authentic_recovery_count(decisions: Sequence[_CertifiedDecision]) -> int:
    return sum(
        1
        for index, decision in enumerate(decisions)
        if decision.explicit_error
        and any(not later.explicit_error for later in decisions[index + 1 :])
    )


def _select_learning_value_per_task(
    eligible_examples: Iterable[JsonObject],
    *,
    evidence_examples: Iterable[JsonObject],
    max_traces_per_task: int,
    require_supported_references: bool = False,
    source_reservation_contains: str | None = None,
    source_reservation_per_eligible_task: int = 0,
    require_source_reservation: bool = False,
) -> tuple[list[JsonObject], JsonObject]:
    """Greedily retain successful full traces with certified learning value.

    Failed sessions shape the exact state graph and expose counterfactual
    branches, but only successful sessions are eligible imitation targets.
    Every score component is emitted on the selected row and in the manifest.
    """

    eligible_by_task: dict[str, list[JsonObject]] = defaultdict(list)
    evidence_by_task: dict[str, list[JsonObject]] = defaultdict(list)
    for example in eligible_examples:
        if not bool(example.get("success")):
            raise AssertionError("failed trajectory entered the SFT candidate pool")
        eligible_by_task[str(example["source_task_id"])].append(example)
    for example in evidence_examples:
        evidence_by_task[str(example["source_task_id"])].append(example)

    selected: list[JsonObject] = []
    evidence_edge_keys: set[str] = set()
    evidence_ccsg_edge_keys: set[str] = set()
    candidate_edge_keys: set[str] = set()
    candidate_ccsg_edge_keys: set[str] = set()
    selected_edge_keys: set[str] = set()
    selected_ccsg_edge_keys: set[str] = set()
    branch_states: set[str] = set()
    ccsg_branch_states: set[str] = set()
    contrastive_states: set[str] = set()
    ccsg_contrastive_states: set[str] = set()
    policy_probe_decision_states: set[str] = set()
    policy_probe_failure_states: set[str] = set()
    policy_probe_unsuccessful_session_states: set[str] = set()
    policy_probe_unknown_outcome_states: set[str] = set()
    policy_gap_states: set[str] = set()
    verified_policy_failure_gap_states: set[str] = set()
    invalid_evidence_tool_calls = 0
    invalid_policy_probe_tool_calls = 0
    rejected_recovery_tool_calls = 0
    recovered_success_trajectories = 0
    candidate_policy_gap_trajectories = 0
    candidate_policy_gap_decisions = 0
    evidence_successes = 0
    evidence_failures = 0
    unsupported_reference_successes_evidence_only = 0
    reservation_eligible_rows_by_task: Counter[str] = Counter()
    reservation_selected_rows_by_task: Counter[str] = Counter()
    reservation_eligible_trace_hashes: set[str] = set()
    reservation_selected_trace_hashes: set[str] = set()
    reservation_eligible_sources: set[str] = set()
    reservation_selected_sources: set[str] = set()

    for task_id in sorted(eligible_by_task):
        candidates = eligible_by_task[task_id]
        evidence = evidence_by_task.get(task_id, candidates)
        decisions_by_hash = {
            str(row["trace_hash"]): _certified_decisions(row)
            for row in evidence
        }
        text_decisions_by_hash = {
            str(row["trace_hash"]): _certified_text_policy_decisions(row)
            for row in evidence
        }
        state_actions: dict[str, set[str]] = defaultdict(set)
        state_success_actions: dict[str, set[str]] = defaultdict(set)
        state_failure_actions: dict[str, set[str]] = defaultdict(set)
        state_policy_actions: dict[str, set[str]] = defaultdict(set)
        state_policy_failure_actions: dict[str, set[str]] = defaultdict(set)
        state_is_ccsg_certified: dict[str, bool] = defaultdict(bool)
        for row in evidence:
            success = bool(row.get("success"))
            role = str(row.get("trajectory_role") or "demonstration")
            invalid_calls = len(row.get("invalid_evidence_tool_calls") or [])
            invalid_evidence_tool_calls += invalid_calls
            recovered_calls = len(row.get("rejected_recovery_tool_calls") or [])
            rejected_recovery_tool_calls += recovered_calls
            recovered_success_trajectories += int(bool(recovered_calls))
            if role == "policy_probe":
                invalid_policy_probe_tool_calls += invalid_calls
            evidence_successes += int(success)
            evidence_failures += int(not success)
            for decision in decisions_by_hash[str(row["trace_hash"])]:
                evidence_edge_keys.add(decision.edge_key)
                if decision.state_certified:
                    evidence_ccsg_edge_keys.add(decision.edge_key)
                    state_is_ccsg_certified[decision.state_key] = True
                state_actions[decision.state_key].add(decision.action_key)
                # Credit is transition-local when the observation itself is
                # an explicit error.  A session that later recovers must not
                # relabel its rejected action as a successful action.
                target = (
                    state_failure_actions
                    if decision.explicit_error or not success
                    else state_success_actions
                )
                target[decision.state_key].add(decision.action_key)
            if success:
                for decision in text_decisions_by_hash[str(row["trace_hash"])]:
                    state_success_actions[decision.state_key].add(
                        decision.action_key
                    )
            if role == "policy_probe":
                probe_outcome = str(row.get("policy_probe_outcome") or "failure")
                for decision in _policy_probe_decisions(row):
                    state_policy_actions[decision.state_key].add(
                        decision.action_key
                    )
                    policy_probe_decision_states.add(decision.state_key)
                    if probe_outcome == "unknown":
                        policy_probe_unknown_outcome_states.add(decision.state_key)
                    if probe_outcome == "failure":
                        policy_probe_unsuccessful_session_states.add(
                            decision.state_key
                        )
                    if decision.explicit_error:
                        state_policy_failure_actions[decision.state_key].add(
                            decision.action_key
                        )
                        policy_probe_failure_states.add(decision.state_key)

        task_branch_states = {
            state for state, actions in state_actions.items() if len(actions) > 1
        }
        task_contrastive_states = {
            state
            for state in task_branch_states
            if state_success_actions[state] and state_failure_actions[state]
        }
        task_ccsg_branch_states = {
            state
            for state in task_branch_states
            if state_is_ccsg_certified[state]
        }
        task_ccsg_contrastive_states = {
            state
            for state in task_contrastive_states
            if state_is_ccsg_certified[state]
        }
        task_policy_gap_states = {
            state
            for state, policy_actions in state_policy_actions.items()
            if state_success_actions[state]
            and not bool(policy_actions & state_success_actions[state])
        }
        task_verified_policy_failure_gap_states = {
            state
            for state, failed_actions in state_policy_failure_actions.items()
            if state_success_actions[state]
            and not bool(failed_actions & state_success_actions[state])
        }
        branch_states.update(task_branch_states)
        ccsg_branch_states.update(task_ccsg_branch_states)
        contrastive_states.update(task_contrastive_states)
        ccsg_contrastive_states.update(task_ccsg_contrastive_states)
        policy_gap_states.update(task_policy_gap_states)
        verified_policy_failure_gap_states.update(
            task_verified_policy_failure_gap_states
        )

        candidate_profiles: dict[str, JsonObject] = {}
        task_max_edges = 1
        task_min_steps: int | None = None
        task_min_chars: int | None = None
        for row in candidates:
            trace_hash = str(row["trace_hash"])
            decisions = decisions_by_hash.get(trace_hash)
            if decisions is None:
                decisions = _certified_decisions(row)
            text_decisions = text_decisions_by_hash.get(trace_hash)
            if text_decisions is None:
                text_decisions = _certified_text_policy_decisions(row)
            unique_edges = {decision.edge_key for decision in decisions}
            ccsg_edges = {
                decision.edge_key
                for decision in decisions
                if decision.state_certified
            }
            candidate_edge_keys.update(unique_edges)
            candidate_ccsg_edge_keys.update(ccsg_edges)
            repeated_edges = max(0, len(decisions) - len(unique_edges))
            action_observation_pairs = {
                (decision.action_key, decision.observation_key)
                for decision in decisions
            }
            repeated_action_observations = max(
                0, len(decisions) - len(action_observation_pairs)
            )
            steps = max(1, len(decisions))
            chars = max(1, len(_canonical_json(row["messages"])))
            branch_count = sum(
                decision.state_key in task_branch_states for decision in decisions
            )
            contrastive_count = sum(
                decision.state_key in task_contrastive_states
                and decision.action_key in state_success_actions[decision.state_key]
                and bool(
                    state_failure_actions[decision.state_key]
                    - {decision.action_key}
                )
                for decision in decisions
            )
            # Merely visiting a state at which the frozen policy has a gap is
            # not a repair.  A recovered successful session may first take an
            # explicitly rejected action at that state and only succeed later.
            # Count (and eventually supervise) only the demonstrated action
            # that itself has successful transition evidence at the exact
            # state.  This prevents session-level success from relabelling a
            # failed turn as a positive policy-gap target.
            policy_gap_turn_indices = sorted(
                {
                    decision.assistant_turn
                    for decision in decisions
                    if decision.state_key in task_policy_gap_states
                    and decision.action_key
                    in state_success_actions[decision.state_key]
                }
                | {
                    decision.assistant_turn
                    for decision in text_decisions
                    if decision.state_key in task_policy_gap_states
                    and decision.action_key
                    in state_success_actions[decision.state_key]
                }
            )
            policy_gap_count = len(policy_gap_turn_indices)
            policy_decision_count = len(
                {
                    decision.assistant_turn
                    for decision in (*decisions, *text_decisions)
                }
            )
            verified_policy_failure_gap_count = sum(
                decision.state_key in task_verified_policy_failure_gap_states
                for decision in decisions
            )
            external_event_instruction_turn_indices = (
                _external_event_instruction_turns(row)
            )
            structured_references = sum(
                decision.structured_references for decision in decisions
            )
            unsupported_references = sum(
                decision.unsupported_references for decision in decisions
            )
            candidate_profiles[trace_hash] = {
                "decisions": decisions,
                "edge_keys": unique_edges,
                "ccsg_edge_keys": ccsg_edges,
                "path_key": tuple(decision.action_key for decision in decisions),
                "steps": steps,
                "chars": chars,
                "repeated_edges": repeated_edges,
                "repeated_action_observations": repeated_action_observations,
                "branch_count": branch_count,
                "contrastive_count": contrastive_count,
                "policy_gap_count": policy_gap_count,
                "policy_decision_count": policy_decision_count,
                "policy_gap_turn_indices": policy_gap_turn_indices,
                "verified_policy_failure_gap_count": (
                    verified_policy_failure_gap_count
                ),
                "external_event_instruction_turn_indices": (
                    external_event_instruction_turn_indices
                ),
                "structured_references": structured_references,
                "unsupported_references": unsupported_references,
                "recovery_count": _authentic_recovery_count(decisions),
            }
            candidate_policy_gap_trajectories += int(policy_gap_count > 0)
            candidate_policy_gap_decisions += policy_gap_count
            task_max_edges = max(task_max_edges, len(unique_edges))
            task_min_steps = steps if task_min_steps is None else min(task_min_steps, steps)
            task_min_chars = chars if task_min_chars is None else min(task_min_chars, chars)

        covered_edges: set[str] = set()
        covered_paths: set[tuple[str, ...]] = set()
        remaining = [
            row
            for row in candidates
            if not require_supported_references
            or candidate_profiles[str(row["trace_hash"])][
                "unsupported_references"
            ]
            == 0
        ]
        unsupported_reference_successes_evidence_only += len(candidates) - len(
            remaining
        )
        if not remaining:
            continue

        def select_winner(
            rank_candidates: Sequence[JsonObject],
            *,
            rank: int,
            reserved_source: bool,
        ) -> JsonObject:
            nonlocal remaining
            scored: list[tuple[tuple[Any, ...], JsonObject, JsonObject]] = []
            for row in rank_candidates:
                profile = candidate_profiles[str(row["trace_hash"])]
                new_edges = profile["edge_keys"] - covered_edges
                marginal_coverage = len(new_edges) / task_max_edges
                decision_count = max(1, len(profile["decisions"]))
                contrastive_frontier = profile["contrastive_count"] / decision_count
                policy_gap = profile["policy_gap_count"] / max(
                    1, profile["policy_decision_count"]
                )
                branch_coverage = profile["branch_count"] / decision_count
                authentic_recovery = min(1.0, float(profile["recovery_count"]))
                path_diversity = float(profile["path_key"] not in covered_paths)
                cost_efficiency = 0.5 * (
                    float(task_min_steps or 1) / profile["steps"]
                    + float(task_min_chars or 1) / profile["chars"]
                )
                unsupported_reference_rate = profile["unsupported_references"] / max(
                    1, profile["structured_references"]
                )
                loop_redundancy = (
                    profile["repeated_action_observations"] / decision_count
                )
                learning_value = evaluate_learning_value(
                    "sft",
                    LearningValueFeatures(
                        evidence_certified=True,
                        success_eligible=True,
                        marginal_certified_coverage=marginal_coverage,
                        contrastive_frontier=contrastive_frontier,
                        policy_gap=policy_gap,
                        branch_coverage=branch_coverage,
                        authentic_recovery=authentic_recovery,
                        path_diversity=path_diversity,
                        cost_efficiency=cost_efficiency,
                        unsupported_reference_risk=(
                            unsupported_reference_rate
                        ),
                        loop_redundancy=loop_redundancy,
                    ),
                )
                score = learning_value.raw_score
                components: JsonObject = {
                    "mode": "learning_value",
                    "learning_value": learning_value.to_dict(),
                    "rank_within_task": rank,
                    "score": round(score, 8),
                    "marginal_certified_edges": len(new_edges),
                    "certified_edges": len(profile["edge_keys"]),
                    "ccsg_state_certified_edges": len(
                        profile["ccsg_edge_keys"]
                    ),
                    "contrastive_frontier_decisions": profile["contrastive_count"],
                    "policy_gap_decisions": profile["policy_gap_count"],
                    "policy_gap_assistant_turn_indices": profile[
                        "policy_gap_turn_indices"
                    ],
                    "verified_policy_failure_gap_decisions": profile[
                        "verified_policy_failure_gap_count"
                    ],
                    "external_event_instruction_decisions": len(
                        profile["external_event_instruction_turn_indices"]
                    ),
                    "external_event_instruction_turn_indices": profile[
                        "external_event_instruction_turn_indices"
                    ],
                    "branch_decisions": profile["branch_count"],
                    "authentic_recovery_events": profile["recovery_count"],
                    "path_is_novel": bool(path_diversity),
                    "cost_efficiency": round(cost_efficiency, 8),
                    "repeated_certified_edges": profile["repeated_edges"],
                    "repeated_action_observations": profile[
                        "repeated_action_observations"
                    ],
                    "structured_references": profile["structured_references"],
                    "unsupported_references": profile["unsupported_references"],
                    "unsupported_reference_rate": round(
                        unsupported_reference_rate, 8
                    ),
                }
                if reserved_source:
                    components["source_reservation"] = {
                        "reserved": True,
                        "source_contains": source_reservation_contains,
                        "per_eligible_task": (
                            source_reservation_per_eligible_task
                        ),
                    }
                # Sorting is deterministic and deliberately makes compactness
                # only a tie-break after evidence value.
                key = (
                    -score,
                    -len(new_edges),
                    -profile["contrastive_count"],
                    -profile["policy_gap_count"],
                    -profile["recovery_count"],
                    profile["unsupported_references"],
                    profile["repeated_action_observations"],
                    profile["chars"],
                    str(row["trace_hash"]),
                )
                scored.append((key, row, components))
            _, winner, components = min(scored, key=lambda item: item[0])
            copied = dict(winner)
            copied["curation"] = components
            selected.append(copied)
            profile = candidate_profiles[str(winner["trace_hash"])]
            covered_edges.update(profile["edge_keys"])
            selected_edge_keys.update(profile["edge_keys"])
            selected_ccsg_edge_keys.update(profile["ccsg_edge_keys"])
            covered_paths.add(profile["path_key"])
            remaining = [
                row for row in remaining if row["trace_hash"] != winner["trace_hash"]
            ]
            return winner

        reserved_candidates = [
            row
            for row in remaining
            if _matches_reserved_source(row, source_reservation_contains)
        ]
        if reserved_candidates:
            reservation_eligible_rows_by_task[task_id] = len(
                reserved_candidates
            )
            reservation_eligible_trace_hashes.update(
                str(row["trace_hash"]) for row in reserved_candidates
            )
            reservation_eligible_sources.update(
                str(row.get("source") or "") for row in reserved_candidates
            )
        reservation_limit = min(
            source_reservation_per_eligible_task,
            len(reserved_candidates),
            max_traces_per_task,
        )
        task_selected = 0
        for _ in range(reservation_limit):
            source_pool = [
                row
                for row in remaining
                if _matches_reserved_source(
                    row, source_reservation_contains
                )
            ]
            winner = select_winner(
                source_pool,
                rank=task_selected + 1,
                reserved_source=True,
            )
            task_selected += 1
            reservation_selected_rows_by_task[task_id] += 1
            reservation_selected_trace_hashes.add(str(winner["trace_hash"]))
            reservation_selected_sources.add(
                str(winner.get("source") or "")
            )
        if reserved_candidates and (
            reservation_selected_rows_by_task[task_id] != reservation_limit
        ):
            raise AssertionError(
                f"source reservation did not fill its task quota: {task_id}"
            )

        first_general_selection = True
        while task_selected < max_traces_per_task and remaining:
            verified_anchors = [
                row
                for row in remaining
                if row.get("verified_task_anchor") is True
            ]
            retention_anchors = [
                row
                for row in remaining
                if row.get("trajectory_role") == "retention_anchor"
            ]
            rank_candidates = (
                verified_anchors
                if first_general_selection and verified_anchors
                else retention_anchors
                if first_general_selection and retention_anchors
                else remaining
            )
            select_winner(
                rank_candidates,
                rank=task_selected + 1,
                reserved_source=False,
            )
            task_selected += 1
            first_general_selection = False

    audit: JsonObject = {
        "mode": "learning_value",
        "weights": dict(LEARNING_VALUE_WEIGHTS),
        "learning_value_contract": learning_value_contract(),
        "evidence_records": evidence_successes + evidence_failures,
        "evidence_success_records": evidence_successes,
        "evidence_failure_records": evidence_failures,
        "evidence_certified_edges": len(evidence_edge_keys),
        "evidence_ccsg_state_certified_edges": len(
            evidence_ccsg_edge_keys
        ),
        "candidate_success_traces": sum(len(rows) for rows in eligible_by_task.values()),
        "candidate_certified_edges": len(candidate_edge_keys),
        "candidate_ccsg_state_certified_edges": len(
            candidate_ccsg_edge_keys
        ),
        "selected_certified_edges": len(selected_edge_keys),
        "selected_ccsg_state_certified_edges": len(
            selected_ccsg_edge_keys
        ),
        "selected_candidate_edge_coverage": round(
            len(selected_edge_keys) / max(1, len(candidate_edge_keys)), 8
        ),
        "branch_states": len(branch_states),
        "ccsg_state_certified_branch_states": len(ccsg_branch_states),
        "contrastive_frontier_states": len(contrastive_states),
        "ccsg_state_certified_contrastive_frontier_states": len(
            ccsg_contrastive_states
        ),
        "policy_probe_decision_states": len(policy_probe_decision_states),
        "policy_probe_failure_states": len(policy_probe_failure_states),
        "policy_probe_unsuccessful_session_states": len(
            policy_probe_unsuccessful_session_states
        ),
        "policy_probe_unknown_outcome_states": len(
            policy_probe_unknown_outcome_states
        ),
        "policy_gap_states": len(policy_gap_states),
        "verified_policy_failure_gap_states": len(
            verified_policy_failure_gap_states
        ),
        "invalid_evidence_tool_calls": invalid_evidence_tool_calls,
        "invalid_policy_probe_tool_calls": invalid_policy_probe_tool_calls,
        "rejected_recovery_tool_calls": rejected_recovery_tool_calls,
        "recovered_success_trajectories": recovered_success_trajectories,
        "candidate_policy_gap_trajectories": candidate_policy_gap_trajectories,
        "candidate_policy_gap_decisions": candidate_policy_gap_decisions,
        "unsupported_reference_successes_evidence_only": (
            unsupported_reference_successes_evidence_only
        ),
        "selected_policy_gap_trajectories": sum(
            int(row["curation"]["policy_gap_decisions"] > 0) for row in selected
        ),
        "selected_external_event_instruction_trajectories": sum(
            int(row["curation"]["external_event_instruction_decisions"] > 0)
            for row in selected
        ),
        "selected_verified_policy_failure_gap_trajectories": sum(
            int(row["curation"]["verified_policy_failure_gap_decisions"] > 0)
            for row in selected
        ),
        "selected_unsupported_reference_trajectories": sum(
            int(row["curation"]["unsupported_references"] > 0) for row in selected
        ),
        "selected_stagnant_loop_trajectories": sum(
            int(row["curation"]["repeated_action_observations"] > 0)
            for row in selected
        ),
        "selected_recovery_trajectories": sum(
            int(row["curation"]["authentic_recovery_events"] > 0) for row in selected
        ),
        "failed_trajectories_selected": sum(
            int(not bool(row.get("success"))) for row in selected
        ),
    }
    if audit["failed_trajectories_selected"]:
        raise AssertionError("failed trajectory selected as an SFT target")
    reservation_enabled = source_reservation_contains is not None
    eligible_tasks = set(reservation_eligible_rows_by_task)
    reserved_tasks = set(reservation_selected_rows_by_task)
    selected_source_rows = [
        row
        for row in selected
        if _matches_reserved_source(row, source_reservation_contains)
    ]
    selected_source_rows_by_task = Counter(
        str(row["source_task_id"]) for row in selected_source_rows
    )
    selected_source_tasks = set(selected_source_rows_by_task)
    selected_source_trace_hashes = {
        str(row["trace_hash"]) for row in selected_source_rows
    }
    selected_source_values = {
        str(row.get("source") or "") for row in selected_source_rows
    }
    expected_selected_rows_by_task = {
        task_id: min(
            source_reservation_per_eligible_task,
            count,
            max_traces_per_task,
        )
        for task_id, count in reservation_eligible_rows_by_task.items()
    }
    reservation_failures: list[str] = []
    if reservation_enabled:
        if selected_source_tasks != eligible_tasks:
            reservation_failures.append(
                "eligible source-reservation tasks were not all selected"
            )
        if dict(reservation_selected_rows_by_task) != (
            expected_selected_rows_by_task
        ):
            reservation_failures.append(
                "source-reservation per-task row counts differ"
            )
        if require_source_reservation and not eligible_tasks:
            reservation_failures.append(
                "required source reservation has no clean supported success candidate"
            )
        if require_source_reservation and not reservation_selected_trace_hashes:
            reservation_failures.append(
                "required source reservation selected no SFT target"
            )
    source_reservation_audit = {
        "status": (
            "disabled"
            if not reservation_enabled
            else "pass"
            if not reservation_failures
            else "fail"
        ),
        "source_contains": source_reservation_contains,
        "per_eligible_task": source_reservation_per_eligible_task,
        "require_nonempty": require_source_reservation,
        "max_traces_per_task": max_traces_per_task,
        "cap_preserved": True,
        "eligible_rows": sum(reservation_eligible_rows_by_task.values()),
        "eligible_tasks": len(eligible_tasks),
        "reserved_rows": sum(reservation_selected_rows_by_task.values()),
        "reserved_tasks": len(reserved_tasks),
        "selected_rows": len(selected_source_rows),
        "selected_tasks": len(selected_source_tasks),
        "eligible_rows_by_task": dict(
            sorted(reservation_eligible_rows_by_task.items())
        ),
        "selected_rows_by_task": dict(
            sorted(selected_source_rows_by_task.items())
        ),
        "reserved_rows_by_task": dict(
            sorted(reservation_selected_rows_by_task.items())
        ),
        "eligible_tasks_sha256": _set_digest(eligible_tasks),
        "selected_tasks_sha256": _set_digest(selected_source_tasks),
        "eligible_trace_hashes_sha256": _set_digest(
            reservation_eligible_trace_hashes
        ),
        "selected_trace_hashes_sha256": _set_digest(
            selected_source_trace_hashes
        ),
        "eligible_sources_sha256": _set_digest(
            reservation_eligible_sources
        ),
        "selected_sources_sha256": _set_digest(
            selected_source_values
        ),
        "reserved_trace_hashes_sha256": _set_digest(
            reservation_selected_trace_hashes
        ),
        "reserved_sources_sha256": _set_digest(
            reservation_selected_sources
        ),
        "failures": reservation_failures,
    }
    audit["source_reservation"] = source_reservation_audit
    if reservation_failures:
        raise ValueError("; ".join(reservation_failures))
    return selected, audit


def _is_validation_task(task_id: str, *, seed: int, fraction: float) -> bool:
    if fraction <= 0:
        return False
    digest = hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return value < fraction


def _matches_reserved_source(
    example: Mapping[str, Any], source_contains: str | None
) -> bool:
    return bool(
        source_contains
        and source_contains in str(example.get("source") or "")
    )


def _deduplicate_compiled_examples(
    examples: Sequence[JsonObject],
    *,
    preserve_source_contains: str | None = None,
) -> tuple[dict[tuple[str, str, str], JsonObject], JsonObject]:
    """Deduplicate collector retries without discarding stronger evidence.

    The trace hash intentionally covers only messages and tools.  Therefore an
    official verified anchor and an unlabelled collector copy of the same
    conversation collide.  A last-write-wins dictionary can silently erase the
    anchor depending on input order.  Resolve collisions deterministically by
    retaining the strongest explicit supervision contract.
    """

    unique: dict[tuple[str, str, str], JsonObject] = {}
    collisions = 0
    verified_anchor_promotions = 0

    def preference(example: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            int(example.get("verified_task_anchor") is True),
            int(example.get("trajectory_role") == "retention_anchor"),
            len(example.get("verified_assistant_turn_indices") or []),
            int(example.get("supervised_assistant_turns") or 0),
            _canonical_json(example),
        )

    for example in examples:
        key = (
            str(example["source_task_id"]),
            str(example["trace_hash"]),
            (
                str(example.get("source") or "")
                if _matches_reserved_source(
                    example, preserve_source_contains
                )
                else ""
            ),
        )
        incumbent = unique.get(key)
        if incumbent is None:
            unique[key] = example
            continue
        collisions += 1
        if preference(example) > preference(incumbent):
            if (
                example.get("verified_task_anchor") is True
                and incumbent.get("verified_task_anchor") is not True
            ):
                verified_anchor_promotions += 1
            unique[key] = example

    return unique, {
        "duplicate_compiled_records": collisions,
        "verified_anchor_promotions_during_dedup": verified_anchor_promotions,
        "unique_verified_task_anchors": sum(
            int(row.get("verified_task_anchor") is True)
            for row in unique.values()
        ),
    }


def build_sft_dataset(
    inputs: Sequence[tuple[str | None, Path]],
    *,
    schema_sources: Sequence[Path],
    config: SFTBuildConfig,
) -> tuple[list[JsonObject], list[JsonObject], JsonObject]:
    """Compile, select, and split full trajectories."""

    schemas = build_schema_bank(schema_sources)
    compiled: list[JsonObject] = []
    evidence_compiled: list[JsonObject] = []
    skipped_unsuccessful = 0
    skipped_evidence_only = 0
    unsafe_demonstrations_evidence_only = 0
    successful_policy_anchors = 0
    rows_without_trainable_assistant_decision = 0
    input_records = 0
    for domain, path in inputs:
        for record in iter_records(path):
            input_records += 1
            success = _record_success(
                record, assume_missing=config.assume_success_when_missing
            )
            trajectory_role = _trajectory_role(record)
            if (
                trajectory_role == "policy_probe"
                and success
                and config.include_successful_policy_anchors
            ):
                # A frozen-policy session that actually succeeded in the real
                # environment is useful retention data.  Promote it under an
                # explicit role; failed probes remain evidence-only.  The
                # verified-agentic loss still masks rejected tool turns inside
                # a session that later recovered.
                record = copy.deepcopy(record)
                record["trajectory_role"] = "retention_anchor"
                metadata = dict(record.get("metadata") or {})
                metadata["trajectory_role"] = "retention_anchor"
                metadata["retention_anchor_source"] = "successful_policy_probe"
                record["metadata"] = metadata
                trajectory_role = "retention_anchor"
                successful_policy_anchors += 1
            evidence_only = trajectory_role in {"policy_probe", "evidence_only"}
            if evidence_only and config.selection_mode not in {
                "learning_value",
                "candidate_pool",
            }:
                # A policy rollout is an audit/evidence row, never an
                # imitation target. Baseline selectors that do not build an
                # evidence graph simply leave it out.
                skipped_evidence_only += 1
                continue
            if config.selection_mode in {"learning_value", "candidate_pool"} and (
                not success or evidence_only
            ):
                # Failed sessions and explicitly tagged policy probes build
                # the exact evidence graph. Their assistant tokens are never
                # training targets, including successful policy probes.
                evidence_record = record
                try:
                    evidence_example = compile_record(
                        evidence_record,
                        domain_override=domain,
                        schemas=schemas,
                        config=replace(
                            config,
                            success_only=False,
                            supervision_mode="all_assistant",
                            selection_mode="path_diverse",
                            policy_frontier_only=False,
                            policy_frontier_loss_only=False,
                            source_reservation_contains=None,
                            source_reservation_per_eligible_task=0,
                            require_source_reservation=False,
                        ),
                    )
                except NoTrainableAssistantDecisionError:
                    rows_without_trainable_assistant_decision += 1
                    if success:
                        skipped_evidence_only += 1
                    else:
                        skipped_unsuccessful += 1
                    continue
                except UnknownToolCallError:
                    if trajectory_role != "demonstration":
                        raise
                    # A collected demonstration containing an out-of-schema
                    # call is unsafe imitation data.  Preserve its real
                    # observations as graph evidence, but make the demotion
                    # explicit and irreversible for this build.
                    evidence_record = copy.deepcopy(record)
                    evidence_record["trajectory_role"] = "evidence_only"
                    metadata = dict(evidence_record.get("metadata") or {})
                    metadata["trajectory_role"] = "evidence_only"
                    metadata["evidence_only_reason"] = "unknown_tool_call"
                    evidence_record["metadata"] = metadata
                    try:
                        evidence_example = compile_record(
                            evidence_record,
                            domain_override=domain,
                            schemas=schemas,
                            config=replace(
                                config,
                                success_only=False,
                                supervision_mode="all_assistant",
                                selection_mode="path_diverse",
                                policy_frontier_only=False,
                                policy_frontier_loss_only=False,
                                source_reservation_contains=None,
                                source_reservation_per_eligible_task=0,
                                require_source_reservation=False,
                            ),
                        )
                    except NoTrainableAssistantDecisionError:
                        rows_without_trainable_assistant_decision += 1
                        if success:
                            skipped_evidence_only += 1
                        else:
                            skipped_unsuccessful += 1
                        continue
                    unsafe_demonstrations_evidence_only += 1
                if evidence_example is None:
                    raise AssertionError("evidence compilation unexpectedly filtered a row")
                evidence_compiled.append(evidence_example)
                if success:
                    skipped_evidence_only += 1
                else:
                    skipped_unsuccessful += 1
            else:
                try:
                    example = compile_record(
                        record,
                        domain_override=domain,
                        schemas=schemas,
                        config=config,
                    )
                except NoTrainableAssistantDecisionError:
                    if config.selection_mode not in {"learning_value", "candidate_pool"}:
                        raise
                    rows_without_trainable_assistant_decision += 1
                    skipped_evidence_only += 1
                    continue
                except UnknownToolCallError:
                    if config.selection_mode not in {"learning_value", "candidate_pool"}:
                        raise
                    evidence_record = copy.deepcopy(record)
                    evidence_record["trajectory_role"] = "evidence_only"
                    metadata = dict(evidence_record.get("metadata") or {})
                    metadata["trajectory_role"] = "evidence_only"
                    metadata["evidence_only_reason"] = "unknown_tool_call"
                    evidence_record["metadata"] = metadata
                    try:
                        evidence_example = compile_record(
                            evidence_record,
                            domain_override=domain,
                            schemas=schemas,
                            config=replace(
                                config,
                                success_only=False,
                                supervision_mode="all_assistant",
                                selection_mode="path_diverse",
                                policy_frontier_only=False,
                                policy_frontier_loss_only=False,
                                source_reservation_contains=None,
                                source_reservation_per_eligible_task=0,
                                require_source_reservation=False,
                            ),
                        )
                    except NoTrainableAssistantDecisionError:
                        rows_without_trainable_assistant_decision += 1
                        skipped_evidence_only += 1
                        continue
                    evidence_compiled.append(evidence_example)
                    unsafe_demonstrations_evidence_only += 1
                    skipped_evidence_only += 1
                    continue
                if example is None:
                    skipped_unsuccessful += 1
                else:
                    compiled.append(example)
                    evidence_compiled.append(example)
    # Exact duplicate traces can occur across collector files.
    # The same generic conversation can legitimately appear under different
    # tasks.  Deduplicate collector retries within a task, never across task
    # identities, otherwise the task-disjoint split silently collapses.
    unique, dedup_audit = _deduplicate_compiled_examples(
        compiled,
        preserve_source_contains=config.source_reservation_contains,
    )
    evidence_unique = {
        (
            str(example["source_task_id"]),
            str(example["trace_hash"]),
            bool(example.get("success")),
            str(example.get("trajectory_role") or "demonstration"),
            (
                str(example.get("source") or "")
                if _matches_reserved_source(
                    example, config.source_reservation_contains
                )
                else ""
            ),
        ): example
        for example in evidence_compiled
    }
    selection_audit: JsonObject | None = None
    if config.selection_mode == "learning_value":
        selected, selection_audit = _select_learning_value_per_task(
            unique.values(),
            evidence_examples=evidence_unique.values(),
            max_traces_per_task=config.max_traces_per_task,
            require_supported_references=config.require_supported_references,
            source_reservation_contains=config.source_reservation_contains,
            source_reservation_per_eligible_task=(
                config.source_reservation_per_eligible_task
            ),
            require_source_reservation=config.require_source_reservation,
        )
        if config.policy_frontier_only:
            selected_before_frontier_filter = len(selected)
            selected = [
                row
                for row in selected
                if row.get("trajectory_role") == "retention_anchor"
                or int((row.get("curation") or {}).get("policy_gap_decisions", 0))
                > 0
            ]
            selection_audit["selected_before_policy_frontier_filter"] = (
                selected_before_frontier_filter
            )
            selection_audit["selected_after_policy_frontier_filter"] = len(selected)
            if not selected:
                raise ValueError(
                    "policy_frontier_only found no policy-gap targets or anchors"
                )
        if config.policy_frontier_loss_only:
            selected_before_loss_filter = len(selected)
            focused: list[JsonObject] = []
            supervised_before = 0
            supervised_after = 0
            for row in selected:
                copied = dict(row)
                existing = {
                    int(index)
                    for index in copied.get("supervised_assistant_turn_indices")
                    or []
                }
                supervised_before += len(existing)
                if copied.get("trajectory_role") == "retention_anchor":
                    target_turns = existing
                else:
                    curation = copied.get("curation") or {}
                    gap_turns = {
                        int(index)
                        for index in curation.get(
                            "policy_gap_assistant_turn_indices", []
                        )
                    }
                    # A row is a usable frontier target only when at least one
                    # successful gap decision is also eligible for SFT loss.
                    # This catches repeated/rejected turns that selection may
                    # legitimately retain as evidence but must not train on.
                    supervised_gap_turns = gap_turns & existing
                    if not supervised_gap_turns:
                        continue
                    external_turns = {
                        int(index)
                        for index in curation.get(
                            "external_event_instruction_turn_indices", []
                        )
                    }
                    # Preserve the demonstrated stop/handoff decision while
                    # focusing all other loss on measured policy gaps and
                    # context-certified external-event instructions.
                    terminal_turns = {max(existing)} if existing else set()
                    target_turns = (
                        supervised_gap_turns | external_turns | terminal_turns
                    ) & existing
                if not target_turns:
                    continue
                copied["supervised_assistant_turn_indices"] = sorted(target_turns)
                copied["supervised_assistant_turns"] = len(target_turns)
                supervised_after += len(target_turns)
                focused.append(copied)
            selected = focused
            selection_audit["selected_before_policy_frontier_loss_filter"] = (
                selected_before_loss_filter
            )
            selection_audit["selected_after_policy_frontier_loss_filter"] = len(
                selected
            )
            selection_audit["supervised_turns_before_policy_frontier_mask"] = (
                supervised_before
            )
            selection_audit["supervised_turns_after_policy_frontier_mask"] = (
                supervised_after
            )
            selection_audit[
                "selected_policy_gap_trajectories_after_filters"
            ] = sum(
                int((row.get("curation") or {}).get("policy_gap_decisions", 0) > 0)
                for row in selected
            )
            selection_audit[
                "selected_external_event_instruction_trajectories_after_filters"
            ] = sum(
                int(
                    (row.get("curation") or {}).get(
                        "external_event_instruction_decisions", 0
                    )
                    > 0
                )
                for row in selected
            )
            if not selected:
                raise ValueError(
                    "policy_frontier_loss_only found no supervised gap targets"
                )
    elif config.selection_mode == "candidate_pool":
        selected = sorted(
            unique.values(),
            key=lambda row: (
                str(row["source_task_id"]),
                str(row["trace_hash"]),
            ),
        )
        selection_audit = {
            "mode": "candidate_pool",
            "selected_successful_eligible_trajectories": len(selected),
            "per_task_cap_applied": False,
            "learning_value_ranking_applied": False,
            "training_permitted": False,
        }
    else:
        selected = _select_per_task(
            unique.values(), max_traces_per_task=config.max_traces_per_task
        )
    train: list[JsonObject] = []
    validation: list[JsonObject] = []
    for example in selected:
        target = (
            validation
            if _is_validation_task(
                str(example["source_task_id"]),
                seed=config.seed,
                fraction=config.validation_fraction,
            )
            else train
        )
        copied = dict(example)
        copied["split"] = "validation" if target is validation else "train"
        target.append(copied)
    train_tasks = {str(row["source_task_id"]) for row in train}
    validation_tasks = {str(row["source_task_id"]) for row in validation}
    if train_tasks & validation_tasks:
        raise AssertionError("task-disjoint split invariant violated")
    manifest: JsonObject = {
        "format": "trajectory2gym_sft_manifest_v0",
        "learning_value_contract": learning_value_contract(),
        "config": {
            "seed": config.seed,
            "validation_fraction": config.validation_fraction,
            "max_traces_per_task": config.max_traces_per_task,
            "success_only": config.success_only,
            "assume_success_when_missing": config.assume_success_when_missing,
            "enable_thinking": config.enable_thinking,
            "supervision_mode": config.supervision_mode,
            "selection_mode": config.selection_mode,
            "include_successful_policy_anchors": (
                config.include_successful_policy_anchors
            ),
            "policy_frontier_only": config.policy_frontier_only,
            "policy_frontier_loss_only": config.policy_frontier_loss_only,
            "require_supported_references": (
                config.require_supported_references
            ),
            "source_reservation_contains": (
                config.source_reservation_contains
            ),
            "source_reservation_per_eligible_task": (
                config.source_reservation_per_eligible_task
            ),
            "require_source_reservation": (
                config.require_source_reservation
            ),
        },
        "input_records": input_records,
        "skipped_unsuccessful": skipped_unsuccessful,
        "skipped_evidence_only": skipped_evidence_only,
        "unsafe_demonstrations_evidence_only": unsafe_demonstrations_evidence_only,
        "successful_policy_anchors": successful_policy_anchors,
        "rows_without_trainable_assistant_decision": (
            rows_without_trainable_assistant_decision
        ),
        "compiled_records": len(compiled),
        "unique_traces": len(unique),
        **dedup_audit,
        "selected_traces": len(selected),
        "selected_retention_anchors": sum(
            int(row.get("trajectory_role") == "retention_anchor")
            for row in selected
        ),
        "selected_verified_task_anchors": sum(
            int(row.get("verified_task_anchor") is True) for row in selected
        ),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_tasks": len(train_tasks),
        "validation_tasks": len(validation_tasks),
        "train_rows_by_domain": dict(sorted(Counter(row["domain"] for row in train).items())),
        "validation_rows_by_domain": dict(
            sorted(Counter(row["domain"] for row in validation).items())
        ),
        "train_assistant_turns": sum(int(row["assistant_turns"]) for row in train),
        "train_supervised_assistant_turns": sum(
            int(row["supervised_assistant_turns"]) for row in train
        ),
        "train_decision_certified_trajectories": sum(
            int(bool(row.get("verified_assistant_turn_indices")))
            for row in train
        ),
        "train_decision_certified_turns": sum(
            len(row.get("verified_assistant_turn_indices") or [])
            for row in train
        ),
        "missing_schema_domains": sorted(
            {str(row["domain"]) for row in selected if not row.get("tools")}
        ),
        "task_overlap": sorted(train_tasks & validation_tasks),
        "source_inputs": [
            {
                "domain_override": domain,
                "path": str(path.resolve()),
                "sha256": _file_sha256(path),
            }
            for domain, path in inputs
        ],
        "schema_inputs": [
            {
                "path": str(path.resolve()),
                "sha256": _file_sha256(path),
            }
            for path in schema_sources
        ],
        "method_contract": {
            "format": "trajectory2gym_training_view_contract_v0",
            "view": (
                "certified_candidate_pool"
                if config.selection_mode == "candidate_pool"
                else "agentic_sft"
            ),
            "learning_value_contract_id": (
                learning_value_contract()["contract_id"]
            ),
            "successful_trajectories_only": config.success_only,
            "explicit_success_certificate_required": (
                not config.assume_success_when_missing
            ),
            "failed_evidence_selected_as_target": bool(
                selection_audit
                and selection_audit.get("failed_trajectories_selected")
            ),
            "failed_and_probe_sessions_used_as_graph_evidence": (
                config.selection_mode == "learning_value"
            ),
            "full_trajectory_unit": True,
            "benchmark_neutral_selection": True,
            "verified_task_anchor_precedes_value_ranking": True,
            "source_reservation_precedes_general_value_ranking": (
                config.source_reservation_contains is not None
            ),
            "verified_task_anchor_survives_exact_trace_dedup": True,
            "decision_certificate_can_enable_sft_without_observation": True,
            "decision_certificate_can_create_environment_edge": False,
        },
        "trace_lineage": {
            "selected_trace_hashes": len(
                {str(row["trace_hash"]) for row in selected}
            ),
            "selected_trace_hashes_sha256": _set_digest(
                str(row["trace_hash"]) for row in selected
            ),
            "evidence_trace_hashes": len(
                {str(row["trace_hash"]) for row in evidence_unique.values()}
            ),
            "evidence_trace_hashes_sha256": _set_digest(
                str(row["trace_hash"]) for row in evidence_unique.values()
            ),
        },
    }
    if selection_audit is not None:
        selection_audit["input_unique_evidence_records"] = len(evidence_unique)
        selection_audit["input_unique_failure_evidence_records"] = sum(
            int(not bool(row.get("success"))) for row in evidence_unique.values()
        )
        manifest["selection_audit"] = selection_audit
    return train, validation, manifest


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

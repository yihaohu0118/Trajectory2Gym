from __future__ import annotations

import ast
import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .schema import ToolAction, TrajectoryEpisode
from .recovery import observation_is_explicit_error
from .tool_semantics import DEFAULT_TOOL_SEMANTICS, ToolSemantics


@dataclass(frozen=True)
class ObservationRecord:
    """One observed tool transition recovered from a static trajectory."""

    key: str
    action: ToolAction
    observations: tuple[dict[str, Any], ...]
    episode_id: str
    step_index: int
    context_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "action": self.action.to_dict(),
            "observations": copy.deepcopy(list(self.observations)),
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "context_key": self.context_key,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class ObservationLookup:
    """Lookup result for a candidate tool action."""

    status: str
    records: tuple[ObservationRecord, ...] = ()
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.status == "hit"

    @property
    def observations(self) -> tuple[dict[str, Any], ...]:
        if not self.records:
            return ()
        return self.records[0].observations

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "records": [record.to_dict() for record in self.records],
            "reason": self.reason,
        }


class ObservationDatabase:
    """Conservative query-observation table compiled from trajectories.

    The database only accepts exact canonical action keys. If the same action key
    maps to conflicting observations, lookup returns `conflict` instead of
    picking one. This keeps the partial environment sound by default.
    """

    def __init__(self, records: list[ObservationRecord] | tuple[ObservationRecord, ...] = ()):
        self.records = tuple(records)
        self._by_key: dict[str, list[ObservationRecord]] = {}
        self._by_context_key: dict[tuple[str, str], list[ObservationRecord]] = {}
        for record in self.records:
            self._by_key.setdefault(record.key, []).append(record)
            if record.context_key:
                self._by_context_key.setdefault((record.context_key, record.key), []).append(record)

    @classmethod
    def from_episodes(
        cls,
        episodes: list[TrajectoryEpisode] | tuple[TrajectoryEpisode, ...],
        *,
        tool_semantics: ToolSemantics = DEFAULT_TOOL_SEMANTICS,
    ) -> "ObservationDatabase":
        records: list[ObservationRecord] = []
        for episode in episodes:
            for position, step in enumerate(episode.steps):
                if len(step.expected_actions) != 1 or not step.observations:
                    continue
                action = step.expected_actions[0]
                observations, projection = observation_evidence_messages(
                    action, step.observations
                )
                records.append(
                    ObservationRecord(
                        key=action_key(action),
                        action=action,
                        observations=observations,
                        episode_id=episode.id,
                        step_index=step.index,
                        context_key=observation_context_key(episode, position),
                        metadata={
                            "source": episode.source,
                            "domain": episode.domain,
                            "step_kind": step.kind,
                            "observation_projection": projection,
                            "recorded_observation_count": len(step.observations),
                            "evidence_observation_count": len(observations),
                            "explicit_error_observation": any(
                                observation_is_explicit_error(observation)
                                for observation in observations
                            ),
                            "prefix_has_state_change": prefix_has_state_change(
                                episode, position, tool_semantics=tool_semantics
                            ),
                        },
                    )
                )
        return cls(records)

    def lookup(
        self,
        action: ToolAction,
        *,
        context_key: str = "",
        allow_context_free: bool = True,
    ) -> ObservationLookup:
        key = action_key(action)
        records = tuple(self._by_key.get(key, ()))
        if not records:
            return ObservationLookup(status="miss", reason="no observation for canonical action key")
        # An identical recorded visible prefix is stronger evidence than a
        # global action key and is required for transitions observed after a
        # state change.
        if context_key:
            contextual_records = tuple(self._by_context_key.get((context_key, key), ()))
            contextual_signatures = {
                _observation_signature(record.observations) for record in contextual_records
            }
            if contextual_records and len(contextual_signatures) == 1:
                return ObservationLookup(
                    status="hit",
                    records=contextual_records,
                    reason="prefix context disambiguates conflicting global observations",
                )
            if len(contextual_signatures) > 1:
                return ObservationLookup(
                    status="conflict",
                    records=contextual_records,
                    reason="same action and prefix context have conflicting observations",
                )
        context_free_records = tuple(
            record
            for record in records
            if not bool(record.metadata.get("prefix_has_state_change", False))
        )
        if allow_context_free and context_free_records:
            context_free_signatures = {
                _observation_signature(record.observations) for record in context_free_records
            }
            if len(context_free_signatures) == 1:
                return ObservationLookup(
                    status="hit",
                    records=context_free_records,
                    reason="context-free action evidence has one observation",
                )
            return ObservationLookup(
                status="conflict",
                records=context_free_records,
                reason="context-free canonical action key has conflicting observations",
            )
        if not allow_context_free:
            return ObservationLookup(
                status="context_required",
                records=records,
                reason="current visible prefix contains a state change and has no exact contextual evidence",
            )
        if not context_free_records:
            return ObservationLookup(
                status="context_required",
                records=records,
                reason="action evidence was observed only after a state change",
            )
        return ObservationLookup(
            status="conflict",
            records=records,
            reason="same canonical action key has conflicting observations and no unique prefix-context match",
        )

    def records_for_tool(self, tool_name: str) -> tuple[ObservationRecord, ...]:
        """Return all observed records for one tool name.

        This is used by audits to construct database-supported non-gold probes.
        Runtime lookup remains conservative and still accepts only exact
        canonical action keys with non-conflicting observations.
        """

        return tuple(record for record in self.records if record.action.name == tool_name)

    def summary(self) -> dict[str, Any]:
        conflicts = 0
        for records in self._by_key.values():
            signatures = {_observation_signature(record.observations) for record in records}
            if len(signatures) > 1:
                conflicts += 1
        return {
            "record_count": len(self.records),
            "unique_action_count": len(self._by_key),
            "conflict_key_count": conflicts,
        }


def action_key(action: ToolAction) -> str:
    return json.dumps(
        {"name": action.name, "arguments": _normalize_value(action.arguments)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def observation_evidence_messages(
    action: ToolAction,
    observations: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Project a transition to the directly observed tool result, when marked.

    Some trajectory formats place the immediate tool result and subsequent
    assistant/user dialogue in the same transition observation list.  The
    latter is valid for exact trajectory replay, but it is not evidence about
    the tool's input-output mapping and must not create cross-trajectory DB
    conflicts.  Standard tool/function messages, tool-call-id messages, and
    JSON result envelopes naming the invoked tool are format-level evidence.

    Unmarked observations retain the historical all-message behavior.  This
    conservative fallback avoids guessing where a dataset has no structural
    boundary marker.
    """

    messages = tuple(copy.deepcopy(list(observations)))
    direct: list[dict[str, Any]] = []
    for message in messages:
        if not _is_direct_tool_observation(action, message):
            break
        direct.append(copy.deepcopy(message))
    if direct:
        return tuple(direct), "direct_tool_result"
    return messages, "unmarked_full_observation"


def _is_direct_tool_observation(action: ToolAction, message: dict[str, Any]) -> bool:
    role = str(message.get("role") or "").strip().lower()
    if role in {"tool", "function"} or message.get("tool_call_id"):
        return True

    content = message.get("content")
    payload: Any = content
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
    if not isinstance(payload, dict):
        return False
    payload_name = str(payload.get("name") or payload.get("tool_name") or "")
    if payload_name != action.name:
        return False
    # A matching name alone may describe another action.  At least one common
    # result-envelope field is required before treating the message as output.
    return any(
        key in payload
        for key in ("response", "result", "output", "data", "error", "tool_reward")
    )


def observation_context_key(episode: TrajectoryEpisode, step_position: int) -> str:
    """Canonical evidence key for the visible prefix before one transition.

    The key uses only recorded messages, actions and observations. It never
    guesses hidden backend state. A global action conflict can therefore be
    resolved only when an identical observed prefix has unique evidence.
    """

    position = max(0, min(int(step_position), len(episode.steps)))
    prefix = [
        {
            "actions": [action.to_dict() for action in step.expected_actions],
            "observations": copy.deepcopy(list(step.observations)),
            "kind": step.kind,
            "action_metadata": copy.deepcopy(step.metadata.get("action_metadata") or []),
            "external_actions": copy.deepcopy(step.metadata.get("external_actions") or []),
        }
        for step in episode.steps[:position]
    ]
    return json.dumps(
        _normalize_value(
            {
                "initial_messages": [
                    canonical_visible_message(message)
                    for message in episode.initial_messages
                ],
                # This compiler ledger preserves state-changing external
                # actions and their declarations when decision-window
                # construction moves transitions into initial_messages.
                "recorded_prefix_actions": copy.deepcopy(
                    episode.metadata.get("recorded_prefix_actions") or []
                ),
                "prefix": prefix,
            }
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def prefix_has_state_change(
    episode: TrajectoryEpisode,
    step_position: int,
    *,
    tool_semantics: ToolSemantics = DEFAULT_TOOL_SEMANTICS,
) -> bool:
    """Whether the recorded visible prefix contains an external-state write."""

    # Decision-window compilation moves prior transitions into the visible
    # message prefix. Honor the compiler-preserved mutation declaration.
    for item in episode.metadata.get("recorded_prefix_actions") or ():
        if not isinstance(item, dict):
            continue
        if item.get("mutates_state") is True:
            return True
        if tool_semantics.is_side_effecting(str(item.get("name") or "")):
            return True

    # Recognize native tool calls in arbitrary pre-windowed trajectory inputs
    # as a conservative fallback. Such messages lack explicit mutation flags,
    # so the thin declarative adapter semantics decide the boundary.
    for message in episode.initial_messages:
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                name = function.get("name") if isinstance(function, dict) else call.get("name")
                if tool_semantics.is_side_effecting(str(name or "")):
                    return True
        function_call = message.get("function_call")
        if isinstance(function_call, dict) and tool_semantics.is_side_effecting(
            str(function_call.get("name") or "")
        ):
            return True

    position = max(0, min(int(step_position), len(episode.steps)))
    for step in episode.steps[:position]:
        raw_metadata = step.metadata.get("action_metadata")
        for index, action in enumerate(step.expected_actions):
            declared_mutation = None
            if (
                isinstance(raw_metadata, list)
                and index < len(raw_metadata)
                and isinstance(raw_metadata[index], dict)
            ):
                candidate = raw_metadata[index].get("mutates_state")
                if isinstance(candidate, bool):
                    declared_mutation = candidate
            # Explicit writes and conservative name semantics both veto
            # context-free replay.  A false declaration never overrides a
            # tool name already known to be side effecting.
            if declared_mutation is True or tool_semantics.is_side_effecting(action.name):
                return True
        if step.kind == "external_event":
            external_actions = step.metadata.get("external_actions") or ()
            for index, action in enumerate(external_actions):
                if not isinstance(action, dict):
                    continue
                declared_mutation = None
                if (
                    isinstance(raw_metadata, list)
                    and index < len(raw_metadata)
                    and isinstance(raw_metadata[index], dict)
                ):
                    candidate = raw_metadata[index].get("mutates_state")
                    if isinstance(candidate, bool):
                        declared_mutation = candidate
                if declared_mutation is True or tool_semantics.is_side_effecting(
                    str(action.get("name") or "")
                ):
                    return True
    return False


_TOOL_CALL_CONTENT_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)


def canonical_visible_message(message: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize policy-visible transport without changing its semantics.

    Transition GRPO compares decisions only when the complete pre-action
    transcript is the same.  Runtime tool calls can, however, reach that
    transcript either as native ``tool_calls`` or as textual
    ``<tool_call>JSON</tool_call>`` content.  Random call IDs, JSON key order,
    whitespace, and string-vs-object argument encoding are transport details,
    not different policy states.

    A call-only assistant payload is projected to the common native
    representation.  For mixed text/tool content, only each parseable tool
    tag is canonicalized and every ordinary-text byte remains visible.
    Malformed calls, tool observations, and argument values remain distinct.
    The projection inspects only actions already visible in the transcript;
    it has no access to an expected or gold action.
    """

    result = copy.deepcopy(message)
    result.pop("tool_call_id", None)

    native_actions = _canonical_actions_from_native_calls(result.get("tool_calls"))
    function_action = _canonical_tool_action(result.get("function_call"))
    if native_actions is None and function_action is not None:
        native_actions = [function_action]

    content_projection = None
    if str(result.get("role") or "").strip().lower() == "assistant":
        content_projection = _canonical_tool_content_projection(result.get("content"))

    # A call-only textual payload and a native call are two encodings of the
    # same visible action.  If both are present, collapse them only when they
    # agree; disagreement remains visible and therefore cannot be grouped.
    if content_projection is not None:
        content_actions, canonical_content, call_only = content_projection
        if native_actions is None or native_actions == content_actions:
            native_actions = content_actions
            result["content"] = None if call_only else canonical_content

    if native_actions is not None:
        result["tool_calls"] = [
            {"function": copy.deepcopy(action)} for action in native_actions
        ]
        result.pop("function_call", None)
        # Native APIs variously omit content, send null, or send an empty
        # string beside a tool call.  These are the same transport state.
        if not str(result.get("content") or "").strip():
            result["content"] = None
    else:
        # Preserve malformed/native non-action structures, but still discard
        # their local call IDs as the historical implementation did.
        calls = result.get("tool_calls")
        if isinstance(calls, list):
            canonical_calls: list[Any] = []
            for call in calls:
                canonical = copy.deepcopy(call)
                if isinstance(canonical, dict):
                    canonical.pop("id", None)
                canonical_calls.append(canonical)
            result["tool_calls"] = canonical_calls
    return result


def _canonical_actions_from_native_calls(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or not value:
        return None
    actions: list[dict[str, Any]] = []
    for item in value:
        action = _canonical_tool_action(item)
        if action is None:
            return None
        actions.append(action)
    return actions


def _canonical_tool_content_projection(
    value: Any,
) -> tuple[list[dict[str, Any]], str, bool] | None:
    if not isinstance(value, str) or not value.strip():
        return None

    matches = list(_TOOL_CALL_CONTENT_PATTERN.finditer(value))
    actions: list[dict[str, Any]] = []
    if matches:
        canonical_parts: list[str] = []
        ordinary_parts: list[str] = []
        cursor = 0
        for match in matches:
            ordinary = value[cursor : match.start()]
            ordinary_parts.append(ordinary)
            canonical_parts.append(ordinary)
            parsed = _parse_tool_payload(match.group(1))
            if parsed is None:
                return None
            payload_actions = _canonical_actions_from_payload(parsed)
            if not payload_actions:
                return None
            actions.extend(payload_actions)
            canonical_parts.extend(
                "<tool_call>"
                + json.dumps(
                    action,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "</tool_call>"
                for action in payload_actions
            )
            cursor = match.end()
        ordinary = value[cursor:]
        ordinary_parts.append(ordinary)
        canonical_parts.append(ordinary)
        return actions, "".join(canonical_parts), not "".join(ordinary_parts).strip()
    else:
        # Some tool-capable APIs emit one bare JSON action.  Accept it only
        # when the entire content parses as an action payload.
        parsed = _parse_tool_payload(value)
        if parsed is None:
            return None
        payload_actions = _canonical_actions_from_payload(parsed)
        if not payload_actions:
            return None
        return payload_actions, "", True


def _parse_tool_payload(value: str) -> Any | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return None


def _canonical_actions_from_payload(payload: Any) -> list[dict[str, Any]] | None:
    if isinstance(payload, dict) and isinstance(payload.get("tool_call"), dict):
        return _canonical_actions_from_payload(payload["tool_call"])
    if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
        return _canonical_actions_from_payload(payload["tool_calls"])
    if isinstance(payload, list):
        if not payload:
            return None
        actions: list[dict[str, Any]] = []
        for item in payload:
            action = _canonical_tool_action(item)
            if action is None:
                return None
            actions.append(action)
        return actions
    action = _canonical_tool_action(payload)
    return [action] if action is not None else None


def _canonical_tool_action(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("tool_call"), dict):
        return _canonical_tool_action(payload["tool_call"])
    if isinstance(payload.get("function"), dict):
        function = payload["function"]
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = payload.get("name") or payload.get("function")
        arguments = payload.get("arguments", payload.get("parameters", {}))
    if not str(name or "").strip():
        return None
    normalized_arguments = _canonical_tool_arguments(arguments)
    if normalized_arguments is None:
        return None
    return {
        "name": str(name),
        "arguments": normalized_arguments,
    }


def _canonical_tool_arguments(arguments: Any) -> dict[str, Any] | None:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return copy.deepcopy(arguments)
    if not isinstance(arguments, str):
        return None
    stripped = arguments.strip()
    if not stripped:
        return {}
    parsed = _parse_tool_payload(stripped)
    return copy.deepcopy(parsed) if isinstance(parsed, dict) else None


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.strip().split())
    return value


def _observation_signature(observations: tuple[dict[str, Any], ...]) -> str:
    return json.dumps(
        _normalize_value(copy.deepcopy(list(observations))),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

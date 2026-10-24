"""Compile immutable trajectory prefixes into auditable reset plans.

The plans in this module do not mutate or simulate an environment.  A real
environment adapter must execute every recorded assistant action again and
compare the resulting observation with the expected message before exposing
the restored state to a policy.  This keeps the compiler benchmark-neutral:
it describes evidence, while the adapter remains the transition oracle.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


JsonObject = dict[str, Any]
VERIFIED_PREFIX_CERTIFICATE = "trajectory_verified_prefix_reset_v1"


@dataclass(frozen=True)
class VerifiedPrefixTurn:
    """One recorded assistant action and its expected emitted message."""

    assistant: JsonObject
    expected: JsonObject


@dataclass(frozen=True)
class VerifiedPrefixPlan:
    """A deterministic plan that ends immediately before an assistant turn."""

    reference: str
    trace_hash: str
    domain: str
    task_id: str
    frontier_assistant_turn: int
    system_message: JsonObject
    initial_user_message: JsonObject
    turns: tuple[VerifiedPrefixTurn, ...]
    certificate: str = VERIFIED_PREFIX_CERTIFICATE

    @property
    def prefix_messages(self) -> tuple[JsonObject, ...]:
        messages = [
            copy.deepcopy(self.system_message),
            copy.deepcopy(self.initial_user_message),
        ]
        for turn in self.turns:
            messages.append(copy.deepcopy(turn.assistant))
            messages.append(copy.deepcopy(turn.expected))
        return tuple(messages)


class VerifiedPrefixIndex:
    """Content-addressed prefix plans built from recorded conversations."""

    def __init__(self, plans: Mapping[str, VerifiedPrefixPlan]) -> None:
        self._plans = dict(plans)

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]) -> "VerifiedPrefixIndex":
        records: list[JsonObject] = []
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"{path}:{line_number}: prefix record must be an object"
                        )
                    records.append(value)
        return cls.from_records(records)

    @classmethod
    def from_records(
        cls, records: Iterable[Mapping[str, Any]]
    ) -> "VerifiedPrefixIndex":
        plans: dict[str, VerifiedPrefixPlan] = {}
        for record in records:
            for plan in plans_from_record(record):
                previous = plans.get(plan.reference)
                if previous is not None and previous != plan:
                    raise ValueError(
                        "conflicting verified prefixes share reference "
                        f"{plan.reference!r}"
                    )
                plans[plan.reference] = plan
        return cls(plans)

    def get(self, reference: str) -> VerifiedPrefixPlan:
        try:
            return self._plans[str(reference)]
        except KeyError as exc:
            raise KeyError(
                f"unknown verified-prefix reference: {reference!r}"
            ) from exc

    def __len__(self) -> int:
        return len(self._plans)

    @property
    def references(self) -> tuple[str, ...]:
        return tuple(sorted(self._plans))


def plans_from_record(record: Mapping[str, Any]) -> list[VerifiedPrefixPlan]:
    """Return every certified non-initial frontier in one conversation."""

    messages = record.get("messages")
    legacy_depth = 0
    legacy = not isinstance(messages, Sequence) or isinstance(messages, (str, bytes))
    if legacy:
        # Early verified-prefix compilers emitted a compact reset row with
        # ``initial_messages`` plus the frontier successor in ``steps``.  It
        # is still authentic trajectory evidence, but it is not in the newer
        # ``messages``/certificate envelope.  Reconstructing the immutable
        # message view here keeps old data usable without weakening the
        # official environment: the backend still replays every prefix edge
        # and checks each observation before reset.
        messages = _legacy_messages(record)
        if not messages:
            return []
        metadata = record.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        try:
            legacy_depth = int(metadata.get("verified_prefix_depth") or 0)
        except (TypeError, ValueError):
            legacy_depth = 0
        if legacy_depth <= 0:
            reference = str(metadata.get("real_reset_reference") or "")
            parts = reference.split(":")
            if len(parts) >= 3 and parts[0] == "vprefix":
                try:
                    legacy_depth = int(parts[-1])
                except ValueError:
                    legacy_depth = 0
        if legacy_depth <= 0:
            return []
    copied = [copy.deepcopy(dict(message)) for message in messages if isinstance(message, Mapping)]
    if len(copied) != len(messages) or len(copied) < 3:
        return []
    if copied[0].get("role") != "system" or copied[1].get("role") != "user":
        return []

    domain = str(record.get("domain") or "").strip()
    task_id = _official_task_id(record)
    # Compiled frontier/prefix records historically stored the content hash
    # under ``metadata.source_trace_hash`` (and some older producers used the
    # top-level ``source_trace_hash`` key), while the canonical trajectory
    # schema uses ``trace_hash``.  The reset reference is content-addressed,
    # so accepting all three representations is semantics-preserving and
    # avoids silently making a valid state-level training row unrunnable.
    metadata = record.get("metadata")
    metadata_trace_hash = (
        metadata.get("source_trace_hash")
        if isinstance(metadata, Mapping)
        else ""
    )
    trace_hash = str(
        record.get("trace_hash")
        or record.get("source_trace_hash")
        or metadata.get("verified_prefix_trace_hash")
        or metadata_trace_hash
        or ""
    ).strip()
    if not trace_hash:
        trace_hash = _canonical_hash(copied)
    certificates = record.get("state_certificates_by_assistant_turn")
    certificates = certificates if isinstance(certificates, Mapping) else {}
    if legacy:
        metadata = record.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        pre_state_id = str(
            metadata.get("trajectory_state_id")
            or ((record.get("steps") or [{}])[0].get("metadata") or {}).get(
                "pre_state_id", ""
            )
        ).strip()
        if pre_state_id:
            certificates = {str(legacy_depth): {"pre_state_id": pre_state_id}}

    assistant_turn = -1
    plans: list[VerifiedPrefixPlan] = []
    for message_index, message in enumerate(copied):
        if message.get("role") != "assistant":
            continue
        assistant_turn += 1
        # Turn zero is the ordinary initial reset and adds no restorable
        # backend state.  Every later turn must name a certified visible state.
        if assistant_turn == 0:
            continue
        certificate = certificates.get(str(assistant_turn))
        if not isinstance(certificate, Mapping) or not str(
            certificate.get("pre_state_id") or ""
        ).strip():
            continue
        prefix = copied[:message_index]
        parsed = _prefix_turns(prefix)
        if parsed is None:
            continue
        reference = verified_prefix_reference(trace_hash, assistant_turn)
        plans.append(
            VerifiedPrefixPlan(
                reference=reference,
                trace_hash=trace_hash,
                domain=domain,
                task_id=task_id,
                frontier_assistant_turn=assistant_turn,
                system_message=copy.deepcopy(prefix[0]),
                initial_user_message=copy.deepcopy(prefix[1]),
                turns=parsed,
            )
        )
    return plans


def _legacy_messages(record: Mapping[str, Any]) -> list[JsonObject]:
    """Expand the pre-v1 compact prefix row into canonical chat messages.

    This is a serialization adapter only.  It copies recorded assistant
    actions and observations; it never invents an observation or a state
    transition.  The real adapter remains responsible for replaying and
    validating this complete prefix against the official backend.
    """

    initial = record.get("initial_messages")
    if not isinstance(initial, Sequence) or isinstance(initial, (str, bytes)):
        return []
    messages = [copy.deepcopy(dict(message)) for message in initial if isinstance(message, Mapping)]
    if len(messages) != len(initial):
        return []
    for step_index, raw_step in enumerate(record.get("steps") or ()):
        if not isinstance(raw_step, Mapping):
            return []
        metadata = raw_step.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        actions = raw_step.get("expected_actions") or ()
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            return []
        tool_calls: list[JsonObject] = []
        for action_index, raw_action in enumerate(actions):
            if not isinstance(raw_action, Mapping):
                return []
            name = str(raw_action.get("name") or "").strip()
            if not name:
                return []
            arguments = copy.deepcopy(dict(raw_action.get("arguments") or {}))
            tool_calls.append(
                {
                    "id": f"legacy-prefix-{step_index}-{action_index}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        assistant: JsonObject = {
            "role": "assistant",
            "content": str(metadata.get("expected_assistant_message") or ""),
        }
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)
        observations = raw_step.get("observations") or ()
        if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
            return []
        messages.extend(
            copy.deepcopy(dict(observation))
            for observation in observations
            if isinstance(observation, Mapping)
        )
        if len(messages) < len(initial) + step_index + 1 + len(observations):
            return []
    return messages


def verified_prefix_reference(trace_hash: str, assistant_turn: int) -> str:
    if assistant_turn <= 0:
        raise ValueError("verified prefix must include at least one assistant turn")
    return f"vprefix:{str(trace_hash).strip()}:{assistant_turn}"


def observation_matches(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    """Compare observation semantics while ignoring transport-only call IDs."""

    if str(actual.get("role") or "") != str(expected.get("role") or ""):
        return False
    if actual.get("role") == "tool" and str(actual.get("name") or "") != str(
        expected.get("name") or ""
    ):
        return False
    return _canonical_content(actual.get("content")) == _canonical_content(
        expected.get("content")
    )


def _prefix_turns(
    prefix: Sequence[Mapping[str, Any]],
) -> tuple[VerifiedPrefixTurn, ...] | None:
    if len(prefix) < 4:
        return None
    turns: list[VerifiedPrefixTurn] = []
    cursor = 2
    while cursor < len(prefix):
        assistant = prefix[cursor]
        if assistant.get("role") != "assistant" or cursor + 1 >= len(prefix):
            return None
        expected = prefix[cursor + 1]
        tool_calls = assistant.get("tool_calls") or []
        if tool_calls:
            if not isinstance(tool_calls, list) or len(tool_calls) != 1:
                return None
            if expected.get("role") != "tool":
                return None
        elif expected.get("role") != "user":
            return None
        turns.append(
            VerifiedPrefixTurn(
                assistant=copy.deepcopy(dict(assistant)),
                expected=copy.deepcopy(dict(expected)),
            )
        )
        cursor += 2
    return tuple(turns) if turns else None


def _official_task_id(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    raw = str(
        record.get("source_task_id")
        or record.get("task_id")
        or metadata.get("source_task_id")
        or ""
    ).strip()
    if not raw:
        raise ValueError("verified-prefix record lacks a source task ID")
    tail = raw.rsplit(":", 1)[-1]
    return tail if tail.isdigit() else raw


def _canonical_content(value: Any) -> tuple[str, str]:
    text = str(value or "")
    try:
        decoded = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return "text", text
    return (
        "json",
        json.dumps(
            decoded,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

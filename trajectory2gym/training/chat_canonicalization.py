"""Compile native chat trajectories into the neutral trajectory schema.

This module deliberately knows nothing about Tau, BFCL, or any tool name.  A
dataset integration supplies native chat rows plus declarative schemas; the
compiler only separates initial context from assistant decisions and their
recorded observations.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


JsonObject = dict[str, Any]


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def load_schema_map(paths: Iterable[Path]) -> dict[str, JsonObject]:
    schemas: dict[str, JsonObject] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError(f"{path}:{line_number}: schema row is not an object")
                domain = str(row.get("domain") or "").strip()
                tools = row.get("tools")
                if not domain or not isinstance(tools, list) or not tools:
                    raise ValueError(f"{path}:{line_number}: invalid domain/tool schema")
                candidate = {
                    "benchmark": str(row.get("benchmark") or "trajectory"),
                    "tools": copy.deepcopy(tools),
                }
                previous = schemas.get(domain)
                if previous is not None and _canonical(previous) != _canonical(candidate):
                    raise ValueError(f"conflicting schemas for domain {domain!r}")
                schemas[domain] = candidate
    if not schemas:
        raise ValueError("schema sources contain no schemas")
    return schemas


def _messages(record: Mapping[str, Any]) -> list[JsonObject]:
    # Native collectors use all three spellings in practice.  In particular,
    # the Tau importers preserve the recorded chat under the top-level
    # ``traj`` field.  Failing to inspect that field silently rejects every
    # otherwise valid imported rollout in ``canonicalize_chat_rows``.
    candidates = [
        record.get("messages"),
        record.get("traj"),
        record.get("trajectory"),
    ]
    info = record.get("info")
    if isinstance(info, Mapping):
        candidates.extend((info.get("traj"), info.get("trajectory"), info.get("messages")))
    for candidate in candidates:
        if isinstance(candidate, list) and all(isinstance(item, Mapping) for item in candidate):
            return [copy.deepcopy(dict(item)) for item in candidate]
    raise ValueError("record has no native message trajectory")


def _actions(message: Mapping[str, Any]) -> list[JsonObject]:
    actions: list[JsonObject] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, Mapping):
            raise ValueError("tool call is not an object")
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("tool call has no function object")
        name = str(function.get("name") or "").strip()
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not name or not isinstance(arguments, dict):
            raise ValueError("tool call has invalid name or arguments")
        actions.append({"name": name, "arguments": copy.deepcopy(arguments)})
    return actions


def canonicalize_chat_record(
    record: Mapping[str, Any],
    *,
    schemas: Mapping[str, Mapping[str, Any]],
) -> JsonObject:
    domain = str(record.get("domain") or "").strip()
    if domain not in schemas:
        raise ValueError(f"no declarative schema for domain {domain!r}")
    schema = schemas[domain]
    messages = _messages(record)
    assistant_positions = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if not assistant_positions:
        raise ValueError("trajectory has no assistant decision")

    initial_messages = messages[: assistant_positions[0]]
    steps: list[JsonObject] = []
    external_actor_tool_calls = 0
    last_action_step: int | None = None
    source_task_id = str(
        record.get("source_task_id") or record.get("task_id") or ""
    ).strip()
    if not source_task_id:
        raise ValueError("trajectory has no source task id")
    benchmark = str(schema.get("benchmark") or "trajectory")
    raw_metadata = record.get("metadata")
    raw_metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    official_split = str(raw_metadata.get("official_split") or "")
    initial_state_id = _hash(
        {
            "benchmark": benchmark,
            "domain": domain,
            "official_split": official_split,
            "source_task_id": source_task_id,
        }
    )
    current_state_id = initial_state_id
    for assistant_index, message_position in enumerate(assistant_positions):
        message = messages[message_position]
        next_position = (
            assistant_positions[assistant_index + 1]
            if assistant_index + 1 < len(assistant_positions)
            else len(messages)
        )
        observations = copy.deepcopy(messages[message_position + 1 : next_position])
        external_actor_tool_calls += sum(
            len(observation.get("tool_calls") or [])
            for observation in observations
            if observation.get("role") == "user"
        )
        actions = _actions(message)
        metadata: JsonObject = {
            "observation_provenance": "real_environment",
            "assistant_message_recorded": True,
            "pre_state_id": current_state_id,
        }
        content = message.get("content")
        if not actions and content not in (None, ""):
            metadata["expected_assistant_message"] = copy.deepcopy(content)
        if actions:
            last_action_step = len(steps)
        current_state_id = _hash(
            {
                "pre_state_id": current_state_id,
                "assistant_content": copy.deepcopy(content),
                "actions": actions,
                "observations": observations,
            }
        )
        metadata["post_state_id"] = current_state_id
        steps.append(
            {
                "index": len(steps),
                "expected_actions": actions,
                "observations": observations,
                "kind": "tool" if actions else "assistant_text",
                "metadata": metadata,
            }
        )

    success = bool(record.get("success"))
    if success and last_action_step is not None:
        steps[last_action_step]["metadata"]["recorded_graph_terminal"] = True
    group_id = f"{benchmark}:{domain}:{source_task_id}"
    trace_hash = _hash(
        {
            "initial_messages": initial_messages,
            "steps": steps,
            "tools": schema["tools"],
        }
    )
    source_id = str(record.get("id") or record.get("task_id") or source_task_id)
    metadata = copy.deepcopy(dict(raw_metadata))
    metadata.update(
        {
            "source_task_id": source_task_id,
            "trajectory_group_id": group_id,
            "trajectory_state_id": initial_state_id,
            "trace_hash": trace_hash,
            "training_eligible": success,
            "terminal_success_eligible": success,
            "source_session_success": success,
            "terminal_outcome_known": True,
            "complete_verifiable_decision_block": success,
            "decision_window_truncated": False,
            "observation_provenance": "real_environment",
            "external_actor_tool_calls": external_actor_tool_calls,
            "trajectory_role": str(
                record.get("trajectory_role")
                or metadata.get("trajectory_role")
                or "policy_rollout"
            ),
        }
    )
    return {
        "id": f"{benchmark}:{domain}:{source_id}:{trace_hash[:12]}",
        "source": str(record.get("source") or benchmark),
        "domain": domain,
        "task": next(
            (
                str(message.get("content") or "")
                for message in initial_messages
                if message.get("role") == "user"
            ),
            "",
        ),
        "tools": copy.deepcopy(schema["tools"]),
        "initial_messages": initial_messages,
        "steps": steps,
        "metadata": metadata,
    }


def canonicalize_chat_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[list[JsonObject], JsonObject]:
    output: list[JsonObject] = []
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for row in rows:
        counts["input_rows"] += 1
        try:
            episode = canonicalize_chat_record(row, schemas=schemas)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            counts[f"rejected:{type(error).__name__}"] += 1
            continue
        if episode["id"] in seen_ids:
            counts["exact_duplicate_ids"] += 1
            continue
        seen_ids.add(episode["id"])
        output.append(episode)
        counts["output_rows"] += 1
        counts[f"domain:{episode['domain']}"] += 1
        if episode["metadata"]["training_eligible"]:
            counts["training_eligible_rows"] += 1
        else:
            counts["evidence_only_rows"] += 1
        counts["external_actor_tool_calls"] += int(
            episode["metadata"]["external_actor_tool_calls"]
        )
    report = {
        "format": "trajectory2gym_chat_canonicalization_v0",
        "input_rows": counts["input_rows"],
        "output_rows": counts["output_rows"],
        "training_eligible_rows": counts["training_eligible_rows"],
        "evidence_only_rows": counts["evidence_only_rows"],
        "exact_duplicate_ids": counts["exact_duplicate_ids"],
        "external_actor_tool_calls": counts["external_actor_tool_calls"],
        "rows_by_domain": {
            key.removeprefix("domain:"): value
            for key, value in sorted(counts.items())
            if key.startswith("domain:")
        },
        "rejections": {
            key.removeprefix("rejected:"): value
            for key, value in sorted(counts.items())
            if key.startswith("rejected:")
        },
    }
    return output, report

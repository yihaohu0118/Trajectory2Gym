from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from trajectory2gym.tool_calls import (
    actions_from_assistant_message,
    argument_similarity,
    same_actions,
    same_tool_names,
    tool_name_from_schema,
    tool_parameters_from_schema,
)

from .external_request import ground_external_request
from .schema import ToolAction, TrajectoryStep


@dataclass(frozen=True)
class SchemaCheck:
    tool_name: str
    valid: bool
    reason: str = "schema_valid"

    def to_dict(self) -> dict[str, Any]:
        return {"tool_name": self.tool_name, "valid": self.valid, "reason": self.reason}


@dataclass(frozen=True)
class ActionMatchResult:
    """Neutral action matching result shared by trajectory2gym adapters."""

    label: str
    exact: bool
    same_tool_names: bool
    schema_valid: bool
    similarity: float
    actual_actions: tuple[ToolAction, ...] = ()
    expected_actions: tuple[ToolAction, ...] = ()
    schema_checks: tuple[SchemaCheck, ...] = ()
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "exact": self.exact,
            "same_tool_names": self.same_tool_names,
            "schema_valid": self.schema_valid,
            "similarity": self.similarity,
            "actual_actions": [action.to_dict() for action in self.actual_actions],
            "expected_actions": [action.to_dict() for action in self.expected_actions],
            "schema_checks": [check.to_dict() for check in self.schema_checks],
            "reason": self.reason,
            "metadata": copy.deepcopy(self.metadata),
        }


def parse_assistant_actions(
    message: dict[str, Any] | str,
    *,
    expected_action_names: set[str] | frozenset[str] | list[str] | tuple[str, ...] | None = None,
) -> tuple[ToolAction, ...]:
    """Parse model output into canonical tool actions without changing old env parsers."""

    return tuple(
        ToolAction.from_raw(action)
        for action in actions_from_assistant_message(
            message,
            expected_action_names=expected_action_names,
        )
    )


def match_step_action(
    step: TrajectoryStep,
    assistant_message: dict[str, Any] | str,
    *,
    tools: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    equivalent_threshold: float = 0.78,
    external_action_catalog: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> ActionMatchResult:
    """Match one assistant response against a parsed trajectory step.

    This is intentionally policy- and benchmark-free. Dataset integrations
    may add declarative metadata around this neutral result.
    """

    expected_action_names = {action.name.strip().lower() for action in step.expected_actions}
    actual_actions = parse_assistant_actions(
        assistant_message,
        expected_action_names=expected_action_names,
    )
    if step.kind == "terminal_stop":
        actual_text = _assistant_text(assistant_message)
        verified_boundary = step.metadata.get("verified_terminal_stop") is True
        metadata = {
            "step_index": step.index,
            "step_kind": step.kind,
            "terminal_stop_decision": True,
            "terminal_stop_boundary_verified": verified_boundary,
            "terminal_stop_text_nonempty": bool(actual_text.strip()),
            "terminal_stop_tool_call_count": len(actual_actions),
            # A successful recorded episode boundary certifies when to stop,
            # not whether arbitrary response wording answers the task.
            "terminal_stop_semantics_verified": False,
            "terminal_stop_provenance": str(
                step.metadata.get("terminal_stop_provenance") or ""
            ),
            "terminal_stop_synthetic": bool(
                step.metadata.get("terminal_stop_synthetic", False)
            ),
        }
        if actual_actions:
            return _result(
                "spurious_tool_call",
                actual_actions,
                (),
                (),
                0.0,
                False,
                False,
                True,
                metadata,
                reason="verified episode boundary expects a stop decision, not another tool call",
            )
        if verified_boundary and actual_text.strip():
            return _result(
                "terminal_stop",
                (),
                (),
                (),
                1.0,
                False,
                True,
                True,
                metadata,
                reason=(
                    "non-empty assistant response stops at a recorded successful "
                    "episode boundary; response semantics are not verified"
                ),
            )
        return _result(
            "unverified_text",
            (),
            (),
            (),
            0.0,
            False,
            True,
            True,
            metadata,
            reason=(
                "terminal stop requires an explicit successful-boundary "
                "certificate and a non-empty assistant response"
            ),
        )
    if step.expected_actions and not actual_actions and _assistant_text(assistant_message).strip():
        return _result(
            "unverified_text",
            (),
            step.expected_actions,
            (),
            0.0,
            False,
            False,
            True,
            {"step_index": step.index, "step_kind": step.kind},
            reason="assistant text has no trajectory-backed environment response",
        )
    expected_text = step.metadata.get("expected_assistant_message")
    if not step.expected_actions and isinstance(expected_text, str):
        actual_text = _assistant_text(assistant_message)
        metadata = {"step_index": step.index, "step_kind": step.kind}
        if not actual_actions and step.kind == "external_request":
            grounding = ground_external_request(
                actual_text,
                step.metadata.get("external_actions") or [],
                candidate_actions=external_action_catalog,
                verb_aliases=step.metadata.get(
                    "external_action_verb_aliases"
                )
                or {},
            )
            metadata["external_request_grounding"] = grounding.to_dict()
            if grounding.grounded:
                return _result(
                    "external_request_grounded", (), (), (), 1.0, False, True, True, metadata,
                    reason=grounding.reason,
                )
        if not actual_actions and _normalized_text(actual_text) == _normalized_text(expected_text):
            return _result(
                "gold_text", (), (), (), 1.0, True, True, True, metadata,
                reason="assistant text exactly matches recorded trajectory evidence",
            )
        if actual_actions:
            return _result(
                "spurious_tool_call", actual_actions, (), (), 0.0, False, False, True, metadata,
                reason="trajectory step expects an evidence-backed assistant text response",
            )
        return _result(
            "unverified_text", (), (), (), 0.0, False, True, True, metadata,
            reason=(
                metadata.get("external_request_grounding", {}).get("reason")
                or "assistant text is not supported by the recorded trajectory evidence"
            ),
        )
    return match_actions(
        actual_actions=actual_actions,
        expected_actions=step.expected_actions,
        tools=tools,
        equivalent_threshold=equivalent_threshold,
        metadata={"step_index": step.index, "step_kind": step.kind},
    )


def _assistant_text(message: dict[str, Any] | str) -> str:
    if isinstance(message, str):
        return message
    content = message.get("content")
    return str(content or "")


def _normalized_text(value: str) -> str:
    # Formatting-only normalization is evidence preserving.  Semantic or
    # fuzzy text equivalence would require a separate verifier and is not
    # inferred from a static trajectory.
    return " ".join(value.split())


def match_actions(
    *,
    actual_actions: tuple[ToolAction, ...] | list[ToolAction],
    expected_actions: tuple[ToolAction, ...] | list[ToolAction],
    tools: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    equivalent_threshold: float = 0.78,
    metadata: dict[str, Any] | None = None,
) -> ActionMatchResult:
    actual = tuple(actual_actions)
    expected = tuple(expected_actions)
    actual_dicts = _actions_to_dicts(actual)
    expected_dicts = _actions_to_dicts(expected)
    schema_checks = tuple(check_action_schema(action, tools) for action in actual)
    schema_valid = all(check.valid for check in schema_checks) if schema_checks else True

    if not expected and not actual:
        return _result("gold_text", actual, expected, schema_checks, 1.0, True, True, schema_valid, metadata)
    if not expected and actual:
        return _result(
            "spurious_tool_call",
            actual,
            expected,
            schema_checks,
            0.0,
            False,
            False,
            schema_valid,
            metadata,
            reason="trajectory step does not expect a tool call",
        )
    if expected and not actual:
        # A substantive assistant message is not a malformed tool call.  The
        # partial environment has no evidence for the corresponding user
        # response, so it must abstain and leave the transition cursor in
        # place.  Reserve ``missing_tool_call`` for genuinely empty/parser-
        # failed actions; conflating ordinary conversation with an invalid
        # action makes long dialogue trajectories spuriously fail.
        return _result(
            "missing_tool_call",
            actual,
            expected,
            schema_checks,
            0.0,
            False,
            False,
            schema_valid,
            metadata,
            reason="trajectory step expects a tool call",
        )

    exact = same_actions(actual_dicts, expected_dicts)
    names_match = same_tool_names(actual_dicts, expected_dicts)
    similarity = argument_similarity(actual_dicts, expected_dicts)

    if exact:
        return _result("gold", actual, expected, schema_checks, similarity, True, True, schema_valid, metadata)
    known_tools = _tool_schemas(tools)
    if known_tools and any(action.name not in known_tools for action in actual):
        return _result(
            "unknown_tool",
            actual,
            expected,
            schema_checks,
            similarity,
            False,
            False,
            False,
            metadata,
            reason="one or more tools are not in the available schema",
        )
    if not schema_valid:
        return _result(
            "schema_invalid",
            actual,
            expected,
            schema_checks,
            similarity,
            False,
            names_match,
            False,
            metadata,
            reason="; ".join(check.reason for check in schema_checks if not check.valid),
        )
    # ``similarity`` is diagnostic only.  A fuzzy threshold is not evidence
    # that two identifiers denote the same external object (for example, two
    # order ids that differ by one digit can have >0.9 string similarity).
    # Equivalent replay is therefore restricted to deterministic,
    # schema-guided canonicalization.
    del equivalent_threshold
    if names_match and controlled_actions_equivalent(actual, expected, tools=tools):
        return _result(
            "equivalent_args",
            actual,
            expected,
            schema_checks,
            similarity,
            False,
            True,
            True,
            metadata,
            reason="arguments match after controlled schema-guided normalization",
        )
    if names_match:
        return _result(
            "wrong_args",
            actual,
            expected,
            schema_checks,
            similarity,
            False,
            True,
            True,
            metadata,
            reason="tool names match but arguments do not match the trajectory",
        )
    return _result(
        "wrong_tool",
        actual,
        expected,
        schema_checks,
        similarity,
        False,
        False,
        schema_valid,
        metadata,
        reason="tool names do not match the trajectory",
    )


def controlled_actions_equivalent(
    actual_actions: tuple[ToolAction, ...] | list[ToolAction],
    expected_actions: tuple[ToolAction, ...] | list[ToolAction],
    *,
    tools: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> bool:
    """Return equivalence only for explicit, deterministic normalizations.

    The generic rules are deliberately small: JSON object order is ignored,
    insignificant surrounding/repeated whitespace is normalized, schema
    defaults are materialized, and a schema may declare an explicit
    ``x-trajectory2gym-aliases`` mapping for a string field.  A schema may
    also explicitly mark an argument with
    ``x-trajectory2gym-equivalence: presence-only`` when its value is
    descriptive payload that cannot change the tool's transition.  No
    edit-distance or inferred semantic-similarity rule can make an action
    executable.
    """

    actual = tuple(actual_actions)
    expected = tuple(expected_actions)
    if len(actual) != len(expected) or not actual:
        return False
    schemas = _tool_schemas(tools)
    for actual_action, expected_action in zip(actual, expected):
        if actual_action.name != expected_action.name:
            return False
        schema = schemas.get(actual_action.name, {})
        actual_arguments = _normalize_equivalent_value(actual_action.arguments, schema)
        expected_arguments = _normalize_equivalent_value(expected_action.arguments, schema)
        if actual_arguments != expected_arguments:
            return False
    return True


def _normalize_equivalent_value(value: Any, schema: dict[str, Any] | None = None) -> Any:
    schema = schema if isinstance(schema, dict) else {}
    if schema.get("x-trajectory2gym-equivalence") == "presence-only":
        # Schema validation runs before equivalence, so the value must still
        # have the declared type and required fields must still be present.
        # Only an explicit adapter declaration can make the payload itself
        # non-identifying.
        return {"__trajectory2gym_presence_only__": True}
    if isinstance(value, dict):
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        normalized = {
            str(key): _normalize_equivalent_value(child, properties.get(key))
            for key, child in value.items()
        }
        for key, property_schema in properties.items():
            if key not in normalized and isinstance(property_schema, dict) and "default" in property_schema:
                normalized[str(key)] = _normalize_equivalent_value(
                    copy.deepcopy(property_schema["default"]), property_schema
                )
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, list):
        item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else {}
        return [_normalize_equivalent_value(item, item_schema) for item in value]
    if isinstance(value, str):
        normalized = " ".join(value.strip().split())
        aliases = schema.get("x-trajectory2gym-aliases")
        if isinstance(aliases, dict):
            normalized = aliases.get(normalized, normalized)
        return normalized
    return value


def check_action_schema(
    action: ToolAction,
    tools: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> SchemaCheck:
    schemas = _tool_schemas(tools)
    if not schemas:
        return SchemaCheck(tool_name=action.name, valid=True)
    schema = schemas.get(action.name)
    if schema is None:
        return SchemaCheck(tool_name=action.name, valid=False, reason="unknown_tool")
    missing = set(schema.get("required", [])) - set(action.arguments)
    if missing:
        return SchemaCheck(tool_name=action.name, valid=False, reason=f"missing_required_args: {sorted(missing)}")
    properties = schema.get("properties", {}) or {}
    if not schema.get("additionalProperties", False):
        extra = set(action.arguments) - set(properties)
        if extra:
            return SchemaCheck(tool_name=action.name, valid=False, reason=f"undefined_args: {sorted(extra)}")
    for key, value in action.arguments.items():
        prop = properties.get(key)
        if not isinstance(prop, dict):
            continue
        ok, reason = _check_json_type(value, prop, key)
        if not ok:
            return SchemaCheck(tool_name=action.name, valid=False, reason=reason)
    return SchemaCheck(tool_name=action.name, valid=True)


def _result(
    label: str,
    actual: tuple[ToolAction, ...],
    expected: tuple[ToolAction, ...],
    schema_checks: tuple[SchemaCheck, ...],
    similarity: float,
    exact: bool,
    names_match: bool,
    schema_valid: bool,
    metadata: dict[str, Any] | None,
    *,
    reason: str = "",
) -> ActionMatchResult:
    return ActionMatchResult(
        label=label,
        exact=exact,
        same_tool_names=names_match,
        schema_valid=schema_valid,
        similarity=similarity,
        actual_actions=actual,
        expected_actions=expected,
        schema_checks=schema_checks,
        reason=reason,
        metadata=copy.deepcopy(metadata or {}),
    )


def _actions_to_dicts(actions: tuple[ToolAction, ...]) -> list[dict[str, Any]]:
    return [action.to_dict() for action in actions]


def _tool_schemas(tools: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    schemas = {}
    for tool in tools:
        name = tool_name_from_schema(tool)
        if not name:
            continue
        schemas[name] = tool_parameters_from_schema(tool)
    return schemas


def _check_json_type(value: Any, schema: dict[str, Any], path: str) -> tuple[bool, str]:
    type_name = schema.get("type")
    if isinstance(type_name, list):
        return any(_check_json_type(value, {**schema, "type": child}, path)[0] for child in type_name), (
            f"type_error_at_{path}: expected one of {type_name}"
        )
    if type_name == "string" and not isinstance(value, str):
        return False, f"type_error_at_{path}: expected string"
    if type_name == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        return False, f"type_error_at_{path}: expected integer"
    if type_name == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        return False, f"type_error_at_{path}: expected number"
    if type_name == "boolean" and not isinstance(value, bool):
        return False, f"type_error_at_{path}: expected boolean"
    if type_name == "array":
        if not isinstance(value, list):
            return False, f"type_error_at_{path}: expected array"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                ok, reason = _check_json_type(item, item_schema, f"{path}[{index}]")
                if not ok:
                    return False, reason
    if type_name == "object":
        if not isinstance(value, dict):
            return False, f"type_error_at_{path}: expected object"
        properties = schema.get("properties", {}) or {}
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            return False, f"missing_nested_args_at_{path}: {sorted(missing)}"
        if not schema.get("additionalProperties", False):
            extra = set(value) - set(properties)
            if extra:
                return False, f"undefined_nested_arg_at_{path}: {sorted(extra)}"
        for key, child in value.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, dict):
                ok, reason = _check_json_type(child, child_schema, f"{path}.{key}")
                if not ok:
                    return False, reason
    if "enum" in schema and value not in schema["enum"]:
        return False, f"enum_error_at_{path}: expected one of {schema['enum']}"
    return True, "type_valid"

from __future__ import annotations

import ast
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


INVALID_ARGUMENTS_KEY = "__invalid_arguments__"
_REASONING_ONLY_ACTION_NAMES = frozenset({"think"})


def load_tool_call_traces(
    path: str | Path, *, max_entries: int | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_entries is not None and len(rows) >= max_entries:
                break
    return rows


def tool_name_from_schema(tool: dict[str, Any]) -> str | None:
    function = tool.get("function", tool)
    name = function.get("name")
    return str(name) if name else None


def tool_parameters_from_schema(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function", tool)
    return function.get("parameters") or {}


def schema_argument_order(tools: list[dict[str, Any]]) -> dict[str, list[str]]:
    order: dict[str, list[str]] = {}
    for tool in tools:
        name = tool_name_from_schema(tool)
        if not name:
            continue
        parameters = tool_parameters_from_schema(tool)
        properties = list((parameters.get("properties") or {}).keys())
        required = list(parameters.get("required") or [])
        order[name] = required + [key for key in properties if key not in required]
    return order


def parse_function_call_string(call: str, argument_order: dict[str, list[str]]) -> dict[str, Any]:
    node = ast.parse(call.strip(), mode="eval").body
    if not isinstance(node, ast.Call):
        raise ValueError(f"expected function call, got: {call}")
    name = _ast_call_name(node.func)
    if not name:
        raise ValueError(f"could not parse function name from: {call}")
    args: dict[str, Any] = {}
    positional_order = argument_order.get(name, [])
    for index, value_node in enumerate(node.args):
        key = positional_order[index] if index < len(positional_order) else f"arg{index}"
        args[key] = ast.literal_eval(value_node)
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ValueError(f"unsupported **kwargs in: {call}")
        args[keyword.arg] = ast.literal_eval(keyword.value)
    return {"name": name, "arguments": args}


def action_to_openai_tool_call(action: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": f"{action['name']}_{index}",
        "type": "function",
        "function": {
            "name": action["name"],
            "arguments": action.get("arguments", {}),
        },
    }


def openai_tool_calls_to_actions(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for item in tool_calls:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not name:
            continue
        actions.append({"name": str(name), "arguments": normalize_arguments_or_invalid(function.get("arguments", {}))})
    return actions


def actions_from_assistant_message(
    message: dict[str, Any] | str,
    *,
    expected_action_names: set[str] | frozenset[str] | list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Extract environment actions from one assistant message.

    ``think`` needs a small amount of context.  Qwen-style private reasoning
    can be decoded by a serving parser as a native ``think`` call, while some
    trajectory datasets legitimately contain a declared ``think`` tool as a
    real transition.  Callers that know the current expected step should pass
    its action names; an expected ``think`` then keeps the declared action.
    Parser-only callers retain the conservative default and drop reasoning
    calls from the environment surface.
    """
    expected_names = (
        {
            str(name).strip().lower()
            for name in expected_action_names
            if str(name).strip()
        }
        if expected_action_names is not None
        else None
    )
    if isinstance(message, str):
        content = message
        tool_calls = []
        reasoning_content = ""
    else:
        content = str(message.get("content") or "")
        tool_calls = message.get("tool_calls") or []
        reasoning_content = str(
            message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("thinking_content")
            or ""
        )
    if isinstance(tool_calls, list) and tool_calls:
        parsed = _drop_reasoning_only_actions(
            openai_tool_calls_to_actions(tool_calls),
            expected_action_names=expected_names,
            content=content,
            reasoning_content=reasoning_content,
        )
        if parsed:
            return parsed

    actions: list[dict[str, Any]] = []
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL):
        raw = match.group(1).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            try:
                payload = ast.literal_eval(raw)
            except Exception:
                continue
        if isinstance(payload, list):
            for item in payload:
                action = action_from_payload(item)
                if action:
                    actions.append(action)
        else:
            action = action_from_payload(payload)
            if action:
                actions.append(action)
    if actions:
        return _drop_reasoning_only_actions(
            actions,
            expected_action_names=expected_names,
            content=content,
            reasoning_content=reasoning_content,
        )

    return _drop_reasoning_only_actions(
        actions_from_bare_json_content(content),
        expected_action_names=expected_names,
        content=content,
        reasoning_content=reasoning_content,
    )


def _drop_reasoning_only_actions(
    actions: list[dict[str, Any]],
    *,
    expected_action_names: set[str] | frozenset[str] | None = None,
    content: str = "",
    reasoning_content: str = "",
) -> list[dict[str, Any]]:
    """Remove only private reasoning calls from the environment action surface.

    Qwen-style thinking can arrive either as ``<think>`` text or as a native
    tool call named ``think`` after a serving tool parser has decoded the
    response.  A trajectory may nevertheless declare ``think`` as a real
    action.  Expected-step context therefore takes precedence; without it,
    parser-only callers use the conservative private-reasoning default.
    """

    if "think" in (expected_action_names or ()):
        return actions

    # A parser-only caller has no expected-step context. Preserve an ordinary
    # declared ``think`` tool unless the message explicitly carries a private
    # reasoning marker; the trajectory matcher supplies an empty expected-name
    # set for text/tool steps and therefore still filters it conservatively.
    private_marker = bool(
        re.search(r"<think\b|</think\s*>", content, flags=re.IGNORECASE)
        or str(reasoning_content or "").strip()
    )
    if expected_action_names is None and not private_marker:
        return actions

    return [
        action
        for action in actions
        if str(action.get("name") or "").strip().lower() not in _REASONING_ONLY_ACTION_NAMES
    ]


def actions_from_bare_json_content(content: str) -> list[dict[str, Any]]:
    """Parse tool calls emitted as bare JSON rather than XML-wrapped calls."""
    visible = re.sub(r"<think>.*?</think>", " ", content, flags=re.DOTALL | re.IGNORECASE)
    visible = re.sub(r"^\s*assistant\s*", "", visible, flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    actions: list[dict[str, Any]] = []
    index = 0
    while index < len(visible):
        char = visible[index]
        if char not in "[{":
            index += 1
            continue
        try:
            payload, end = decoder.raw_decode(visible[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        actions.extend(actions_from_payload_any(payload))
        index += max(end, 1)
    return actions


def actions_from_payload_any(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("tool_call"), dict):
        action = action_from_payload(payload["tool_call"])
        return [action] if action else []
    if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
        actions = []
        for item in payload["tool_calls"]:
            action = action_from_payload(item)
            if action:
                actions.append(action)
        return actions
    if isinstance(payload, list):
        actions = []
        for item in payload:
            actions.extend(actions_from_payload_any(item))
        return actions
    action = action_from_payload(payload)
    return [action] if action else []


def action_from_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("tool_call"), dict):
        return action_from_payload(payload["tool_call"])
    if isinstance(payload.get("function"), dict):
        function = payload["function"]
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = payload.get("name") or payload.get("function")
        arguments = payload.get("arguments", payload.get("parameters", {}))
    if not name:
        return None
    return {"name": str(name), "arguments": normalize_arguments_or_invalid(arguments)}


def normalize_arguments_or_invalid(arguments: Any) -> dict[str, Any]:
    try:
        return normalize_arguments(arguments)
    except Exception as exc:
        return {INVALID_ARGUMENTS_KEY: f"{type(exc).__name__}: {exc}"}


def normalize_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(stripped)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError(f"tool arguments must be dict-like, got {type(arguments).__name__}")


def canonical_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"name": str(action["name"]), "arguments": normalize_arguments(action.get("arguments", {}))}
        for action in actions
    ]


def action_signature(actions: list[dict[str, Any]]) -> str:
    return json.dumps(canonical_actions(actions), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def same_actions(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    return action_signature(left) == action_signature(right)


def same_tool_names(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    return [item["name"] for item in canonical_actions(left)] == [item["name"] for item in canonical_actions(right)]


def argument_similarity(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    if len(left) != len(right):
        return 0.0
    if not left:
        return 1.0
    pairs = zip(canonical_actions(left), canonical_actions(right))
    return sum(_value_similarity(a["arguments"], b["arguments"]) for a, b in pairs) / len(left)


def _value_similarity(left: Any, right: Any) -> float:
    if left == right:
        return 1.0
    if isinstance(left, str) and isinstance(right, str):
        return SequenceMatcher(None, "".join(left.split()).lower(), "".join(right.split()).lower()).ratio()
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) | set(right)
        if not keys:
            return 1.0
        return sum(_value_similarity(left.get(key), right.get(key)) for key in keys) / len(keys)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return 0.0
        if not left:
            return 1.0
        return sum(_value_similarity(lv, rv) for lv, rv in zip(left, right)) / len(left)
    return 0.0


def _ast_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _ast_call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None

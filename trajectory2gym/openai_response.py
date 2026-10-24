"""Small helpers for normalizing OpenAI-compatible model responses."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib import error


CONTEXT_LENGTH_ERROR_RE = re.compile(
    r"maximum context length is\s+(?P<context>\d+)\s+tokens.*?"
    r"requested\s+(?P<requested>\d+)\s+tokens\s+"
    r"\((?P<messages>\d+)\s+in the messages,\s+"
    r"(?P<completion>\d+)\s+in the completion\)",
    re.IGNORECASE | re.DOTALL,
)


def _as_openai_tool_call(
    payload: dict[str, Any],
    index: int,
) -> dict[str, Any] | None:
    if "function" in payload and isinstance(payload["function"], dict):
        function = payload["function"]
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = payload.get("name")
        arguments = payload.get("arguments", {})
    if not name:
        return None
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": payload.get("id") or f"parsed_tool_call_{index}",
        "type": payload.get("type") or "function",
        "function": {"name": name, "arguments": arguments},
    }


def _tool_calls_from_json(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(
        payload.get("tool_calls"), list
    ):
        values = payload["tool_calls"]
    elif isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = [payload]
    else:
        return []
    return [
        tool_call
        for index, item in enumerate(values)
        if isinstance(item, dict)
        for tool_call in [_as_openai_tool_call(item, index)]
        if tool_call is not None
    ]


def extract_tool_calls_from_content(
    content: str | None,
) -> list[dict[str, Any]]:
    """Recover tool calls emitted as JSON text by compatible endpoints."""

    if not content:
        return []
    tool_calls: list[dict[str, Any]] = []
    for match in re.finditer(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        content,
        re.DOTALL,
    ):
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        tool_calls.extend(_tool_calls_from_json(parsed))
    if tool_calls:
        return tool_calls
    stripped = content.strip()
    if not stripped or stripped[0] not in "[{":
        return []
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return []
    return _tool_calls_from_json(parsed)


def http_error_body(exc: error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return str(exc)


def context_length_retry_max_tokens(
    error_body: str,
    current_max_tokens: int,
) -> int | None:
    match = CONTEXT_LENGTH_ERROR_RE.search(error_body)
    if not match:
        return None
    context_tokens = int(match.group("context"))
    message_tokens = int(match.group("messages"))
    retry_max_tokens = context_tokens - message_tokens - 32
    if retry_max_tokens < 64 or retry_max_tokens >= current_max_tokens:
        return None
    return retry_max_tokens

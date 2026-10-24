from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolSemantics:
    """Small declarative mapping from tool names to external-state semantics.

    The core owns the semantic categories and matching algorithm. Benchmark
    adapters may replace these tuples, but should not implement workflow or
    reward logic.
    """

    read_only_prefixes: tuple[str, ...] = (
        "calculate",
        "check",
        "compute",
        "describe",
        "fetch",
        "find",
        "get",
        "list",
        "lookup",
        "query",
        "read",
        "retrieve",
        "search",
        "show",
        "validate",
        "view",
    )
    side_effect_tokens: tuple[str, ...] = (
        "add",
        "activate",
        "book",
        "buy",
        "cancel",
        "create",
        "deactivate",
        "delete",
        "exchange",
        "fill",
        "engage",
        "login",
        "logout",
        "lock",
        "modify",
        "move",
        "order",
        "place",
        "post",
        "purchase",
        "register",
        "remove",
        "rename",
        "reserve",
        "resolve",
        "return",
        "schedule",
        "send",
        "set",
        "sort",
        "start",
        "stop",
        "submit",
        "transfer",
        "unlock",
        "update",
        "write",
    )
    side_effect_exact_names: tuple[str, ...] = (
        "cd",
        "cp",
        "echo",
        "mkdir",
        "mv",
        "rm",
        "rmdir",
        "touch",
    )
    observation_db_safe_exact_names: tuple[str, ...] | None = None
    allow_name_inferred_observation_db_replay: bool = False

    @staticmethod
    def normalize(name: str) -> str:
        return str(name).rsplit(".", 1)[-1].strip().lower()

    def is_read_only(self, name: str) -> bool:
        normalized = self.normalize(name)
        return normalized.startswith(self.read_only_prefixes)

    def is_side_effecting(self, name: str) -> bool:
        normalized = self.normalize(name)
        # Explicit adapter declarations are stronger than name morphology.
        # Writes win on contradiction; otherwise an explicitly safe read such
        # as ``can_send_mms`` must not become a write merely because its name
        # contains the token ``send``.
        if normalized in self.side_effect_exact_names:
            return True
        if (
            self.observation_db_safe_exact_names is not None
            and normalized in self.observation_db_safe_exact_names
        ):
            return False
        if self.is_read_only(normalized):
            return False
        return any(token in normalized for token in self.side_effect_tokens)

    def allows_observation_db_replay(self, name: str) -> bool:
        """Whether cross-trajectory observations may be reused for this tool."""

        normalized = self.normalize(name)
        if self.is_side_effecting(normalized):
            return False
        if self.observation_db_safe_exact_names is not None:
            return normalized in self.observation_db_safe_exact_names
        # A name such as ``get_order_status`` does not prove that the result
        # is context independent. Prefixes may suggest adapter declarations,
        # but they are not execution evidence in core Partial.
        return bool(
            self.allow_name_inferred_observation_db_replay
            and self.is_read_only(normalized)
        )


DEFAULT_TOOL_SEMANTICS = ToolSemantics()


def tool_semantics_from_mapping(
    value: dict[str, Any], *, default: ToolSemantics = DEFAULT_TOOL_SEMANTICS
) -> ToolSemantics:
    """Compile the small declarative adapter format into core semantics."""

    def names(key: str) -> tuple[str, ...]:
        raw = value.get(key, ())
        if not isinstance(raw, (list, tuple)):
            raise ValueError(f"tool_semantics.{key} must be a list")
        return tuple(
            sorted({ToolSemantics.normalize(item) for item in raw if str(item).strip()})
        )

    safe = names("observation_db_safe_tools")
    mutating = names("mutating_tools")
    return ToolSemantics(
        read_only_prefixes=default.read_only_prefixes,
        side_effect_tokens=default.side_effect_tokens,
        side_effect_exact_names=tuple(
            sorted(set(default.side_effect_exact_names) | set(mutating))
        ),
        observation_db_safe_exact_names=safe,
        allow_name_inferred_observation_db_replay=False,
    )


def load_tool_semantics_file(
    path: str | Path, *, default: ToolSemantics = DEFAULT_TOOL_SEMANTICS
) -> ToolSemantics:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("tool semantics file must contain a JSON object")
    return tool_semantics_from_mapping(value, default=default)

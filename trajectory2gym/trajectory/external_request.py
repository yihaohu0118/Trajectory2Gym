from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Any

from trajectory2gym.text import strip_visible_thinking


_WORD_RE = re.compile(r"[a-z0-9]+")
_SEGMENT_RE = re.compile(r"[.!?;\n]+")
_GENERIC_NAME_WORDS = {
    "action", "app", "device", "function", "preference", "setting",
    "settings", "status", "user",
}
_VERB_FAMILIES = {
    "can": {"can", "check", "send", "test", "try", "verify"},
    "check": {"check", "inspect", "look", "see", "show", "verify"},
    "disable": {"disable", "off", "stop", "turn"},
    "disconnect": {"disconnect", "off", "stop", "turn"},
    "enable": {"enable", "on", "start", "turn"},
    "get": {"check", "find", "get", "look", "retrieve", "show"},
    "grant": {"allow", "enable", "give", "grant", "permission"},
    "reboot": {"reboot", "restart"},
    "remove": {"delete", "remove"},
    "reset": {"restore", "reset"},
    "run": {"check", "run", "test", "try"},
    "set": {"change", "choose", "select", "set", "switch"},
    "toggle": {"change", "disable", "enable", "off", "on", "switch", "toggle", "turn"},
}
_REQUEST_CUES = {
    "allow", "ask", "change", "check", "choose", "could", "disconnect",
    "disable", "enable", "go", "grant", "let", "need", "perform", "please",
    "reboot", "reinsert", "remove", "reseat", "reset", "restart", "run", "select",
    "set", "switch", "toggle", "try", "turn", "use", "would",
}


@dataclass(frozen=True)
class ExternalRequestGrounding:
    grounded: bool
    reason: str
    matched_actions: int = 0
    expected_actions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounded": self.grounded,
            "reason": self.reason,
            "matched_actions": self.matched_actions,
            "expected_actions": self.expected_actions,
        }


def ground_external_request(
    text: str,
    expected_actions: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    candidate_actions: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    verb_aliases: dict[str, list[str] | tuple[str, ...] | set[str]] | None = None,
) -> ExternalRequestGrounding:
    """Ground text to recorded external actions without predicting effects."""

    actions = [action for action in expected_actions if isinstance(action, dict)]
    if not actions:
        return ExternalRequestGrounding(False, "external request has no recorded action evidence")
    # Private model reasoning is neither shown to the user nor executable by
    # the environment. Models often enumerate several rejected alternatives
    # inside <think>; treating those tokens as user instructions creates false
    # multi-action conflicts and trains against sound reasoning.
    visible_text = strip_visible_thinking(text)
    segments = [set(_word_list(segment)) for segment in _SEGMENT_RE.split(visible_text)]
    segments = [segment for segment in segments if segment]
    if not segments:
        return ExternalRequestGrounding(
            False,
            "assistant response contains no groundable request text",
            expected_actions=len(actions),
        )
    matched = 0
    reasons: list[str] = []
    for action in actions:
        attempts = [
            _grounds_action(segment, action, verb_aliases=verb_aliases)
            for segment in segments
            if _has_request_cue(segment)
        ]
        if any(ok for ok, _ in attempts):
            matched += 1
        else:
            reasons.append(
                next((reason for ok, reason in attempts if not ok), "no directive sentence grounds action")
            )
    if matched != len(actions):
        return ExternalRequestGrounding(
            False,
            "; ".join(reasons) or "not every recorded external action is grounded",
            matched_actions=matched,
            expected_actions=len(actions),
        )
    expected_signatures = {_action_signature(action) for action in actions}
    for candidate in candidate_actions:
        if not isinstance(candidate, dict) or _action_signature(candidate) in expected_signatures:
            continue
        if any(
            _grounds_action(
                segment, candidate, verb_aliases=verb_aliases
            )[0]
            for segment in segments
            if _has_request_cue(segment)
        ):
            return ExternalRequestGrounding(
                False,
                "assistant text also grounds a different catalogued external action",
                matched_actions=matched,
                expected_actions=len(actions),
            )
    return ExternalRequestGrounding(
        True,
        "assistant text grounds every recorded external action",
        matched_actions=matched,
        expected_actions=len(actions),
    )


def _grounds_action(
    text_tokens: set[str],
    action: dict[str, Any],
    *,
    verb_aliases: dict[str, list[str] | tuple[str, ...] | set[str]] | None = None,
) -> tuple[bool, str]:
    name = str(action.get("name") or "").strip().lower()
    name_words = _word_list(name)
    if not name_words:
        return False, "recorded external action has no name"
    operation = name_words[0]
    content_words = [word for word in name_words[1:] if word not in _GENERIC_NAME_WORDS]
    if not content_words:
        content_words = name_words[1:] or name_words
    declared_aliases = (verb_aliases or {}).get(operation, ())
    verb_family = set(_VERB_FAMILIES.get(operation, {operation})) | {
        str(alias).lower() for alias in declared_aliases
    }
    if not any(_token_matches(candidate, token) for candidate in verb_family for token in text_tokens):
        return False, f"request does not ground operation {operation!r}"
    missing_content = [
        word for word in content_words if not any(_token_matches(word, token) for token in text_tokens)
    ]
    if missing_content:
        return False, f"request omits action object tokens {missing_content}"
    arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
    for key, value in arguments.items():
        if value is None or isinstance(value, (dict, list)):
            continue
        value_words = _word_list(str(value))
        if value_words and not all(
            any(_token_matches(word, token) for token in text_tokens) for word in value_words
        ):
            return False, f"request omits argument value for {key!r}"
    return True, "grounded"


def _word_list(value: str) -> list[str]:
    return _WORD_RE.findall(value.lower().replace("_", " ").replace("-", " "))


def _has_request_cue(tokens: set[str]) -> bool:
    return bool(tokens & _REQUEST_CUES)


def _token_matches(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    if len(expected) >= 5 and len(actual) >= 5:
        return expected[:5] == actual[:5]
    return False


def _action_signature(action: dict[str, Any]) -> str:
    return json.dumps(
        {
            "name": str(action.get("name") or ""),
            "arguments": action.get("arguments") if isinstance(action.get("arguments"), dict) else {},
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def external_action_catalog_from_episodes(episodes: Any) -> tuple[dict[str, Any], ...]:
    """Collect unique declared external actions from parsed episodes."""

    catalog: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        for step in getattr(episode, "steps", ()):
            metadata = getattr(step, "metadata", {})
            if not isinstance(metadata, dict):
                continue
            for action in metadata.get("external_actions") or []:
                if not isinstance(action, dict):
                    continue
                canonical = {
                    "name": str(action.get("name") or ""),
                    "arguments": (
                        action.get("arguments")
                        if isinstance(action.get("arguments"), dict)
                        else {}
                    ),
                }
                catalog[_action_signature(canonical)] = canonical
    return tuple(catalog[key] for key in sorted(catalog))

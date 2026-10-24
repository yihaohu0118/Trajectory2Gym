from __future__ import annotations

import copy
import ast
import json
from typing import Any

from trajectory2gym.tool_calls import parse_function_call_string, schema_argument_order

from .schema import ToolAction, TrajectoryEpisode, TrajectoryStep


TERMINAL_ACTION_NAMES = {"prepare_to_answer", "ask_user_for_required_parameters"}


def parse_wildtoolbench_episode(
    entry: dict[str, Any],
    *,
    task_idx: int,
    language: str = "en",
    use_conversation_history: bool = True,
) -> TrajectoryEpisode:
    """Convert one WildToolBench task to the neutral trajectory2gym schema."""

    tools, tasks, answer_lists = _wildtoolbench_language_fields(entry, language)
    task = str(tasks[task_idx])
    answer_list = answer_lists[task_idx]
    steps = tuple(_wildtoolbench_steps(answer_list))
    return TrajectoryEpisode(
        id=f"{entry.get('id', 'wildtoolbench')}:{task_idx}",
        source="wildtoolbench",
        domain=str(entry.get("env_info") or entry.get("english_env_info") or ""),
        task=task,
        tools=tuple(copy.deepcopy(tools)),
        initial_messages=tuple(
            _wildtoolbench_initial_messages(
                entry,
                tasks,
                task_idx,
                language=language,
                use_conversation_history=use_conversation_history,
            )
        ),
        steps=steps,
        metadata={
            "source_id": entry.get("id"),
            "task_idx": task_idx,
            "language": language,
            "task_type": _list_get(entry.get("english_task_types" if language == "en" else "task_types"), task_idx),
            "turn_subtype": _list_get(
                entry.get("english_turn_subtypes" if language == "en" else "turn_subtypes"),
                task_idx,
            ),
            # Generic compiler declaration, intentionally independent of the
            # WildToolBench format.  Dataset converters may place the same
            # field at episode level without implementing an adapter class.
            "observation_db_safe_tools": copy.deepcopy(
                entry.get("observation_db_safe_tools", [])
            ),
        },
    )


def parse_tau_episode(entry: dict[str, Any], *, source: str = "tau_bench") -> TrajectoryEpisode:
    """Convert a tau/tau2 compiled trace entry to the neutral schema."""

    steps = []
    for fallback_index, raw_step in enumerate(entry.get("steps", [])):
        if not isinstance(raw_step, dict):
            continue
        step_index = int(raw_step.get("step_index", fallback_index))
        metadata = copy.deepcopy(raw_step.get("metadata", {}))
        if not isinstance(metadata, dict):
            metadata = {}
        if raw_step.get("action_metadata") is not None:
            metadata["action_metadata"] = copy.deepcopy(raw_step.get("action_metadata"))
        expected_actions = tuple(
            ToolAction.from_raw(action)
            for action in raw_step.get("gold_actions", [])
            if isinstance(action, dict) and action.get("name")
        )
        steps.append(
            TrajectoryStep(
                index=step_index,
                expected_actions=expected_actions,
                observations=tuple(_message_dicts(raw_step.get("observations", []))),
                kind=_step_kind(expected_actions=expected_actions, metadata=metadata),
                metadata=metadata,
            )
        )

    return TrajectoryEpisode(
        id=str(entry.get("id") or ""),
        source=str(entry.get("source") or source),
        domain=entry.get("domain") or entry.get("category") or entry.get("ability"),
        task=str(entry.get("question") or entry.get("user_prompt") or ""),
        tools=tuple(copy.deepcopy(entry.get("tools", []))),
        initial_messages=tuple(_message_dicts(entry.get("prompt", []))),
        steps=tuple(steps),
        metadata=copy.deepcopy(entry.get("metadata", {})) if isinstance(entry.get("metadata"), dict) else {},
    )


def parse_tau2_episode(entry: dict[str, Any]) -> TrajectoryEpisode:
    """Convert a tau2 compiled trace entry to the neutral schema."""

    return parse_tau_episode(entry, source="tau2_bench")


def parse_apigenmt_episode(entry: dict[str, Any], *, row_index: int = 0) -> TrajectoryEpisode:
    """Convert one APIGen-MT row to the neutral trajectory2gym schema."""

    conversations = entry.get("conversations", [])
    if not isinstance(conversations, list):
        conversations = []
    tools = _apigen_tools(entry.get("tools", []))
    steps: list[TrajectoryStep] = []
    initial_messages: list[dict[str, Any]] = []
    step_index = 0
    index = 0
    while index < len(conversations):
        turn = conversations[index]
        if not isinstance(turn, dict):
            index += 1
            continue
        role = turn.get("from")
        value = turn.get("value", "")
        if role in {"human", "system", "gpt"} and not steps:
            mapped = _apigen_message(role, value)
            if mapped is not None:
                initial_messages.append(mapped)
        if role == "function_call":
            action = _json_object(value)
            observations: tuple[dict[str, Any], ...] = ()
            if index + 1 < len(conversations):
                next_turn = conversations[index + 1]
                if isinstance(next_turn, dict) and next_turn.get("from") == "observation":
                    observations = ({"role": "user", "content": copy.deepcopy(next_turn.get("value", ""))},)
                    index += 1
            expected_actions: tuple[ToolAction, ...] = ()
            if action and action.get("name") != "think":
                expected_actions = (ToolAction.from_raw(action),)
            steps.append(
                TrajectoryStep(
                    index=step_index,
                    expected_actions=expected_actions,
                    observations=observations,
                    kind="text" if not expected_actions else "tool",
                    metadata={"raw_role": "function_call", "raw_action": copy.deepcopy(action)},
                )
            )
            step_index += 1
        index += 1

    if not initial_messages:
        initial_messages = [{"role": "user", "content": ""}]
    domain = _apigen_domain(entry)
    return TrajectoryEpisode(
        id=str(entry.get("id") or f"apigen_mt:{row_index}"),
        source="apigen_mt",
        domain=domain,
        task=_first_user_text(conversations),
        tools=tuple(copy.deepcopy(tools)),
        initial_messages=tuple(initial_messages),
        steps=tuple(steps),
        metadata={"row_index": row_index, "system": entry.get("system", "")},
    )


def parse_bfcl_episode(entry: dict[str, Any]) -> TrajectoryEpisode:
    """Convert one compiled BFCL trajectory row to the neutral schema."""

    turns = entry.get("turns", [])
    if not isinstance(turns, list):
        turns = []
    steps: list[TrajectoryStep] = []
    initial_messages: list[dict[str, Any]] = []
    for fallback_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        user = turn.get("user")
        if isinstance(user, dict) and not initial_messages:
            initial_messages.append(copy.deepcopy(user))
        raw_actions = turn.get("gold_actions", [])
        if not isinstance(raw_actions, list):
            raw_actions = []
        expected_actions = tuple(
            ToolAction.from_raw(action) for action in raw_actions if isinstance(action, dict) and action.get("name")
        )
        step_index = int(turn.get("turn_index", fallback_index))
        observations = tuple(_message_dicts(turn.get("observations", [])))
        tools = turn.get("tools")
        metadata = {
            "gold_action_signature": turn.get("gold_action_signature"),
            "turn_index": step_index,
        }
        if isinstance(tools, list):
            metadata["tools"] = copy.deepcopy(tools)
        steps.append(
            TrajectoryStep(
                index=step_index,
                expected_actions=expected_actions,
                observations=observations,
                kind="tool" if expected_actions else "text",
                metadata=metadata,
            )
        )
    if not initial_messages:
        initial_messages = [{"role": "user", "content": ""}]
    return TrajectoryEpisode(
        id=str(entry.get("id") or ""),
        source="bfcl",
        domain=entry.get("category"),
        task=str(initial_messages[0].get("content") or ""),
        tools=tuple(copy.deepcopy(entry.get("tools", []))),
        initial_messages=tuple(initial_messages),
        steps=tuple(steps),
        metadata=copy.deepcopy(entry.get("metadata", {})) if isinstance(entry.get("metadata"), dict) else {},
    )


def parse_hardgen_episode(entry: dict[str, Any], *, row_index: int = 0) -> TrajectoryEpisode:
    """Convert one HardGen/FunReason-MT BFCL-style row to the neutral schema."""

    turns = [
        value
        for key, value in sorted(
            ((key, value) for key, value in entry.items() if str(key).isdigit()),
            key=lambda item: int(item[0]),
        )
        if isinstance(value, dict)
    ]
    system = turns[0] if turns else {}
    tools = _hardgen_tools(system.get("tools", []))
    argument_order = schema_argument_order(list(tools))
    first_user = next((turn for turn in turns if turn.get("role") == "user"), None)
    initial_messages: list[dict[str, Any]] = []
    if system.get("content"):
        initial_messages.append({"role": "system", "content": str(system.get("content"))})
    if first_user is not None:
        initial_messages.append({"role": "user", "content": str(first_user.get("content") or "")})
    if not initial_messages:
        initial_messages = [{"role": "user", "content": ""}]

    steps: list[TrajectoryStep] = []
    step_index = 0
    for turn_index, turn in enumerate(turns):
        if turn.get("role") != "tool":
            continue
        for call, observation in _hardgen_tool_observation_pairs(str(turn.get("content") or "")):
            try:
                action = parse_function_call_string(call, argument_order)
            except Exception:
                continue
            steps.append(
                TrajectoryStep(
                    index=step_index,
                    expected_actions=(ToolAction.from_raw(action),),
                    observations=({"role": "user", "content": observation},),
                    kind="tool",
                    metadata={
                        "turn_index": turn_index,
                        "raw_function_call": call,
                        "error_tool_response": bool(turn.get("error_tool_response", False)),
                    },
                )
            )
            step_index += 1

    return TrajectoryEpisode(
        id=str(entry.get("id") or f"hardgen:{row_index}"),
        source="hardgen",
        domain="bfcl_hardgen",
        task=str(first_user.get("content") if first_user else ""),
        tools=tools,
        initial_messages=tuple(initial_messages),
        steps=tuple(steps),
        metadata={"row_index": row_index, "dataset": "Bingguang/HardGen"},
    )


def _wildtoolbench_steps(answer_list: list[dict[str, Any]]) -> list[TrajectoryStep]:
    steps = []
    for fallback_index, raw_node in enumerate(answer_list):
        if not isinstance(raw_node, dict):
            continue
        action = raw_node.get("action", {})
        if not isinstance(action, dict):
            action = {}
        name = str(action.get("name") or "")
        expected_actions: tuple[ToolAction, ...] = ()
        kind = "text" if name in TERMINAL_ACTION_NAMES else "tool"
        observations: tuple[dict[str, Any], ...]
        metadata = {
            "dependency_list": copy.deepcopy(
                raw_node.get("dependency_list", [])
            ),
            "raw_action": copy.deepcopy(action),
            "user_input": copy.deepcopy(raw_node.get("user_input")),
        }
        if name and name not in TERMINAL_ACTION_NAMES:
            expected_actions = (ToolAction.from_raw(action),)
            observations = (
                {
                    "role": "tool",
                    "content": copy.deepcopy(raw_node.get("observation")),
                },
            )
        else:
            # WildToolBench stores the reference assistant utterance in the
            # terminal node's ``observation`` field.  It is not a tool
            # observation.  Keeping it as an expected assistant message makes
            # terminal acceptance verifiable and prevents arbitrary non-empty
            # text from being labelled successful.
            metadata["expected_assistant_message"] = copy.deepcopy(
                raw_node.get("observation")
            )
            user_input = raw_node.get("user_input")
            observations = (
                ({"role": "user", "content": copy.deepcopy(user_input)},)
                if name == "ask_user_for_required_parameters"
                and user_input is not None
                else ()
            )
        steps.append(
            TrajectoryStep(
                index=int(raw_node.get("idx", fallback_index)),
                expected_actions=expected_actions,
                observations=observations,
                kind=kind,
                metadata=metadata,
            )
        )
    return steps


def _apigen_tools(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        parsed = _json_value(value)
    else:
        parsed = value
    if not isinstance(parsed, list):
        return []
    tools = []
    for tool in parsed:
        if not isinstance(tool, dict):
            continue
        if "function" in tool:
            tools.append(copy.deepcopy(tool))
        else:
            tools.append({"type": "function", "function": copy.deepcopy(tool)})
    return tools


def _hardgen_tools(value: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(value, str):
        parsed = _json_value(value)
    else:
        parsed = value
    if not isinstance(parsed, list):
        return ()
    tools = []
    for tool in parsed:
        if not isinstance(tool, dict):
            continue
        if "function" in tool:
            tools.append(copy.deepcopy(tool))
        else:
            tools.append({"type": "function", "function": copy.deepcopy(tool)})
    return tuple(tools)


def _hardgen_tool_observation_pairs(content: str) -> list[tuple[str, str]]:
    try:
        parsed = ast.literal_eval(content)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    pairs: list[tuple[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        for call, observation in item.items():
            pairs.append((str(call), str(observation)))
    return pairs


def _apigen_message(role: Any, value: Any) -> dict[str, Any] | None:
    if role == "human":
        return {"role": "user", "content": str(value)}
    if role == "system":
        return {"role": "system", "content": str(value)}
    if role == "gpt":
        return {"role": "assistant", "content": str(value)}
    return None


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _json_object(value: Any) -> dict[str, Any]:
    parsed = _json_value(value)
    return parsed if isinstance(parsed, dict) else {}


def _apigen_domain(entry: dict[str, Any]) -> str:
    # Dataset integration must declare its domain. Inferring benchmark
    # domains from prompt keywords silently injects task-specific policy into
    # the shared parser and misclassifies unrelated corpora.
    return str(entry.get("domain") or entry.get("category") or "")


def _first_user_text(conversations: list[Any]) -> str:
    for turn in conversations:
        if isinstance(turn, dict) and turn.get("from") == "human":
            return str(turn.get("value") or "")
    return ""


def _wildtoolbench_language_fields(
    entry: dict[str, Any],
    language: str,
) -> tuple[list[dict[str, Any]], list[str], list[list[dict[str, Any]]]]:
    if language == "en":
        return entry["english_tools"], entry["english_tasks"], entry["english_answer_list"]
    return entry["tools"], entry["tasks"], entry["answer_list"]


def _wildtoolbench_initial_messages(
    entry: dict[str, Any],
    tasks: list[str],
    task_idx: int,
    *,
    language: str,
    use_conversation_history: bool,
) -> list[dict[str, Any]]:
    if not use_conversation_history:
        return [{"role": "user", "content": tasks[task_idx]}]

    messages_key = "english_messages" if language == "en" else "messages"
    raw_messages = entry.get(messages_key, [])
    target_position = None
    task_cursor = 0
    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        if task_cursor < len(tasks) and message.get("content") == tasks[task_cursor]:
            if task_cursor == task_idx:
                target_position = index
                break
            task_cursor += 1
    if target_position is None:
        return [{"role": "user", "content": tasks[task_idx]}]
    return [
        copy.deepcopy(message)
        for message in raw_messages[: target_position + 1]
        if isinstance(message, dict) and message.get("role") in {"system", "user", "assistant"}
    ]


def _message_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [copy.deepcopy(item) for item in value if isinstance(item, dict)]


def _step_kind(*, expected_actions: tuple[ToolAction, ...], metadata: dict[str, Any]) -> str:
    if expected_actions:
        return "tool"
    if metadata.get("kind"):
        return str(metadata["kind"])
    return "text"


def _list_get(value: Any, index: int) -> Any:
    if isinstance(value, list) and 0 <= index < len(value):
        return value[index]
    return None

from __future__ import annotations

import copy
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from .adapter import DeclarativeTrajectoryAdapter, TrajectoryAdapter
from .parser import (
    parse_apigenmt_episode,
    parse_bfcl_episode,
    parse_hardgen_episode,
    parse_tau2_episode,
    parse_tau_episode,
    parse_wildtoolbench_episode,
)
from .schema import ToolAction, TrajectoryEpisode, TrajectoryStep


TRAJECTORY_SOURCES = {
    "auto",
    "wildtoolbench",
    "tau_bench",
    "tau2_bench",
    "apigen_mt",
    "bfcl",
    "hardgen",
    "trajectory2gym",
}


def load_trajectory_rows(path: str | Path, *, max_entries: int | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".zip":
        return _load_trajectory_rows_from_zip(path, max_entries=max_entries)
    if _starts_with_json_array(path):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"expected JSON array in trajectory file: {path}")
        rows = [_normalize_loaded_row(row) for row in rows]
        rows = [row for row in rows if row is not None]
        return rows[:max_entries] if max_entries is not None else rows

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = _normalize_loaded_row(json.loads(line))
            if row is None:
                continue
            rows.append(row)
            if max_entries is not None and len(rows) >= max_entries:
                break
    return rows


def _load_trajectory_rows_from_zip(path: Path, *, max_entries: int | None = None) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if len(names) != 1:
            raise ValueError(f"expected one data file in trajectory zip: {path}")
        name = names[0]
        with archive.open(name) as handle:
            text = handle.read().decode("utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError(f"expected JSON array in trajectory zip member: {path}:{name}")
        rows = [_normalize_loaded_row(row) for row in rows]
        rows = [row for row in rows if row is not None]
        return rows[:max_entries] if max_entries is not None else rows

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = _normalize_loaded_row(json.loads(line))
        if row is not None:
            rows.append(row)
        if max_entries is not None and len(rows) >= max_entries:
            break
    return rows


def _starts_with_json_array(path: Path) -> bool:
    with path.open(encoding="utf-8") as handle:
        while True:
            char = handle.read(1)
            if not char:
                return False
            if char.isspace():
                continue
            return char == "["


def _normalize_loaded_row(row: Any) -> dict[str, Any] | None:
    if isinstance(row, dict):
        return row
    if isinstance(row, list) and all(isinstance(item, dict) for item in row):
        return {str(index): item for index, item in enumerate(row)}
    return None


def load_trajectory_episodes(
    path: str | Path,
    *,
    source: str = "auto",
    language: str = "en",
    use_conversation_history: bool = True,
    max_entries: int | None = None,
    max_tasks: int | None = None,
) -> list[TrajectoryEpisode]:
    """Load any supported trajectory jsonl into the neutral episode schema."""

    if source not in TRAJECTORY_SOURCES:
        raise ValueError(f"unsupported trajectory source: {source}")

    episodes: list[TrajectoryEpisode] = []
    for row in load_trajectory_rows(path, max_entries=max_entries):
        row_source = _infer_source(row, source)
        if row_source == "trajectory2gym":
            episodes.append(_parse_trajectory2gym_episode(row))
        elif row_source == "wildtoolbench":
            episodes.extend(
                _wildtoolbench_episodes(
                    row,
                    language=language,
                    use_conversation_history=use_conversation_history,
                    remaining_tasks=None if max_tasks is None else max_tasks - len(episodes),
                )
            )
        elif row_source == "apigen_mt":
            episodes.append(parse_apigenmt_episode(row, row_index=len(episodes)))
        elif row_source == "bfcl":
            episodes.append(parse_bfcl_episode(row))
        elif row_source == "hardgen":
            episodes.append(parse_hardgen_episode(row, row_index=len(episodes)))
        elif row_source == "tau2_bench":
            episodes.append(parse_tau2_episode(row))
        else:
            episodes.append(parse_tau_episode(row))

        if max_tasks is not None and len(episodes) >= max_tasks:
            return [_fold_leading_assistant_setup(item) for item in episodes[:max_tasks]]
    return [_fold_leading_assistant_setup(item) for item in episodes]


def _fold_leading_assistant_setup(episode: TrajectoryEpisode) -> TrajectoryEpisode:
    """Move a structurally leading assistant setup turn into initial context.

    Logged dialogues sometimes start before the user request: the assistant
    greets, then the user task arrives as that step's observation. A deployed
    policy is normally invoked after the user message already exists. Folding
    this setup turn prevents a trajectory compiler from teaching a one-turn
    response offset without matching on benchmark names or greeting text.
    """

    initial_messages = list(episode.initial_messages)
    steps = list(episode.steps)
    folded = 0
    while (
        steps
        and not any(message.get("role") == "user" for message in initial_messages)
        and steps[0].kind == "assistant_text"
        and any(message.get("role") == "user" for message in steps[0].observations)
    ):
        setup = steps.pop(0)
        setup_content = str(
            setup.metadata.get("expected_assistant_message") or ""
        ).strip()
        if setup_content:
            initial_messages.append(
                {"role": "assistant", "content": setup_content}
            )
        initial_messages.extend(copy.deepcopy(list(setup.observations)))
        folded += 1
    if not folded:
        return episode
    reindexed_steps = tuple(replace(step, index=index) for index, step in enumerate(steps))
    metadata = copy.deepcopy(episode.metadata)
    metadata["leading_assistant_steps_folded"] = (
        int(metadata.get("leading_assistant_steps_folded") or 0) + folded
    )
    task = episode.task
    if task is None or task == "":
        task = _task_from_user_messages(initial_messages)
    return replace(
        episode,
        initial_messages=tuple(initial_messages),
        steps=reindexed_steps,
        task=task,
        metadata=metadata,
    )


def _task_from_user_messages(messages: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
    return "\n".join(
        str(message.get("content"))
        for message in messages
        if message.get("role") == "user"
        and message.get("content") is not None
        and str(message.get("content")) != ""
    )


def adapter_for_source(source: str) -> TrajectoryAdapter:
    """Return the same thin declarative adapter for every source.

    Source-specific parsing belongs to optional data compilers. Once a row is
    in the trajectory2gym schema, the training runtime is benchmark agnostic.
    """

    return DeclarativeTrajectoryAdapter(source=source)


def _infer_source(row: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    # The row shape describes its serialization; ``source`` describes where
    # the evidence came from.  A canonical trajectory2gym row deliberately
    # retains labels such as ``tau2_bench`` for provenance, so shape must take
    # precedence or a canonical round-trip will be reparsed as a legacy row
    # and lose ``initial_messages``/``expected_actions``.
    if "steps" in row and "initial_messages" in row:
        return "trajectory2gym"
    source = str(row.get("source") or "")
    if source in {"tau_bench", "tau2_bench"}:
        return source
    if source == "bfcl":
        return "bfcl"
    if _has_wildtoolbench_fields(row):
        return "wildtoolbench"
    if "conversations" in row and "tools" in row:
        return "apigen_mt"
    if row.get("source") == "bfcl" or "turns" in row:
        return "bfcl"
    if "0" in row and isinstance(row.get("0"), dict) and "tools" in row.get("0", {}):
        return "hardgen"
    return "tau_bench"


def _has_wildtoolbench_fields(row: dict[str, Any]) -> bool:
    return any(key in row for key in ("english_answer_list", "answer_list")) and any(
        key in row for key in ("english_tasks", "tasks")
    )


def _parse_trajectory2gym_episode(row: dict[str, Any]) -> TrajectoryEpisode:
    steps: list[TrajectoryStep] = []
    for raw_step in row.get("steps") or []:
        if not isinstance(raw_step, dict):
            continue

        # Early trajectory compilers used ``gold_actions``/``step_index`` and
        # stored the transition kind in metadata.  These are serialization
        # aliases of the neutral schema, not benchmark-specific semantics.
        # Normalize them here so tools that deliberately force the
        # ``trajectory2gym`` loader (for example evidence deduplication) do not
        # silently erase actions or state-mutation declarations.
        raw_actions = raw_step.get("expected_actions")
        if raw_actions is None:
            raw_actions = raw_step.get("gold_actions", ())
        if not isinstance(raw_actions, (list, tuple)):
            raw_actions = ()
        expected_actions = tuple(
            ToolAction.from_raw(action)
            for action in raw_actions
            if isinstance(action, dict) and action.get("name")
        )

        metadata = copy.deepcopy(raw_step.get("metadata") or {})
        if not isinstance(metadata, dict):
            metadata = {}
        if raw_step.get("action_metadata") is not None:
            metadata["action_metadata"] = copy.deepcopy(
                raw_step.get("action_metadata")
            )

        raw_index = raw_step.get("index")
        if raw_index is None:
            raw_index = raw_step.get("step_index")
        if raw_index is None:
            raw_index = len(steps)

        if expected_actions:
            kind = "tool"
        else:
            kind = str(raw_step.get("kind") or metadata.get("kind") or "text")
        steps.append(
            TrajectoryStep(
                index=int(raw_index),
                expected_actions=expected_actions,
                observations=tuple(
                    copy.deepcopy(observation)
                    for observation in raw_step.get("observations", ())
                    if isinstance(observation, dict)
                ),
                kind=kind,
                metadata=metadata,
            )
        )

    initial_messages = row.get("initial_messages")
    if not initial_messages:
        initial_messages = row.get("prompt", ())
    if not isinstance(initial_messages, (list, tuple)):
        initial_messages = ()

    task = row.get("task")
    if task is None or task == "":
        task = row.get("question")
    if task is None or task == "":
        task = _task_from_user_messages(list(initial_messages))
    return TrajectoryEpisode(
        id=str(row.get("id") or row.get("task_id") or "trajectory2gym"),
        source=str(row.get("source") or "trajectory2gym"),
        domain=row.get("domain"),
        task=task,
        tools=tuple(
            copy.deepcopy(tool)
            for tool in row.get("tools", ())
            if isinstance(tool, dict)
        ),
        initial_messages=tuple(
            copy.deepcopy(message)
            for message in initial_messages
            if isinstance(message, dict)
        ),
        steps=tuple(steps),
        metadata=copy.deepcopy(row.get("metadata") or {}),
    )


def _wildtoolbench_episodes(
    row: dict[str, Any],
    *,
    language: str,
    use_conversation_history: bool,
    remaining_tasks: int | None,
) -> list[TrajectoryEpisode]:
    tasks_key = "english_tasks" if language == "en" else "tasks"
    answers_key = "english_answer_list" if language == "en" else "answer_list"
    tasks = row.get(tasks_key, [])
    answers = row.get(answers_key, [])
    count = min(len(tasks), len(answers))
    if remaining_tasks is not None:
        count = min(count, max(remaining_tasks, 0))
    return [
        parse_wildtoolbench_episode(
            row,
            task_idx=task_idx,
            language=language,
            use_conversation_history=use_conversation_history,
        )
        for task_idx in range(count)
    ]

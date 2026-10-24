from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trajectory2gym.tool_calls import tool_name_from_schema

from .execution_graph import build_execution_graph
from .loader import load_trajectory_episodes
from .observation_db import ObservationDatabase, action_key
from .tool_semantics import DEFAULT_TOOL_SEMANTICS, ToolSemantics
from .schema import ToolAction, TrajectoryEpisode
from .tool_taxonomy import infer_tool_type
from .verifiability import VerifiabilityAnalyzer


READ_ONLY_PREFIXES = (
    "get",
    "list",
    "search",
    "retrieve",
    "find",
    "lookup",
    "check",
    "query",
)
DB_PROBE_TOOL_TYPES = {"search_retrieve_query", "deterministic_api"}
STATEFUL_PREFIXES = (
    "book",
    "buy",
    "cancel",
    "create",
    "delete",
    "exchange",
    "modify",
    "place",
    "purchase",
    "refund",
    "reserve",
    "return",
    "send",
    "set",
    "submit",
    "transfer",
    "update",
)
DETERMINISTIC_HINTS = (
    "calculate",
    "compute",
    "convert",
    "parse",
    "format",
)


@dataclass(frozen=True)
class TrajectoryCaseRow:
    dataset: str
    trajectory_id: str
    domain: str | None
    step_count: int
    tool_step_count: int
    tool_names: tuple[str, ...]
    tool_types: dict[str, str]
    exact_replayable_steps: int
    observation_db_candidate_steps: int
    ambiguous_probe_count: int
    invalid_probe_count: int
    graph_node_count: int
    graph_edge_count: int
    hidden_initial_state: str = "unknown"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "trajectory_id": self.trajectory_id,
            "domain": self.domain,
            "step_count": self.step_count,
            "tool_step_count": self.tool_step_count,
            "tool_names": list(self.tool_names),
            "tool_types": copy.deepcopy(self.tool_types),
            "exact_replayable_steps": self.exact_replayable_steps,
            "observation_db_candidate_steps": self.observation_db_candidate_steps,
            "ambiguous_probe_count": self.ambiguous_probe_count,
            "invalid_probe_count": self.invalid_probe_count,
            "graph_node_count": self.graph_node_count,
            "graph_edge_count": self.graph_edge_count,
            "hidden_initial_state": self.hidden_initial_state,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ActionCaseRow:
    dataset: str
    trajectory_id: str
    step_index: int
    tool_name: str
    tool_type: str
    classification: str
    runtime_label: str
    runtime_feedback: str
    evidence: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "trajectory_id": self.trajectory_id,
            "step_index": self.step_index,
            "tool_name": self.tool_name,
            "tool_type": self.tool_type,
            "arguments": copy.deepcopy(self.arguments),
            "classification": self.classification,
            "runtime_label": self.runtime_label,
            "runtime_feedback": self.runtime_feedback,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class CaseStudyReport:
    dataset: str
    source_path: str
    trajectory_rows: tuple[TrajectoryCaseRow, ...]
    action_rows: tuple[ActionCaseRow, ...]
    observation_db_summary: dict[str, Any]
    label_counts: dict[str, int]
    graph_dots: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_path": self.source_path,
            "trajectory_rows": [row.to_dict() for row in self.trajectory_rows],
            "action_rows": [row.to_dict() for row in self.action_rows],
            "observation_db_summary": copy.deepcopy(self.observation_db_summary),
            "label_counts": copy.deepcopy(self.label_counts),
            "graph_dots": copy.deepcopy(self.graph_dots),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Trajectory2Gym Case Study: {self.dataset}",
            "",
            f"Source: `{self.source_path}`",
            "",
            "## Summary",
            "",
            f"- trajectories audited: {len(self.trajectory_rows)}",
            f"- action rows: {len(self.action_rows)}",
            f"- observation DB: `{json.dumps(self.observation_db_summary, sort_keys=True)}`",
            f"- runtime label counts: `{json.dumps(self.label_counts, sort_keys=True)}`",
            "",
            "## Trajectory Rows",
            "",
            "| id | domain | steps | tool steps | tools | exact | db candidates | ambiguous probes | graph |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
        ]
        for row in self.trajectory_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.trajectory_id,
                        str(row.domain or ""),
                        str(row.step_count),
                        str(row.tool_step_count),
                        ", ".join(row.tool_names),
                        str(row.exact_replayable_steps),
                        str(row.observation_db_candidate_steps),
                        str(row.ambiguous_probe_count),
                        f"{row.graph_node_count} nodes/{row.graph_edge_count} edges",
                    ]
                )
                + " |"
            )
        lines.extend(["", "## Action Classification Samples", ""])
        lines.extend(
            [
                "| trajectory | step | tool | type | class | label | feedback | evidence |",
                "| --- | ---: | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in self.action_rows[:40]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.trajectory_id,
                        str(row.step_index),
                        row.tool_name,
                        row.tool_type,
                        row.classification,
                        row.runtime_label,
                        row.runtime_feedback,
                        row.evidence,
                    ]
                )
                + " |"
            )
        lines.extend(["", "## DAG Examples", ""])
        for episode_id, dot in self.graph_dots.items():
            lines.extend([f"### {episode_id}", "", "```dot", dot, "```", ""])
        return "\n".join(lines).rstrip() + "\n"


def analyze_episodes(
    episodes: list[TrajectoryEpisode] | tuple[TrajectoryEpisode, ...],
    *,
    dataset: str,
    source_path: str = "",
    graph_limit: int = 3,
    accept_equivalent: bool = True,
    use_observation_db: bool = True,
    include_probes: bool = True,
    tool_semantics: ToolSemantics = DEFAULT_TOOL_SEMANTICS,
) -> CaseStudyReport:
    observation_db = (
        ObservationDatabase.from_episodes(episodes, tool_semantics=tool_semantics)
        if use_observation_db
        else None
    )
    analyzer = VerifiabilityAnalyzer(
        observation_db=observation_db,
        accept_equivalent=accept_equivalent,
        tool_semantics=tool_semantics,
    )
    trajectory_rows: list[TrajectoryCaseRow] = []
    action_rows: list[ActionCaseRow] = []
    label_counts: Counter[str] = Counter()
    graph_dots: dict[str, str] = {}

    for episode in episodes:
        graph = build_execution_graph(episode)
        if len(graph_dots) < graph_limit:
            graph_dots[episode.id] = graph.to_dot()
        tool_types = infer_tool_types(episode.tools)
        tool_names = tuple(sorted({action.name for step in episode.steps for action in step.expected_actions}))
        tool_step_count = sum(1 for step in episode.steps if step.expected_actions)
        exact_replayable_steps = 0
        observation_db_candidate_steps = 0
        ambiguous_probe_count = 0
        invalid_probe_count = 0

        for step in episode.steps:
            if not step.expected_actions:
                continue
            action = step.expected_actions[0]
            action_payload: dict[str, Any] | list[dict[str, Any]]
            if len(step.expected_actions) == 1:
                action_payload = action.to_dict()
            else:
                action_payload = [item.to_dict() for item in step.expected_actions]
            exact = analyzer.classify(episode, step, action_payload)
            label_counts[exact.label] += 1
            exact_replayable_steps += int(exact.label == "valid_replay_exact")
            observation_db_candidate_steps += int(bool(step.observations) and len(step.expected_actions) == 1)
            action_rows.append(_action_row(dataset, episode.id, step.index, action, tool_types, exact.label, "valid"))

            db_probe = (
                _database_supported_probe(observation_db, action, tool_types)
                if include_probes and observation_db is not None
                else None
            )
            if db_probe is not None:
                db_decision = analyzer.classify(episode, step, db_probe.to_dict())
                label_counts[db_decision.label] += 1
                action_rows.append(
                    _action_row(
                        dataset,
                        episode.id,
                        step.index,
                        db_probe,
                        tool_types,
                        db_decision.label,
                        db_decision.action_class,
                    )
                )

            ambiguous_action = (
                _mutate_action(action) if include_probes and len(step.expected_actions) == 1 else None
            )
            if ambiguous_action is not None:
                ambiguous = analyzer.classify(episode, step, ambiguous_action.to_dict())
                label_counts[ambiguous.label] += 1
                ambiguous_probe_count += int(ambiguous.is_ambiguous)
                action_rows.append(
                    _action_row(
                        dataset,
                        episode.id,
                        step.index,
                        ambiguous_action,
                        tool_types,
                        ambiguous.label,
                        ambiguous.action_class,
                    )
                )

            invalid_action = _schema_invalid_action(action) if include_probes else None
            if invalid_action is not None:
                invalid = analyzer.classify(episode, step, invalid_action.to_dict())
                label_counts[invalid.label] += 1
                invalid_probe_count += int(invalid.is_invalid)
                action_rows.append(
                    _action_row(
                        dataset,
                        episode.id,
                        step.index,
                        invalid_action,
                        tool_types,
                        invalid.label,
                        invalid.action_class,
                    )
                )

        trajectory_rows.append(
            TrajectoryCaseRow(
                dataset=dataset,
                trajectory_id=episode.id,
                domain=episode.domain,
                step_count=episode.step_count,
                tool_step_count=tool_step_count,
                tool_names=tool_names,
                tool_types={name: tool_types.get(name, "unknown") for name in tool_names},
                exact_replayable_steps=exact_replayable_steps,
                observation_db_candidate_steps=observation_db_candidate_steps,
                ambiguous_probe_count=ambiguous_probe_count,
                invalid_probe_count=invalid_probe_count,
                graph_node_count=len(graph.nodes),
                graph_edge_count=len(graph.edges),
            )
        )

    return CaseStudyReport(
        dataset=dataset,
        source_path=source_path,
        trajectory_rows=tuple(trajectory_rows),
        action_rows=tuple(action_rows),
        observation_db_summary=observation_db.summary() if observation_db is not None else {},
        label_counts=dict(sorted(label_counts.items())),
        graph_dots=graph_dots,
    )


def infer_tool_types(tools: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> dict[str, str]:
    return {name: infer_tool_type(name) for name in (tool_name_from_schema(tool) for tool in tools) if name}


def _action_row(
    dataset: str,
    trajectory_id: str,
    step_index: int,
    action: ToolAction,
    tool_types: dict[str, str],
    label: str,
    action_class: str,
) -> ActionCaseRow:
    return ActionCaseRow(
        dataset=dataset,
        trajectory_id=trajectory_id,
        step_index=step_index,
        tool_name=action.name,
        tool_type=tool_types.get(action.name, infer_tool_type(action.name)),
        arguments=action.arguments,
        classification=action_class,
        runtime_label=label,
        runtime_feedback=_runtime_feedback(label),
        evidence=_evidence(label),
    )


def _runtime_feedback(label: str) -> str:
    if label.startswith("valid_"):
        return "observation"
    if label.startswith("invalid_"):
        return "typed_error"
    if label.startswith("ambiguous_"):
        return "abstain"
    return "unknown"


def _evidence(label: str) -> str:
    if label == "valid_replay_exact":
        return "same trajectory step"
    if label == "valid_replay_database":
        return "compiled observation database"
    if label.startswith("invalid_"):
        return "tool schema"
    if label.startswith("ambiguous_"):
        return "missing or conflicting observation evidence"
    return "matcher"


def _database_supported_probe(
    observation_db: ObservationDatabase,
    action: ToolAction,
    tool_types: dict[str, str],
) -> ToolAction | None:
    tool_type = tool_types.get(action.name, infer_tool_type(action.name))
    if tool_type not in DB_PROBE_TOOL_TYPES:
        return None
    current_key = action_key(action)
    for record in observation_db.records_for_tool(action.name):
        if record.key == current_key:
            continue
        lookup = observation_db.lookup(record.action)
        if lookup.status == "hit":
            return record.action
    return None


def _mutate_action(action: ToolAction) -> ToolAction | None:
    if not action.arguments:
        return None
    arguments = copy.deepcopy(action.arguments)
    key = sorted(arguments)[0]
    value = arguments[key]
    if isinstance(value, str):
        arguments[key] = value + "__unseen"
    elif isinstance(value, bool):
        arguments[key] = not value
    elif isinstance(value, int):
        arguments[key] = value + 99991
    elif isinstance(value, float):
        arguments[key] = value + 99991.0
    else:
        arguments[key] = "__unseen__"
    return ToolAction(name=action.name, arguments=arguments)


def _schema_invalid_action(action: ToolAction) -> ToolAction | None:
    if not action.arguments:
        return None
    arguments = copy.deepcopy(action.arguments)
    key = sorted(arguments)[0]
    arguments[key] = {"schema": "invalid"}
    return ToolAction(name=action.name, arguments=arguments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze trajectory2gym partial-environment case studies.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", default="auto")
    parser.add_argument("--source-label", default=None)
    parser.add_argument("--max-entries", type=int, default=10)
    parser.add_argument(
        "--no-accept-equivalent",
        action="store_true",
        help="disable heuristic equivalent-argument replay and report only exact/database supported accepts",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    episodes = load_trajectory_episodes(args.data_path, source=args.source, max_entries=args.max_entries)
    report = analyze_episodes(
        episodes,
        dataset=args.dataset,
        source_path=args.source_label or args.data_path,
        accept_equivalent=not args.no_accept_equivalent,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.write_text(report.to_markdown(), encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(report.to_markdown())


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trajectory2gym.tool_calls import tool_name_from_schema

from .loader import load_trajectory_episodes
from .schema import TrajectoryEpisode


SEARCH_PREFIXES = ("search", "find")
RETRIEVE_PREFIXES = ("retrieve", "fetch")
QUERY_PREFIXES = ("get", "list", "lookup", "check", "query")
DETERMINISTIC_PREFIXES = ("calculate", "compute", "convert", "parse", "format")
DETERMINISTIC_EXACT_NAMES = {
    "absolute_value",
    "add",
    "divide",
    "gallon_to_liter",
    "imperial_si_conversion",
    "liter_to_gallon",
    "logarithm",
    "max_value",
    "mean",
    "min_value",
    "multiply",
    "percentage",
    "power",
    "standard_deviation",
    "subtract",
}
BOOKING_PREFIXES = ("book", "buy", "order", "place", "purchase", "reserve")
STATEFUL_PREFIXES = (
    "add",
    "cancel",
    "create",
    "delete",
    "exchange",
    "modify",
    "refund",
    "remove",
    "return",
    "send",
    "set",
    "submit",
    "transfer",
    "update",
)


TOOL_TYPE_DEFINITIONS: dict[str, dict[str, str]] = {
    "search_retrieve_query": {
        "definition": "read-only information access where identical or equivalent queries can replay a recorded observation",
        "difficulty": "low",
        "compile_rule": "index canonical query arguments and replay only non-conflicting observations",
    },
    "deterministic_api": {
        "definition": "pure function API whose output can be recomputed or directly checked from arguments",
        "difficulty": "low",
        "compile_rule": "prefer deterministic execution or exact input-output table replay",
    },
    "booking_order": {
        "definition": "transactional tool that may depend on inventory, user account, reservations, or order state",
        "difficulty": "high",
        "compile_rule": "replay only observed transitions unless pre/post state can be recovered",
    },
    "stateful_update": {
        "definition": "tool call that mutates environment state outside the local observation",
        "difficulty": "high",
        "compile_rule": "require recovered state transition evidence; otherwise abstain off trajectory",
    },
    "hidden_state_tool": {
        "definition": "tool whose semantics depend on hidden initial state or opaque external context",
        "difficulty": "highest",
        "compile_rule": "accept exact verified transitions only; abstain on unsupported variants",
    },
    "unknown": {
        "definition": "tool name or schema is unavailable",
        "difficulty": "unknown",
        "compile_rule": "treat unsupported actions as invalid or ambiguous based on schema availability",
    },
}


@dataclass(frozen=True)
class ToolTaxonomyRow:
    dataset: str
    tool_name: str
    tool_type: str
    difficulty: str
    episodes: int
    trajectory_actions: int
    domains: tuple[str, ...] = ()
    example_arguments: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "tool_name": self.tool_name,
            "tool_type": self.tool_type,
            "difficulty": self.difficulty,
            "episodes": self.episodes,
            "trajectory_actions": self.trajectory_actions,
            "domains": list(self.domains),
            "example_arguments": copy.deepcopy(list(self.example_arguments)),
        }


@dataclass(frozen=True)
class ToolTaxonomyReport:
    dataset: str
    source_path: str
    rows: tuple[ToolTaxonomyRow, ...]
    type_counts: dict[str, int]
    action_type_counts: dict[str, int]
    definitions: dict[str, dict[str, str]] = field(default_factory=lambda: copy.deepcopy(TOOL_TYPE_DEFINITIONS))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_path": self.source_path,
            "rows": [row.to_dict() for row in self.rows],
            "type_counts": dict(self.type_counts),
            "action_type_counts": dict(self.action_type_counts),
            "definitions": copy.deepcopy(self.definitions),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Trajectory2Gym Tool Taxonomy: {self.dataset}",
            "",
            f"Source: `{self.source_path}`",
            "",
            "## Type Definitions",
            "",
            "| type | difficulty | definition | compile rule |",
            "| --- | --- | --- | --- |",
        ]
        for tool_type, spec in self.definitions.items():
            lines.append(
                f"| {tool_type} | {spec['difficulty']} | {spec['definition']} | {spec['compile_rule']} |"
            )
        lines.extend(
            [
                "",
                "## Dataset Counts",
                "",
                "| type | tools | trajectory actions |",
                "| --- | ---: | ---: |",
            ]
        )
        for tool_type in sorted(set(self.type_counts) | set(self.action_type_counts)):
            lines.append(
                f"| {tool_type} | {self.type_counts.get(tool_type, 0)} | {self.action_type_counts.get(tool_type, 0)} |"
            )
        lines.extend(
            [
                "",
                "## Tool Rows",
                "",
                "| tool | type | difficulty | episodes | actions | domains | example arguments |",
                "| --- | --- | --- | ---: | ---: | --- | --- |",
            ]
        )
        for row in self.rows:
            examples = json.dumps(list(row.example_arguments), ensure_ascii=False, sort_keys=True)
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.tool_name,
                        row.tool_type,
                        row.difficulty,
                        str(row.episodes),
                        str(row.trajectory_actions),
                        ", ".join(row.domains),
                        examples,
                    ]
                )
                + " |"
            )
        return "\n".join(lines).rstrip() + "\n"


def infer_tool_type(tool_name: str) -> str:
    name = tool_name.lower()
    if name in DETERMINISTIC_EXACT_NAMES or name.startswith(DETERMINISTIC_PREFIXES):
        return "deterministic_api"
    if name.startswith(SEARCH_PREFIXES + RETRIEVE_PREFIXES + QUERY_PREFIXES):
        return "search_retrieve_query"
    if name.startswith(BOOKING_PREFIXES):
        return "booking_order"
    if name.startswith(STATEFUL_PREFIXES):
        return "stateful_update"
    return "hidden_state_tool"


def analyze_tool_taxonomy(
    episodes: list[TrajectoryEpisode] | tuple[TrajectoryEpisode, ...],
    *,
    dataset: str,
    source_path: str = "",
    max_examples_per_tool: int = 3,
) -> ToolTaxonomyReport:
    tool_names_by_episode: dict[str, set[str]] = defaultdict(set)
    tool_domains: dict[str, set[str]] = defaultdict(set)
    action_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for episode in episodes:
        schema_tool_names = {name for name in (tool_name_from_schema(tool) for tool in episode.tools) if name}
        action_tool_names = {
            action.name
            for step in episode.steps
            for action in step.expected_actions
            if action.name
        }
        for name in schema_tool_names | action_tool_names:
            tool_names_by_episode[name].add(episode.id)
            if episode.domain:
                tool_domains[name].add(str(episode.domain))
        for step in episode.steps:
            for action in step.expected_actions:
                action_counts[action.name] += 1
                if len(examples[action.name]) < max_examples_per_tool:
                    examples[action.name].append(copy.deepcopy(action.arguments))

    rows: list[ToolTaxonomyRow] = []
    for tool_name in sorted(tool_names_by_episode):
        tool_type = infer_tool_type(tool_name)
        rows.append(
            ToolTaxonomyRow(
                dataset=dataset,
                tool_name=tool_name,
                tool_type=tool_type,
                difficulty=TOOL_TYPE_DEFINITIONS[tool_type]["difficulty"],
                episodes=len(tool_names_by_episode[tool_name]),
                trajectory_actions=int(action_counts.get(tool_name, 0)),
                domains=tuple(sorted(tool_domains.get(tool_name, set()))[:8]),
                example_arguments=tuple(examples.get(tool_name, [])),
            )
        )
    type_counts = Counter(row.tool_type for row in rows)
    action_type_counts = Counter()
    for row in rows:
        action_type_counts[row.tool_type] += row.trajectory_actions
    return ToolTaxonomyReport(
        dataset=dataset,
        source_path=source_path,
        rows=tuple(rows),
        type_counts=dict(sorted(type_counts.items())),
        action_type_counts=dict(sorted(action_type_counts.items())),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a dataset-level tool taxonomy report for trajectory2gym.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", default="auto")
    parser.add_argument("--source-label", default=None)
    parser.add_argument("--max-entries", type=int, default=100)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    episodes = load_trajectory_episodes(args.data_path, source=args.source, max_entries=args.max_entries)
    report = analyze_tool_taxonomy(
        episodes,
        dataset=args.dataset,
        source_path=args.source_label or args.data_path,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.write_text(report.to_markdown(), encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(report.to_markdown())


if __name__ == "__main__":
    main()

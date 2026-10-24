#!/usr/bin/env python3
"""Build nested scale views from certified retention and structural support.

The selector deliberately does not read Base NLL, benchmark names, domains,
teachers, or evaluation outcomes.  Every scale retains the same certified
reference core, closes the executable training-task and capability universes,
and then raises the minimum support of each benchmark-neutral ordered policy
path according to one predeclared scale law.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory2gym.training.global_sft_selection import (  # noqa: E402
    Candidate,
    prepare_candidates,
)


FORMAT = "trajectory2gym_structural_support_scaling_selection_v1"
DEFAULT_DEPTHS = {"4b": 2, "8b": 3, "14b": 4, "30b-a3b": 6}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _action_profile(row: Mapping[str, Any]) -> tuple[int, int, int]:
    calls: list[str] = []
    supervised = {
        int(value) for value in row.get("supervised_assistant_turn_indices") or []
    }
    supervised_text = 0
    assistant_ordinal = -1
    for message in row.get("messages") or []:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        assistant_ordinal += 1
        message_calls = message.get("tool_calls") or []
        if assistant_ordinal in supervised and not message_calls:
            supervised_text += 1
        for call in message_calls:
            function = call.get("function") if isinstance(call, Mapping) else None
            payload = function if isinstance(function, Mapping) else call
            calls.append(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
    return len(calls), len(calls) - len(set(calls)), supervised_text


def _anchor_key(candidate: Candidate) -> tuple[Any, ...]:
    actions, repeats, supervised_text = _action_profile(candidate.row)
    return (
        repeats,
        actions,
        candidate.tokens,
        -supervised_text,
        candidate.trace_hash,
    )


def _quality_key(candidate: Candidate) -> tuple[Any, ...]:
    """Universal structural quality; no provenance or learner likelihood."""

    return (
        -candidate.unsupported_reference_risk,
        -candidate.loop_redundancy,
        candidate.recovery,
        candidate.branch,
        candidate.cost_efficiency,
        -candidate.tokens,
        candidate.trace_hash,
    )


def _parse_depths(values: Iterable[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        scale, separator, depth = value.partition(":")
        if not separator or not scale or int(depth) < 1:
            raise ValueError(f"invalid SCALE:DEPTH value: {value}")
        parsed[scale] = int(depth)
    if not parsed:
        parsed = dict(DEFAULT_DEPTHS)
    if list(parsed.values()) != sorted(parsed.values()):
        raise ValueError("support depth must be non-decreasing in declared scale order")
    return parsed


def select_views(
    candidates: list[Candidate],
    *,
    reference_traces: set[str],
    scale_depths: Mapping[str, int],
    max_per_task: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    if not candidates or max_per_task < 1:
        raise ValueError("non-empty candidates and positive task cap are required")
    by_trace = {candidate.trace_hash: candidate for candidate in candidates}
    if len(by_trace) != len(candidates):
        raise ValueError("candidate trace identity is not unique")
    missing_reference = reference_traces - set(by_trace)
    if missing_reference:
        raise ValueError(f"reference core has {len(missing_reference)} absent traces")

    by_task: dict[str, list[Candidate]] = defaultdict(list)
    by_path: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_task[candidate.task].append(candidate)
        by_path[candidate.family].append(candidate)

    selected: dict[str, Candidate] = {}
    phase: dict[str, str] = {}
    admitted_at_scale: dict[str, str] = {}
    task_counts: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    covered_atoms: set[str] = set()

    def admit(candidate: Candidate, *, selected_phase: str, scale: str) -> bool:
        if candidate.trace_hash in selected:
            return False
        # The immutable retention core has precedence over the expansion cap.
        if selected_phase != "certified_reference_retention" and task_counts[candidate.task] >= max_per_task:
            return False
        selected[candidate.trace_hash] = candidate
        phase[candidate.trace_hash] = selected_phase
        admitted_at_scale[candidate.trace_hash] = scale
        task_counts[candidate.task] += 1
        path_counts[candidate.family] += 1
        covered_atoms.update(candidate.atoms)
        return True

    first_scale = next(iter(scale_depths))
    for trace in sorted(reference_traces):
        admit(
            by_trace[trace],
            selected_phase="certified_reference_retention",
            scale=first_scale,
        )

    # Every executable training task receives an anchor under one ordering.
    for task in sorted(by_task):
        if task_counts[task]:
            continue
        options = [candidate for candidate in by_task[task] if candidate.trace_hash not in selected]
        if not options:
            raise AssertionError("task closure has no eligible candidate")
        if not admit(min(options, key=_anchor_key), selected_phase="training_task_closure", scale=first_scale):
            raise AssertionError("training-task anchor exceeded the universal cap")

    # Close the set-valued capability vocabulary before support expansion.
    atom_universe = set().union(*(candidate.atoms for candidate in candidates))
    while atom_universe - covered_atoms:
        missing = atom_universe - covered_atoms
        options = []
        for candidate in candidates:
            if candidate.trace_hash in selected or task_counts[candidate.task] >= max_per_task:
                continue
            gain = len(candidate.atoms & missing)
            if gain:
                options.append(
                    (
                        gain / candidate.tokens,
                        gain,
                        _quality_key(candidate),
                        candidate,
                    )
                )
        if not options:
            raise RuntimeError("capability closure is blocked by the universal task cap")
        winner = max(options, key=lambda item: item[:3])[3]
        admit(winner, selected_phase="capability_atom_closure", scale=first_scale)

    views: dict[str, list[dict[str, Any]]] = {}
    audits: dict[str, dict[str, Any]] = {}
    previous_traces: set[str] = set()
    for scale, depth in scale_depths.items():
        # A depth is a support floor, not a cap on the frozen reference core.
        # Fill one ordinal globally across paths before proceeding to the next.
        for ordinal in range(1, depth + 1):
            choices = []
            for path, path_candidates in by_path.items():
                if path_counts[path] >= ordinal:
                    continue
                options = [
                    candidate
                    for candidate in path_candidates
                    if candidate.trace_hash not in selected
                    and task_counts[candidate.task] < max_per_task
                ]
                if options:
                    winner = max(options, key=_quality_key)
                    choices.append((_quality_key(winner), path, winner))
            for _, path, winner in sorted(choices, reverse=True, key=lambda item: (item[0], item[1])):
                if path_counts[path] >= ordinal:
                    continue
                admit(
                    winner,
                    selected_phase="ordered_path_support",
                    scale=scale,
                )

        traces = set(selected)
        if not previous_traces <= traces:
            raise AssertionError("scale views are not nested")
        previous_traces = traces
        rows: list[dict[str, Any]] = []
        for trace, candidate in sorted(
            selected.items(), key=lambda item: (item[1].task, item[1].family, item[0])
        ):
            row = copy.deepcopy(candidate.row)
            row["prior_global_selection"] = row.get("global_selection")
            row["global_selection"] = {
                "format": FORMAT,
                "phase": phase[trace],
                "admitted_at_scale": admitted_at_scale[trace],
                "support_depth": depth,
                "certified_target_tokens": candidate.tokens,
            }
            rows.append(row)
        by_source = Counter(str(row.get("source_corpus")) for row in rows)
        by_domain = Counter(str(row.get("domain")) for row in rows)
        path_support = Counter(path_counts.values())
        audits[scale] = {
            "format": FORMAT,
            "status": "pass",
            "scale": scale,
            "support_depth": depth,
            "semantic_parents": len(rows),
            "certified_target_tokens": sum(selected[trace].tokens for trace in traces),
            "tasks_selected": len(task_counts),
            "tasks_candidate": len(by_task),
            "paths_selected": sum(bool(path_counts[path]) for path in by_path),
            "paths_candidate": len(by_path),
            "capability_atoms_selected": len(covered_atoms),
            "capability_atoms_candidate": len(atom_universe),
            "rows_by_source": dict(sorted(by_source.items())),
            "rows_by_domain_provenance_only": dict(sorted(by_domain.items())),
            "phases": dict(sorted(Counter(phase[trace] for trace in traces).items())),
            "path_support_histogram": dict(sorted(path_support.items())),
            "reference_core_retained": len(reference_traces),
            "maximum_expansion_rows_per_task": max_per_task,
            "nll_used": False,
            "teacher_used_as_selection_feature": False,
            "benchmark_domain_category_used_as_selection_feature": False,
            "official_evaluation_information_used": False,
        }
        views[scale] = rows
    return views, audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--reference-phase", default="certified_reference_retention")
    parser.add_argument("--reference-lineage-field", default="")
    parser.add_argument("--reference-recovery-receipt", type=Path)
    parser.add_argument("--include-source", action="append", required=True)
    parser.add_argument("--scale-depth", action="append", default=[])
    parser.add_argument("--max-per-task", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    depths = _parse_depths(args.scale_depth)
    candidate_manifest = json.loads((args.candidate_dir / "manifest.json").read_text())
    token_audit = json.loads((args.candidate_dir / "token_audit.json").read_text())
    if not (
        candidate_manifest.get("status") == "pass"
        and candidate_manifest.get("official_evaluation_information_used") is False
        and token_audit.get("status") == "pass"
        and int(token_audit.get("truncated_rows") or 0) == 0
    ):
        raise ValueError("candidate evidence/token gate is incomplete")
    included_sources = set(args.include_source)
    rows = [
        row
        for row in _rows(args.candidate_dir / "train.jsonl")
        if str(row.get("source_corpus")) in included_sources
    ]
    token_map = {
        str(item["trace_hash"]): int(item["supervised_tokens"])
        for item in token_audit.get("row_lengths") or []
    }
    candidates, preparation = prepare_candidates(
        rows,
        supervised_tokens=token_map,
        require_multiscale_policy_gap=False,
        loop_redundancy_mode="certified_exact_adjacent_v2",
        path_family_mode="capability_path_v2",
    )
    reference_rows = _rows(args.reference_dir / "train.jsonl")
    original_reference_traces = {
        str(row["trace_hash"])
        for row in reference_rows
        if not args.reference_phase
        or (row.get("global_selection") or {}).get("phase") == args.reference_phase
    }
    if not original_reference_traces:
        raise ValueError("frozen reference core is empty")
    reference_traces: set[str] = set()
    lineage_to_trace: dict[str, str] = {}
    for row in rows:
        trace = str(row.get("trace_hash") or "")
        lineage = (
            str(row.get(args.reference_lineage_field) or "")
            if args.reference_lineage_field
            else ""
        )
        if lineage:
            if lineage in lineage_to_trace and lineage_to_trace[lineage] != trace:
                raise ValueError("one reference lineage maps to multiple policy targets")
            lineage_to_trace[lineage] = trace
        if trace in original_reference_traces:
            reference_traces.add(trace)
    for trace in original_reference_traces - reference_traces:
        mapped = lineage_to_trace.get(trace)
        if mapped:
            reference_traces.add(mapped)
    unmapped_reference = original_reference_traces - {
        trace
        for trace in original_reference_traces
        if trace in {str(row.get("trace_hash") or "") for row in rows}
        or trace in lineage_to_trace
    }
    recovery_receipt = None
    if unmapped_reference:
        if args.reference_recovery_receipt is None:
            raise ValueError(
                "not every frozen reference target has an exact or transport-lineage mapping"
            )
        recovery_receipt = json.loads(args.reference_recovery_receipt.read_text())
        reference_by_trace = {
            str(row["trace_hash"]): row
            for row in reference_rows
            if str(row["trace_hash"]) in original_reference_traces
        }
        excluded_by_source = dict(
            sorted(Counter(str(reference_by_trace[trace].get("source_corpus") or "") for trace in unmapped_reference).items())
        )
        if not (
            recovery_receipt.get("format") == "trajectory2gym_transport_recoverable_reference_core_v1"
            and recovery_receipt.get("status") == "frozen"
            and recovery_receipt.get("candidate_train_sha256") == _sha(args.candidate_dir / "train.jsonl")
            and recovery_receipt.get("reference_train_sha256") == _sha(args.reference_dir / "train.jsonl")
            and recovery_receipt.get("original_reference_rows") == len(original_reference_traces)
            and recovery_receipt.get("recoverable_reference_rows") == len(reference_traces)
            and recovery_receipt.get("excluded_unmapped_rows") == len(unmapped_reference)
            and recovery_receipt.get("excluded_by_source") == excluded_by_source
            and recovery_receipt.get("official_evaluation_information_used") is False
            and recovery_receipt.get("policy") == "exclude only old reference targets without exact or explicit transport-lineage identity"
        ):
            raise ValueError("reference recovery receipt differs from observed transport mapping")
    views, audits = select_views(
        candidates,
        reference_traces=reference_traces,
        scale_depths=depths,
        max_per_task=args.max_per_task,
    )

    temporary = args.output_dir.with_name(f".{args.output_dir.name}.{os.getpid()}.tmp")
    temporary.mkdir(parents=True)
    try:
        summaries: dict[str, Any] = {}
        for scale, rows_for_scale in views.items():
            directory = temporary / scale
            directory.mkdir()
            train = directory / "train.jsonl"
            with train.open("x", encoding="utf-8") as handle:
                for row in rows_for_scale:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            (directory / "validation.jsonl").touch()
            traces = {str(row["trace_hash"]) for row in rows_for_scale}
            selected_lengths = [
                {**item, "index": index}
                for index, item in enumerate(
                    item
                    for item in token_audit.get("row_lengths") or []
                    if str(item["trace_hash"]) in traces
                )
            ]
            if {str(item["trace_hash"]) for item in selected_lengths} != traces:
                raise AssertionError("selected token accounting is incomplete")
            scale_token_audit = {
                "format": "trajectory2gym_semantic_parent_token_audit_v1",
                "status": "pass",
                "input": str((args.output_dir / scale / "train.jsonl").resolve()),
                "input_sha256": _sha(train),
                "rows": len(rows_for_scale),
                "truncated_rows": 0,
                "row_lengths": selected_lengths,
                "supervised_tokens": {
                    "total": sum(int(item["supervised_tokens"]) for item in selected_lengths)
                },
            }
            (directory / "token_audit.json").write_text(
                json.dumps(scale_token_audit, indent=2, sort_keys=True) + "\n"
            )
            audit = {**audits[scale], "train_sha256": _sha(train)}
            (directory / "selection_audit.json").write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n"
            )
            manifest = {
                "format": "trajectory2gym_structural_support_semantic_manifest_v1",
                "status": "pass",
                "training_ready": False,
                "method_contract_id": "structural_support_scaling_v1",
                "method_role": "certified_retention_task_capability_and_ordered_path_support",
                "candidate_manifest": str((args.candidate_dir / "manifest.json").resolve()),
                "candidate_train_sha256": _sha(args.candidate_dir / "train.jsonl"),
                "reference_train_sha256": _sha(args.reference_dir / "train.jsonl"),
                "train_rows": len(rows_for_scale),
                "train_sha256": _sha(train),
                "validation_rows": 0,
                "validation_sha256": _sha(directory / "validation.jsonl"),
                "train_rows_by_corpus": audit["rows_by_source"],
                "train_rows_by_domain": audit["rows_by_domain_provenance_only"],
                "selection_audit": audit,
                "config": {
                    "model_scale": scale,
                    "selection_mode": "structural_support_scaling_v1",
                    "support_depth": depths[scale],
                    "max_per_task": args.max_per_task,
                    "enable_thinking": False,
                    "success_only": True,
                    "supervision_mode": "verified_agentic",
                },
                "benchmark_specific_selection": False,
                "benchmark_specific_loss_weights": False,
                "official_evaluation_information_used": False,
                "base_nll_used": False,
                "active_training_sources": sorted(included_sources),
                "excluded_training_sources": sorted(
                    {"bfcl", "tau1", "tau2", "tau3"} - included_sources
                ),
            }
            (directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            (directory / "SEMANTIC_SELECTION_READY").touch()
            summaries[scale] = audit
        receipt = {
            "format": FORMAT,
            "status": "pass",
            "candidate": str(args.candidate_dir.resolve()),
            "candidate_train_sha256": _sha(args.candidate_dir / "train.jsonl"),
            "reference": str(args.reference_dir.resolve()),
            "reference_train_sha256": _sha(args.reference_dir / "train.jsonl"),
            "reference_phase": args.reference_phase,
            "reference_lineage_field": args.reference_lineage_field or None,
            "original_reference_rows": len(original_reference_traces),
            "mapped_reference_rows": len(reference_traces),
            "excluded_unmapped_reference_rows": len(unmapped_reference),
            "reference_recovery_receipt": (
                str(args.reference_recovery_receipt.resolve())
                if args.reference_recovery_receipt is not None
                else None
            ),
            "reference_recovery_receipt_sha256": (
                _sha(args.reference_recovery_receipt)
                if args.reference_recovery_receipt is not None
                else None
            ),
            "active_sources": sorted(included_sources),
            "scale_depths": depths,
            "maximum_expansion_rows_per_task": args.max_per_task,
            "preparation": preparation,
            "views": summaries,
            "base_nll_used": False,
            "teacher_used_as_selection_feature": False,
            "benchmark_domain_category_used_as_selection_feature": False,
            "official_evaluation_information_used": False,
        }
        (temporary / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, args.output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

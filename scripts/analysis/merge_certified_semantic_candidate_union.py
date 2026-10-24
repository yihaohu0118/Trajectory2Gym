#!/usr/bin/env python3
"""Merge immutable certified semantic pools without selection or NLL features."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _semantic(row: Mapping[str, Any]) -> str:
    value = {
        "messages": row.get("messages") or [],
        "tools": row.get("tools") or [],
        "supervised_assistant_turn_indices": row.get("supervised_assistant_turn_indices") or [],
        "verified_assistant_turn_indices": row.get("verified_assistant_turn_indices") or [],
    }
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _contains_think_tag(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return "<think" in lowered or "</think" in lowered
    if isinstance(value, Mapping):
        return any(_contains_think_tag(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_think_tag(item) for item in value)
    return False


def _assistant_reasoning_leak(row: Mapping[str, Any]) -> bool:
    for message in row.get("messages") or []:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        if str(message.get("reasoning_content") or "").strip():
            return True
        if _contains_think_tag(message.get("content")):
            return True
        for call in message.get("tool_calls") or []:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function") if isinstance(call.get("function"), Mapping) else call
            if _contains_think_tag(function.get("arguments")):
                return True
    return False


def merge(
    inputs: list[Path],
    active_sources: set[str],
    input_source_filters: list[set[str]] | None = None,
    rehash_input_indices: set[int] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    if input_source_filters is None:
        input_source_filters = [set(active_sources) for _ in inputs]
    if len(input_source_filters) != len(inputs):
        raise ValueError("one input-source filter is required per input directory")
    if any(not sources or not sources <= active_sources for sources in input_source_filters):
        raise ValueError("input-source filters must be non-empty active-source subsets")
    rehash_input_indices = set(rehash_input_indices or ())
    if any(index < 1 or index > len(inputs) for index in rehash_input_indices):
        raise ValueError("rehash-input indices are one-based and must name an input")
    output: list[dict] = []
    lengths: list[dict] = []
    traces: dict[str, str] = {}
    semantics: set[str] = set()
    duplicate_trace = duplicate_semantic = 0
    nll_fields_removed = 0
    transport_trace_rebindings = 0
    reasoning_leak_rows_removed = 0
    reasoning_leak_by_source: Counter[str] = Counter()
    input_receipts = []
    for input_index, (root, source_filter) in enumerate(zip(inputs, input_source_filters), start=1):
        manifest = json.loads((root / "manifest.json").read_text())
        audit = json.loads((root / "token_audit.json").read_text())
        if not (
            manifest.get("status") == "pass"
            and manifest.get("official_evaluation_information_used") is False
            and audit.get("status") == "pass"
            and int(audit.get("truncated_rows") or 0) == 0
        ):
            raise ValueError(f"input gate failed: {root}")
        token_by_trace = {
            str(item["trace_hash"]): item for item in audit.get("row_lengths") or []
        }
        admitted = 0
        for source in _rows(root / "train.jsonl"):
            if str(source.get("source_corpus") or "") not in source_filter:
                continue
            original_trace = str(source.get("trace_hash") or "")
            if not original_trace or original_trace not in token_by_trace:
                raise ValueError("row lacks trace-bound target-token accounting")
            if source.get("success") is not True or source.get("enable_thinking") is not False:
                raise ValueError("candidate violates success/non-thinking gate")
            if not source.get("supervised_assistant_turn_indices"):
                raise ValueError("candidate has no certified policy target")
            if _assistant_reasoning_leak(source):
                reasoning_leak_rows_removed += 1
                reasoning_leak_by_source[str(source.get("source_corpus") or "")] += 1
                continue
            semantic = _semantic(source)
            row = copy.deepcopy(source)
            trace = original_trace
            if input_index in rehash_input_indices:
                trace = hashlib.sha256(
                    json.dumps(
                        {
                            "parent_trace_hash": original_trace,
                            "policy_target_semantic_hash": semantic,
                            "transport_source_train_sha256": manifest.get("train_sha256"),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                row["pre_candidate_union_trace_hash"] = original_trace
                row["trace_hash"] = trace
                transport_trace_rebindings += 1
            if trace in traces:
                if traces[trace] != semantic:
                    raise ValueError("same trace hash names different policy targets")
                duplicate_trace += 1
                continue
            if semantic in semantics:
                duplicate_semantic += 1
                continue
            for field in list(row):
                lowered = field.lower()
                if lowered == "multi_scale_policy_gap" or "base_nll" in lowered:
                    row.pop(field, None)
                    nll_fields_removed += 1
            traces[trace] = semantic
            semantics.add(semantic)
            output.append(row)
            lengths.append(
                {
                    **token_by_trace[original_trace],
                    "trace_hash": trace,
                    "index": len(lengths),
                }
            )
            admitted += 1
        input_receipts.append(
            {
                "path": str(root.resolve()),
                "train_sha256": _sha(root / "train.jsonl"),
                "source_filter": sorted(source_filter),
                "trace_rebound_for_transport": input_index in rehash_input_indices,
                "eligible_rows_admitted_before_cross_input_dedup": admitted,
            }
        )
    by_source = Counter(str(row["source_corpus"]) for row in output)
    by_domain = Counter(str(row.get("domain") or "") for row in output)
    tasks = {
        str(row.get("source_task_id") or row.get("task_id") or "") for row in output
    }
    receipt = {
        "format": "trajectory2gym_certified_semantic_candidate_union_v1",
        "status": "pass",
        "inputs": input_receipts,
        "active_sources": sorted(active_sources),
        "excluded_sources": sorted({"bfcl", "tau1", "tau2", "tau3"} - active_sources),
        "rows": len(output),
        "tasks": len(tasks),
        "rows_by_source": dict(sorted(by_source.items())),
        "rows_by_domain_provenance_only": dict(sorted(by_domain.items())),
        "duplicate_trace_rows_removed": duplicate_trace,
        "duplicate_semantic_rows_removed": duplicate_semantic,
        "nll_fields_removed": nll_fields_removed,
        "transport_trace_rebindings": transport_trace_rebindings,
        "transport_trace_rebinding_formula": "sha256(parent trace, policy-target semantic hash, transport source train sha256)",
        "assistant_reasoning_leak_rows_removed": reasoning_leak_rows_removed,
        "assistant_reasoning_leak_rows_removed_by_source": dict(sorted(reasoning_leak_by_source.items())),
        "assistant_reasoning_leak_policy": "drop the complete trajectory only for non-empty reasoning_content or literal think-tag leakage; an explicit environment-visible tool named think is not internal reasoning",
        "base_nll_used": False,
        "benchmark_domain_category_teacher_or_evaluation_feature_used": False,
    }
    return output, lengths, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--input-source", action="append", help="comma-separated source filter aligned with --input-dir")
    parser.add_argument("--rehash-input", action="append", type=int, default=[], help="one-based input index whose corrected policy transport requires new trace identities")
    parser.add_argument("--include-source", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    active_sources = set(args.include_source)
    source_filters = None
    if args.input_source:
        source_filters = [
            {item.strip() for item in value.split(",") if item.strip()}
            for value in args.input_source
        ]
    rows, lengths, receipt = merge(
        args.input_dir,
        active_sources,
        input_source_filters=source_filters,
        rehash_input_indices=set(args.rehash_input),
    )
    temporary = args.output_dir.with_name(f".{args.output_dir.name}.{os.getpid()}.tmp")
    temporary.mkdir(parents=True)
    try:
        train = temporary / "train.jsonl"
        with train.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        (temporary / "validation.jsonl").touch()
        train_sha = _sha(train)
        audit = {
            "format": "trajectory2gym_semantic_parent_target_mass_v1",
            "status": "pass",
            "input": str((args.output_dir / "train.jsonl").resolve()),
            "input_sha256": train_sha,
            "rows": len(rows),
            "truncated_rows": 0,
            "row_lengths": lengths,
            "supervised_tokens": {
                "total": sum(int(item["supervised_tokens"]) for item in lengths)
            },
        }
        (temporary / "token_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "format": "trajectory2gym_generalist_certified_sft_manifest_v1",
            "status": "pass",
            "method_role": "active_interaction_synthesis_candidate_union",
            "train_rows": len(rows),
            "validation_rows": 0,
            "train_sha256": train_sha,
            "validation_sha256": _sha(temporary / "validation.jsonl"),
            "train_rows_by_corpus": receipt["rows_by_source"],
            "train_rows_by_domain": receipt["rows_by_domain_provenance_only"],
            "config": {
                "enable_thinking": False,
                "success_only": True,
                "supervision_mode": "verified_agentic",
                "selection_mode": "candidate_pool",
            },
            "candidate_union_receipt": receipt,
            "benchmark_specific_selection": False,
            "benchmark_specific_loss_weights": False,
            "official_evaluation_information_used": False,
            "base_nll_used": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "CANDIDATE_UNION_READY").touch()
        os.replace(temporary, args.output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({**receipt, "train_sha256": train_sha}, sort_keys=True))


if __name__ == "__main__":
    main()

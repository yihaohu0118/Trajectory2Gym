#!/usr/bin/env python3
"""Overlay certified active-synthesis rows on a frozen successful SFT view.

This is an explicitly targeted diagnostic.  It preserves every base row and
adds all semantically distinct, already-certified teacher rows matching a
frozen provenance filter.  It never reads evaluation outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _semantic(row: dict) -> str:
    payload = {
        "messages": row.get("messages") or [],
        "tools": row.get("tools") or [],
        "supervised_assistant_turn_indices": row.get(
            "supervised_assistant_turn_indices"
        )
        or [],
        "verified_assistant_turn_indices": row.get(
            "verified_assistant_turn_indices"
        )
        or [],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def overlay(base: list[dict], additions: list[dict]) -> tuple[list[dict], dict]:
    traces = set()
    semantics = set()
    output = []
    for row in base:
        trace = str(row.get("trace_hash") or "")
        if not trace or trace in traces:
            raise ValueError("base trace identity is missing or duplicated")
        if row.get("success") is not True or row.get("enable_thinking") is not False:
            raise ValueError("base row violates success/non-thinking gate")
        traces.add(trace)
        semantics.add(_semantic(row))
        output.append(copy.deepcopy(row))
    duplicate_trace = duplicate_semantic = 0
    selected = []
    for row in sorted(additions, key=lambda item: str(item.get("trace_hash"))):
        trace = str(row.get("trace_hash") or "")
        if not trace:
            raise ValueError("addition lacks trace identity")
        if row.get("success") is not True or row.get("enable_thinking") is not False:
            raise ValueError("addition violates success/non-thinking gate")
        if not row.get("supervised_assistant_turn_indices"):
            raise ValueError("addition lacks certified target decisions")
        if trace in traces:
            duplicate_trace += 1
            continue
        semantic = _semantic(row)
        if semantic in semantics:
            duplicate_semantic += 1
            continue
        copied = copy.deepcopy(row)
        copied["prior_global_selection"] = copied.get("global_selection")
        copied["global_selection"] = {
            "format": "trajectory2gym_targeted_active_overlay_v1",
            "phase": "guaranteed_active_synthesis_admission",
            "evaluation_outcome_used": False,
        }
        output.append(copied)
        selected.append(copied)
        traces.add(trace)
        semantics.add(semantic)
    audit = {
        "format": "trajectory2gym_targeted_active_overlay_v1",
        "status": "pass",
        "base_rows": len(base),
        "candidate_addition_rows": len(additions),
        "selected_addition_rows": len(selected),
        "duplicate_trace_additions": duplicate_trace,
        "duplicate_semantic_additions": duplicate_semantic,
        "output_rows": len(output),
        "selected_additions_by_domain_provenance_only": dict(
            sorted(Counter(str(row.get("domain")) for row in selected).items())
        ),
        "base_rows_removed": 0,
        "messages_or_masks_changed": False,
        "official_evaluation_information_used": False,
        "non_mainline_method_deviation": True,
    }
    return output, audit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, required=True)
    p.add_argument("--candidate-dir", type=Path, required=True)
    p.add_argument("--source-corpus", required=True)
    p.add_argument("--source-prefix", required=True)
    p.add_argument("--exclude-trace", action="append", default=[])
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    bm = json.loads((a.base_dir / "manifest.json").read_text())
    cm = json.loads((a.candidate_dir / "manifest.json").read_text())
    if not (
        bm.get("status") == "pass"
        and bm.get("official_evaluation_information_used") is False
        and cm.get("status") == "pass"
        and cm.get("official_evaluation_information_used") is False
    ):
        raise ValueError("base/candidate evidence gates are incomplete")
    excluded_traces = set(a.exclude_trace)
    candidates = [
        row
        for row in _rows(a.candidate_dir / "train.jsonl")
        if str(row.get("source_corpus")) == a.source_corpus
        and str(row.get("source") or "").startswith(a.source_prefix)
        and str(row.get("trace_hash") or "") not in excluded_traces
    ]
    available_traces = {
        str(row.get("trace_hash") or "")
        for row in _rows(a.candidate_dir / "train.jsonl")
    }
    if not excluded_traces <= available_traces:
        raise ValueError("one or more excluded traces are absent from the candidate")
    selected, audit = overlay(_rows(a.base_dir / "train.jsonl"), candidates)
    audit["excluded_candidate_traces"] = sorted(excluded_traces)
    temporary = a.output_dir.with_name(f".{a.output_dir.name}.{os.getpid()}.tmp")
    temporary.mkdir(parents=True)
    try:
        train = temporary / "train.jsonl"
        with train.open("x", encoding="utf-8") as f:
            for row in selected:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        (temporary / "validation.jsonl").touch()
        manifest = {
            **bm,
            "format": "trajectory2gym_targeted_active_overlay_semantic_sft_manifest_v1",
            "status": "pass",
            "method_role": "full_base_plus_targeted_certified_active_synthesis",
            "training_ready": False,
            "parent_manifest": str((a.base_dir / "manifest.json").resolve()),
            "parent_manifest_sha256": _sha(a.base_dir / "manifest.json"),
            "candidate_manifest": str((a.candidate_dir / "manifest.json").resolve()),
            "candidate_manifest_sha256": _sha(a.candidate_dir / "manifest.json"),
            "train_rows": len(selected),
            "train_sha256": _sha(train),
            "validation_rows": 0,
            "validation_sha256": _sha(temporary / "validation.jsonl"),
            "targeted_active_overlay": audit,
            "targeted_source_corpus": a.source_corpus,
            "targeted_source_prefix": a.source_prefix,
            "excluded_candidate_traces": sorted(excluded_traces),
            "benchmark_specific_selection": True,
            "benchmark_specific_loss_weights": False,
            "official_evaluation_information_used": False,
            "non_mainline_method_deviation": True,
            "token_audit_pending": True,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "SEMANTIC_SELECTION_READY").touch()
        os.replace(temporary, a.output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Restore semantic-parent loss mass after transport-only row projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


FIELD = "transport_parent_loss_weight"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parent(row: dict) -> str:
    return str(row.get("transport_projection_parent_trace_hash") or row["trace_hash"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest_path = args.input_dir / "manifest.json"
    train_path = args.input_dir / "train.jsonl"
    validation_path = args.input_dir / "validation.jsonl"
    audit_path = args.input_dir / "token_audit.json"
    readiness_path = args.input_dir / "selection_readiness.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    readiness = json.loads(readiness_path.read_text())
    if not (
        manifest.get("format") == "trajectory2gym_global_selected_rendered_sft_manifest_v1"
        and manifest.get("training_ready") is True
        and manifest.get("benchmark_specific_loss_weights") is False
        and audit.get("status") == "pass"
        and audit.get("input_sha256") == _sha(train_path)
        and readiness.get("training_permitted") is True
        and validation_path.stat().st_size == 0
    ):
        raise ValueError("input is not a passed immutable unified SFT view")
    rows = _read_jsonl(train_path)
    token_by_trace = {
        str(item["trace_hash"]): int(item["supervised_tokens"])
        for item in audit.get("row_lengths") or []
    }
    if len(token_by_trace) != len(rows):
        raise ValueError("token audit does not cover every rendered row")
    tokens_by_parent: Counter[str] = Counter()
    rows_by_parent: Counter[str] = Counter()
    corpus_by_parent: dict[str, str] = {}
    for row in rows:
        trace = str(row.get("trace_hash") or "")
        if trace not in token_by_trace:
            raise ValueError(f"rendered row lacks audited tokens: {trace}")
        if any("loss_weight" in key for key in row):
            raise ValueError("input already carries a loss-weight field")
        parent = _parent(row)
        tokens_by_parent[parent] += token_by_trace[trace]
        rows_by_parent[parent] += 1
        corpus = str(row.get("source_corpus") or "unknown")
        if parent in corpus_by_parent and corpus_by_parent[parent] != corpus:
            raise ValueError("one semantic parent crosses source corpora")
        corpus_by_parent[parent] = corpus
    rendered_rows = len(rows)
    semantic_parents = len(rows_by_parent)
    global_rescale = rendered_rows / semantic_parents
    corrected = []
    mass_by_parent: Counter[str] = Counter()
    for row in rows:
        trace = str(row["trace_hash"])
        parent = _parent(row)
        weight = global_rescale * token_by_trace[trace] / tokens_by_parent[parent]
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("derived transport loss weight is invalid")
        corrected.append({**row, FIELD: weight})
        mass_by_parent[parent] += weight
    max_error = max(
        (abs(value - global_rescale) for value in mass_by_parent.values()),
        default=0.0,
    )
    if max_error > 1e-10:
        raise ValueError(f"parent loss mass is not conserved: {max_error}")
    parents_by_corpus: Counter[str] = Counter(corpus_by_parent.values())
    rendered_by_corpus: Counter[str] = Counter(
        str(row.get("source_corpus") or "unknown") for row in rows
    )
    projected_parents = sum(count > 1 for count in rows_by_parent.values())
    correction = {
        "format": "trajectory2gym_transport_parent_loss_mass_v1",
        "status": "pass",
        "field": FIELD,
        "semantic_parents": semantic_parents,
        "rendered_rows": rendered_rows,
        "global_rescale": global_rescale,
        "parent_target_mass": global_rescale,
        "maximum_parent_mass_error": max_error,
        "projected_parents_with_multiple_children": projected_parents,
        "maximum_children_per_parent": max(rows_by_parent.values(), default=0),
        "semantic_parents_by_source_corpus": dict(sorted(parents_by_corpus.items())),
        "rendered_rows_by_source_corpus": dict(sorted(rendered_by_corpus.items())),
        "benchmark_or_category_feature_used": False,
        "weight_depends_only_on": [
            "audited_child_supervised_tokens",
            "audited_parent_supervised_tokens",
            "global_rendered_row_count",
            "global_semantic_parent_count",
        ],
    }
    temporary = args.output_dir.with_name(f".{args.output_dir.name}.{os.getpid()}.tmp")
    temporary.mkdir(parents=True)
    try:
        output_train = temporary / "train.jsonl"
        with output_train.open("x", encoding="utf-8") as handle:
            for row in corrected:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        shutil.copy2(validation_path, temporary / "validation.jsonl")
        shutil.copy2(readiness_path, temporary / "selection_readiness.json")
        derived_audit = {
            **audit,
            "input": str((args.output_dir / "train.jsonl").resolve()),
            "input_sha256": _sha(output_train),
            "derived_from_parent_token_audit": str(audit_path.resolve()),
            "derived_from_parent_token_audit_sha256": _sha(audit_path),
            "metadata_only_transport_weight_insertion": True,
        }
        (temporary / "token_audit.json").write_text(
            json.dumps(derived_audit, indent=2, sort_keys=True) + "\n"
        )
        output_manifest = {
            **manifest,
            "parent_manifest": str(manifest_path.resolve()),
            "parent_manifest_sha256": _sha(manifest_path),
            "train_sha256": _sha(output_train),
            "validation_sha256": _sha(temporary / "validation.jsonl"),
            "token_audit": str((args.output_dir / "token_audit.json").resolve()),
            "token_audit_sha256": _sha(temporary / "token_audit.json"),
            "transport_parent_mass_correction": correction,
            "row_loss_weight_field": FIELD,
            "benchmark_specific_loss_weights": False,
            "training_ready": True,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(output_manifest, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "DATA_READY").touch()
        os.replace(temporary, args.output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(correction, sort_keys=True))


if __name__ == "__main__":
    main()

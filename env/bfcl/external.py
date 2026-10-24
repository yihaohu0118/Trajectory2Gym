from __future__ import annotations

"""BFCL external batch-evaluation helpers.

Paper-facing batch evaluation may still call an external official checkout.
Interactive training uses :mod:`env.bfcl.official`, which wraps the same
official ``bfcl_eval`` loader, executable multi-turn backend, and checkers.
Neither facade is a compiled trajectory runtime.
"""

import json
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BFCL_EXTERNAL_RUN_MANIFEST = "_trajectory2gym_bfcl_external_run_manifest.json"

BFCL_SINGLE_CATEGORIES = (
    "simple_python",
    "multiple",
    "parallel",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_irrelevance",
)
BFCL_MULTI_CATEGORIES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    "multi_turn_composite",
)


@dataclass(frozen=True)
class ExternalBFCLRunConfig:
    external_root: Path
    model: str
    output_dir: Path
    mode: str = "multi"
    category: str = "multi_turn_base"
    python_bin: str = sys.executable
    run_all: bool = False
    num_tasks: int | None = None
    task_ids: tuple[str, ...] = ()
    task_ids_file: Path | None = None
    workers: int | None = None
    max_retries: int | None = None
    resume: bool = False
    multi_turn_schema_dirs: tuple[Path, ...] = ()
    debug: bool = False
    verbose: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def runner_path(self) -> Path:
        runner = "run_bfcl_multi.py" if self.mode == "multi" else "run_bfcl_single.py"
        return self.external_root / runner

    def validate(self) -> None:
        if self.mode not in {"single", "multi"}:
            raise ValueError(f"unsupported BFCL mode: {self.mode}")
        valid_categories = BFCL_MULTI_CATEGORIES if self.mode == "multi" else BFCL_SINGLE_CATEGORIES
        if self.category not in valid_categories:
            raise ValueError(
                f"unsupported BFCL {self.mode} category {self.category!r}; "
                f"expected one of {', '.join(valid_categories)}"
            )
        if not self.runner_path().is_file():
            raise FileNotFoundError(f"BFCL external runner not found: {self.runner_path()}")
        if self.run_all and (self.task_ids or self.task_ids_file is not None):
            raise ValueError("run_all cannot be combined with task_ids or task_ids_file")

    def command(self) -> list[str]:
        self.validate()
        cmd = [
            self.python_bin,
            str(self.runner_path()),
            "--model",
            self.model,
            "--category",
            self.category,
            "--output-dir",
            str(self.output_dir),
        ]
        if self.run_all:
            cmd.append("--all")
        if self.num_tasks is not None:
            cmd.extend(["--num-tasks", str(self.num_tasks)])
        if self.task_ids:
            cmd.extend(["--ids", ",".join(self.task_ids)])
        if self.task_ids_file is not None:
            cmd.extend(["--ids-file", str(self.task_ids_file)])
        if self.workers is not None:
            cmd.extend(["--workers", str(self.workers)])
        if self.max_retries is not None:
            cmd.extend(["--max-retries", str(self.max_retries)])
        if self.resume:
            cmd.append("--resume")
        if self.mode == "multi":
            for schema_dir in self.multi_turn_schema_dirs:
                cmd.extend(["--multi-turn-schema-dir", str(schema_dir)])
        if self.debug:
            cmd.append("--debug")
        if self.verbose:
            cmd.append("--verbose")
        cmd.extend(self.extra_args)
        return cmd


def write_external_bfcl_run_manifest(
    config: ExternalBFCLRunConfig,
    *,
    command: list[str],
    status: str,
    dry_run: bool,
) -> Path:
    """Record the external BFCL runner provenance for one arm directory."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "bfcl_external_run_manifest_v1",
        "runner_type": "bfcl_external",
        "status": status,
        "dry_run": dry_run,
        "uses_legacy_trajectory2gym_env": False,
        "external_root": str(config.external_root),
        "runner_path": str(config.runner_path()),
        "model": config.model,
        "output_dir": str(config.output_dir),
        "mode": config.mode,
        "category": config.category,
        "run_all": config.run_all,
        "num_tasks": config.num_tasks,
        "task_ids": list(config.task_ids),
        "task_ids_file": str(config.task_ids_file) if config.task_ids_file is not None else None,
        "workers": config.workers,
        "max_retries": config.max_retries,
        "resume": config.resume,
        "multi_turn_schema_dirs": [str(path) for path in config.multi_turn_schema_dirs],
        "debug": config.debug,
        "verbose": config.verbose,
        "extra_args": list(config.extra_args),
        "command": command,
        "command_sha256": hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest(),
    }
    path = config.output_dir / BFCL_EXTERNAL_RUN_MANIFEST
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_score(payload: Any) -> tuple[int | None, int | None, float | None]:
    """Best-effort extraction from BFCL external JSON score artifacts."""

    if not isinstance(payload, dict):
        return None, None, None
    candidates = [payload]
    for key in ("summary", "overall", "metrics", "score", "result", "accuracy"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)

    correct = None
    total = None
    accuracy = None
    for item in candidates:
        for key in ("correct", "correct_count", "success", "success_count", "passed"):
            value = _numeric(item.get(key))
            if value is not None:
                correct = int(value)
                break
        for key in ("total", "total_count", "num_tasks", "count"):
            value = _numeric(item.get(key))
            if value is not None:
                total = int(value)
                break
        for key in ("accuracy", "score", "pass_rate", "success_rate"):
            value = _numeric(item.get(key))
            if value is not None:
                accuracy = float(value)
                break
        if correct is not None or total is not None or accuracy is not None:
            break
    if accuracy is None and correct is not None and total:
        accuracy = correct / total
    if accuracy is not None and accuracy > 1.0:
        accuracy = accuracy / 100.0
    return correct, total, accuracy


def summarize_external_bfcl_results(paths: list[Path]) -> dict[str, Any]:
    files: list[Path] = []
    manifests = [_run_manifest_provenance(path) for path in paths]
    manifests = [item for item in manifests if item is not None]
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
    rows = []
    for path in sorted(set(files)):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        correct, total, accuracy = _extract_score(payload)
        if correct is None and total is None and accuracy is None:
            continue
        rows.append(
            {
                "path": str(path),
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            }
        )
    weighted_correct = sum(row["correct"] for row in rows if row["correct"] is not None)
    weighted_total = sum(row["total"] for row in rows if row["total"] is not None)
    accuracies = [row["accuracy"] for row in rows if row["accuracy"] is not None]
    return {
        "format": "bfcl_external_result_summary_v1",
        "files": rows,
        "file_count": len(rows),
        "correct": weighted_correct if weighted_total else None,
        "total": weighted_total if weighted_total else None,
        "accuracy": (weighted_correct / weighted_total if weighted_total else None),
        "mean_file_accuracy": (sum(accuracies) / len(accuracies) if accuracies else None),
        "run_manifest": manifests[0] if manifests else None,
        "run_manifests": manifests,
    }


def _run_manifest_provenance(path: Path) -> dict[str, Any] | None:
    manifest_path = path / BFCL_EXTERNAL_RUN_MANIFEST if path.is_dir() else None
    if manifest_path is None or not manifest_path.is_file():
        return None
    payload = _load_json(manifest_path)
    if not isinstance(payload, dict):
        return None
    return {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "mtime_ns": manifest_path.stat().st_mtime_ns,
        "format": payload.get("format"),
        "runner_type": payload.get("runner_type"),
        "status": payload.get("status"),
        "dry_run": payload.get("dry_run"),
        "uses_legacy_trajectory2gym_env": payload.get("uses_legacy_trajectory2gym_env"),
        "external_root": payload.get("external_root"),
        "runner_path": payload.get("runner_path"),
        "mode": payload.get("mode"),
        "category": payload.get("category"),
        "command_sha256": payload.get("command_sha256"),
    }

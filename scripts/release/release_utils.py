"""Offline publication checks. Findings never include matching secret values."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

MANIFEST = "RELEASE_MANIFEST.json"
LITERAL_REVIEW = "NATIVE_LITERAL_REVIEW.json"
REVIEWABLE_RULES = {"private_network", "personal_or_infrastructure_path"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
             "build", "dist", "outputs", "checkpoints", "logs", "artifacts"}
SECRET_NAMES = {"credentials.json", "id_rsa", "id_ed25519", "ssh_config"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".toml", ".yaml", ".yml", ".txt", ".sh"}
SPECIAL_NAMES = {"LICENSE", "NOTICE", ".gitignore", ".gitattributes"}
RULES = {
    "credential_token": re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|olp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "internal_host": re.compile(r"(?:code\.alipay\.com|yuque\.antfin\.com|aicloud-intl\.com|antglobal\.com)", re.I),
    "personal_or_infrastructure_path": re.compile(r"/(?:home|Users|upfs_[A-Za-z_]+|mnt/a6[^/]*)/[A-Za-z0-9_.-]+"),
    "private_network": re.compile(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"),
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def parse_literal_review(review: dict, source_sha256: str) -> dict[str, set[str]]:
    if (review.get("format") != "trajectory2gym_native_literal_review_v1"
            or review.get("source_input_sha256") != source_sha256
            or not re.fullmatch(r"[a-f0-9]{64}", source_sha256)
            or not review.get("rationale")):
        raise ValueError("Literal review must be bound to the frozen source hash")
    values = review.get("allowed_sha256_by_rule", {})
    if not isinstance(values, dict) or set(values) - REVIEWABLE_RULES:
        raise ValueError("Credentials and internal hosts cannot be exempted")
    for hashes in values.values():
        if not isinstance(hashes, list) or any(not re.fullmatch(r"[a-f0-9]{64}", h) for h in hashes):
            raise ValueError("Only explicit literal SHA-256 values can be reviewed")
    return {name: set(hashes) for name, hashes in values.items()}


def is_reviewed_literal(rule: str, value: str, reviewed: dict) -> bool:
    return rule in REVIEWABLE_RULES and hashlib.sha256(value.encode()).hexdigest() in reviewed.get(rule, set())


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts
            or str(path) != name or not re.fullmatch(r"[A-Za-z0-9_./-]+", name)):
        raise ValueError("Unsafe release-relative filename")
    return name


def files_for_release(root: Path, kind: str = "code") -> list[Path]:
    if kind not in {"code", "dataset"}:
        raise ValueError("Unknown release kind")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Release root must be a real directory")
    root = root.resolve()
    result = []
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            if (Path(current) / name).is_symlink():
                raise ValueError("Symlink directories cannot enter a release")
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for name in sorted(files):
            path = Path(current) / name
            if name == ".DS_Store":
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError("Only regular files can enter a release")
            relative = safe_name(path.relative_to(root).as_posix())
            if (name == ".env" or name.startswith(".env.") or name in SECRET_NAMES
                    or path.suffix in {".key", ".pem", ".p12"}):
                raise ValueError("Credential/configuration file is not publishable: " + relative)
            if kind == "dataset":
                allowed = path.suffix in {".md", ".json", ".jsonl"} or name.endswith(".jsonl.gz") or name in {"LICENSE", "NOTICE", ".gitattributes"}
            else:
                allowed = path.suffix in TEXT_SUFFIXES or name in SPECIAL_NAMES
            if not allowed:
                raise ValueError("Unsupported release file: " + relative)
            result.append(path)
    return sorted(result)


def scan(root: Path, kind: str = "code") -> dict:
    if root.is_symlink():
        raise ValueError("Release root cannot be a symlink")
    root = root.resolve()
    findings, checked, reviewed_count = [], 0, 0
    release_files = files_for_release(root, kind)
    reviewed = {}
    if kind == "dataset" and (root / LITERAL_REVIEW).exists():
        if (root / LITERAL_REVIEW).is_symlink() or (root / "provenance.json").is_symlink():
            raise ValueError("Review metadata cannot be a symlink")
        provenance = json.loads((root / "provenance.json").read_text())
        reviewed = parse_literal_review(json.loads((root / LITERAL_REVIEW).read_text()),
                                        provenance.get("source_input_sha256", ""))
    for path in release_files:
        opener = gzip.open if path.name.endswith(".gz") else open
        checked += 1
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                for code, pattern in RULES.items():
                    matches = list(pattern.finditer(line))
                    reviewed_count += sum(is_reviewed_literal(code, m.group(), reviewed) for m in matches)
                    if any(not is_reviewed_literal(code, m.group(), reviewed) for m in matches):
                        findings.append({"file": path.relative_to(root).as_posix(),
                                         "line": line_no, "rule": code})
    return {"status": "pass" if not findings else "needs_review",
            "checked_files": checked, "findings": findings,
            "reviewed_native_literal_occurrences": reviewed_count,
            "scope": "Pattern scan only; not a license, PII, or data-quality certification."}


def create_manifest(root: Path, kind: str) -> dict:
    if root.is_symlink():
        raise ValueError("Release root cannot be a symlink")
    root = root.resolve()
    audit = scan(root, kind)
    if audit["status"] != "pass":
        raise ValueError(json.dumps(audit))
    records = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
                "sha256": sha256(p)} for p in files_for_release(root, kind) if p.name != MANIFEST]
    if not records:
        raise ValueError("Cannot publish an empty release")
    return {"format": "trajectory2gym_release_v1", "kind": kind, "files": records,
            "automated_scan": audit, "public_data_review_required": kind == "dataset"}


def validate_manifest(root: Path) -> dict:
    if root.is_symlink():
        raise ValueError("Release root cannot be a symlink")
    root = root.resolve()
    target = root / MANIFEST
    if target.is_symlink():
        raise ValueError("Manifest cannot be a symlink")
    manifest = json.loads(target.read_text(encoding="utf-8"))
    if manifest.get("format") != "trajectory2gym_release_v1":
        raise ValueError("Unrecognized release manifest")
    kind = manifest.get("kind")
    actual = {p.relative_to(root).as_posix() for p in files_for_release(root, kind)}
    declared = [safe_name(r["path"]) for r in manifest["files"]]
    if not declared or len(set(declared)) != len(declared):
        raise ValueError("Manifest must list unique files")
    if actual != set(declared) | {MANIFEST}:
        raise ValueError("Directory contents do not match the exact manifest")
    for row in manifest["files"]:
        path = root / row["path"]
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError("Manifest integrity mismatch: " + row["path"])
    audit = scan(root, kind)
    if audit["status"] != "pass":
        raise ValueError(json.dumps(audit))
    return manifest

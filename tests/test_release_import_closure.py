import ast
import hashlib
import json
from pathlib import Path

from trajectory2gym.training.probe_contract import (
    FULL_SUCCESSOR_BINDING_VERSION, full_successor_binding_sha256,
)


def test_internal_absolute_imports_have_packaged_modules():
    root = Path(__file__).resolve().parents[1]
    missing = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0
                       else [])
            for module in modules:
                if module and module.split(".")[0] in {"trajectory2gym", "env"}:
                    target = root.joinpath(*module.split("."))
                    if not target.with_suffix(".py").exists() and not (target / "__init__.py").exists():
                        missing.append(module)
    assert not missing


def test_evidence_binding_hash_preserves_the_canonical_payload():
    payload = {"pre_state_id": "a", "post_state_id": "b", "entry_context_id": "c",
               "next_policy_context_id": "d", "action_key": "e", "observation_key": "f"}
    expected = hashlib.sha256(json.dumps(
        {**payload, "version": FULL_SUCCESSOR_BINDING_VERSION},
        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert full_successor_binding_sha256(**payload) == expected
    assert full_successor_binding_sha256(**{**payload, "observation_key": "other"}) != expected

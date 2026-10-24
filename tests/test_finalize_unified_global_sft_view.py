from pathlib import Path


def test_finalizer_requires_exact_parent_mass_and_zero_truncation() -> None:
    text = Path("scripts/analysis/finalize_unified_global_sft_view.py").read_text()
    assert 'projection.get("effective_supervised_decisions_preserved") is not True' in text
    assert '"exact_token_count_conservation_required": False' in text
    assert 'int(audit.get("truncated_rows", -1)) == 0' in text
    assert '"required_multiscale": 12_000_000' in text
    assert '"model_boundary_v1": 8_000_000' in text
    assert '"capability_aligned_v1": 8_000_000' in text
    assert 'in {"4b", "8b", "14b", "30b-a3b"}' in text
    assert "global selection readiness gate did not pass" in text
    assert 'temporary / "selection_readiness.json"' in text
    assert '(temporary / "DATA_READY").touch()' in text

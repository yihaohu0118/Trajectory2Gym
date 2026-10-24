from pathlib import Path


def test_candidate_pool_builder_uses_preselection_mode() -> None:
    text = Path("scripts/analysis/rebuild_certified_candidate_pool.py").read_text()
    assert 'selection_mode="candidate_pool"' in text
    assert "candidate_pool_reconstructed_before_learning_value_selection" in text
    assert "validation_fraction=0.0" in text

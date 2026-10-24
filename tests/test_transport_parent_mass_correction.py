from pathlib import Path


def test_transport_parent_mass_builder_is_benchmark_neutral() -> None:
    text = Path("scripts/analysis/apply_transport_parent_mass_correction.py").read_text()
    assert 'FIELD = "transport_parent_loss_weight"' in text
    assert "token_by_trace[trace] / tokens_by_parent[parent]" in text
    assert '"benchmark_or_category_feature_used": False' in text
    assert '"maximum_parent_mass_error"' in text

from scripts.analysis.build_targeted_active_synthesis_overlay import overlay


def row(trace, content):
    return {
        "trace_hash": trace,
        "messages": [{"role": "assistant", "content": content}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "supervised_assistant_turn_indices": [0],
        "verified_assistant_turn_indices": [],
        "success": True,
        "enable_thinking": False,
        "domain": "tau1/retail",
    }


def test_overlay_preserves_base_and_adds_only_semantic_novel_rows():
    base = row("base", "a")
    duplicate = row("dup", "a")
    novel = row("novel", "b")
    selected, audit = overlay([base], [duplicate, novel])
    assert [item["trace_hash"] for item in selected] == ["base", "novel"]
    assert audit["base_rows_removed"] == 0
    assert audit["selected_addition_rows"] == 1
    assert audit["duplicate_semantic_additions"] == 1
    assert audit["official_evaluation_information_used"] is False


def test_cli_supports_exact_trace_exclusion_only():
    from pathlib import Path
    text = Path("scripts/analysis/build_targeted_active_synthesis_overlay.py").read_text()
    assert '"--exclude-trace"' in text
    assert 'excluded_traces <= available_traces' in text

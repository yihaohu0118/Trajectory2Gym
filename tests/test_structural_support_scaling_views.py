import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/analysis/select_structural_support_scaling_views.py"
SPEC = importlib.util.spec_from_file_location("structural_support", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

from trajectory2gym.training.global_sft_selection import prepare_candidates


def row(trace, task, assistant_turns, *, corpus="tau1", gap=None):
    messages = [{"role": "user", "content": f"request {task}"}]
    supervised = []
    for index, text in enumerate(assistant_turns):
        messages.append({"role": "assistant", "content": text})
        supervised.append(index)
        if index + 1 < len(assistant_turns):
            messages.append({"role": "user", "content": "continue"})
    result = {
        "trace_hash": trace,
        "source_task_id": task,
        "source_corpus": corpus,
        "domain": f"{corpus}/domain",
        "success": True,
        "enable_thinking": False,
        "messages": messages,
        "tools": [],
        "supervised_assistant_turn_indices": supervised,
        "verified_assistant_turn_indices": [],
    }
    if gap is not None:
        result["multi_scale_policy_gap"] = gap
    return result


def candidates(rows):
    return prepare_candidates(
        rows,
        supervised_tokens={item["trace_hash"]: 10 + index for index, item in enumerate(rows)},
        require_multiscale_policy_gap=False,
        loop_redundancy_mode="certified_exact_adjacent_v2",
        path_family_mode="capability_path_v2",
    )[0]


def test_reference_task_and_path_support_are_nested():
    rows = [
        row("r1", "task-a", ["done"]),
        row("r2", "task-a", ["done again"]),
        row("r3", "task-a", ["done third"]),
        row("r4", "task-b", ["first", "second"]),
        row("r5", "task-b", ["first alt", "second alt"]),
        row("ignored", "task-c", ["x"], corpus="tau3"),
    ]
    active = [item for item in rows if item["source_corpus"] != "tau3"]
    views, audits = MODULE.select_views(
        candidates(active),
        reference_traces={"r1"},
        scale_depths={"4b": 1, "8b": 2, "14b": 3},
        max_per_task=8,
    )
    trace_sets = [{item["trace_hash"] for item in views[scale]} for scale in views]
    assert trace_sets[0] <= trace_sets[1] <= trace_sets[2]
    assert "r1" in trace_sets[0]
    assert {item["source_task_id"] for item in views["4b"]} == {"task-a", "task-b"}
    assert all(audits[scale]["nll_used"] is False for scale in audits)
    assert len(trace_sets[2]) >= len(trace_sets[1]) >= len(trace_sets[0])


def test_nll_like_policy_gap_cannot_change_selection():
    base = [
        row("r1", "task-a", ["one"], gap=0.0),
        row("r2", "task-a", ["two"], gap=0.0),
        row("r3", "task-b", ["first", "second"], gap=0.0),
    ]
    changed = [{**item, "multi_scale_policy_gap": 999.0 - index} for index, item in enumerate(base)]
    first, _ = MODULE.select_views(
        candidates(base),
        reference_traces={"r1"},
        scale_depths={"4b": 2},
        max_per_task=8,
    )
    second, _ = MODULE.select_views(
        candidates(changed),
        reference_traces={"r1"},
        scale_depths={"4b": 2},
        max_per_task=8,
    )
    assert [item["trace_hash"] for item in first["4b"]] == [
        item["trace_hash"] for item in second["4b"]
    ]

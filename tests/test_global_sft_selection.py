from __future__ import annotations

import copy

import pytest

from trajectory2gym.training.global_sft_selection import (
    capability_atoms,
    capability_path_signature,
    prepare_candidates,
    select_global_candidates,
)


def row(trace: str, task: str, *, call: str = "get_item", error: bool = False) -> dict:
    observation = "Error: unavailable" if error else '{"value": 1}'
    return {
        "format": "trajectory2gym_sft_v0",
        "split": "train",
        "success": True,
        "enable_thinking": False,
        "domain": "benchmark/private_category",
        "source_task_id": task,
        "task_id": f"{task}:{trace}",
        "trace_hash": trace,
        "trajectory_family_id": task,
        "source": "teacher-a",
        "supervised_assistant_turn_indices": [0],
        "messages": [
            {"role": "user", "content": "Find item 1"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": call, "arguments": '{"item_id": "1"}'}}
                ],
            },
            {"role": "tool", "content": observation},
        ],
        "tools": [{"type": "function", "function": {"name": call, "parameters": {}}}],
        "sft_loss_weight": 9.0,
    }


def test_atoms_do_not_contain_benchmark_category_or_exact_tool_name() -> None:
    atoms = capability_atoms(row("a" * 64, "task-a", call="get_private_widget"))
    rendered = "\n".join(sorted(atoms))
    assert "benchmark" not in rendered
    assert "private_category" not in rendered
    assert "get_private_widget" not in rendered
    assert "access:read" in atoms


def test_path_signature_is_ordered_and_excludes_exact_values_and_provenance() -> None:
    first = row("a" * 64, "task-a", call="get_private_widget")
    renamed = copy.deepcopy(first)
    renamed["domain"] = "other_benchmark/other_category"
    renamed["source_task_id"] = "other-task"
    renamed["task_id"] = "other-task:trace"
    renamed["messages"][0]["content"] = "Find item 2"
    renamed["messages"][1]["tool_calls"][0]["function"]["arguments"] = (
        '{"item_id": "2"}'
    )
    assert capability_path_signature(first) == capability_path_signature(renamed)

    different_successor = copy.deepcopy(first)
    different_successor["messages"][2]["content"] = '[{"value": 1}]'
    assert capability_path_signature(first) != capability_path_signature(
        different_successor
    )


def test_global_selection_is_invariant_to_domain_renaming() -> None:
    rows = [
        row("a" * 64, "task-a", call="get_item"),
        row("b" * 64, "task-b", call="update_item"),
    ]
    tokens = {"a" * 64: 10, "b" * 64: 10}
    first, _ = prepare_candidates(rows, supervised_tokens=tokens)
    renamed = copy.deepcopy(rows)
    for item in renamed:
        item["domain"] = "different_benchmark/other_cell"
    second, _ = prepare_candidates(renamed, supervised_tokens=tokens)
    selected_first, audit_first, _ = select_global_candidates(first, token_budget=10)
    selected_second, audit_second, _ = select_global_candidates(second, token_budget=10)
    assert [item["trace_hash"] for item in selected_first] == [
        item["trace_hash"] for item in selected_second
    ]
    assert audit_first["benchmark_or_category_feature_used"] is False
    assert audit_second["benchmark_or_category_feature_used"] is False


def test_selector_removes_source_loss_weights_and_enforces_budget() -> None:
    rows = [
        row("a" * 64, "task-a", call="get_item"),
        row("b" * 64, "task-b", call="update_item"),
    ]
    tokens = {"a" * 64: 8, "b" * 64: 8}
    candidates, prep = prepare_candidates(rows, supervised_tokens=tokens)
    selected, audit, ledger = select_global_candidates(candidates, token_budget=8)
    assert len(selected) == 1
    assert audit["selected_supervised_tokens"] == 8
    assert all("sft_loss_weight" not in item for item in selected)
    assert prep["stripped_weight_fields"] == {"sft_loss_weight": 2}
    assert sum(bool(item["selected"]) for item in ledger) == 1


def test_coverage_closure_is_a_hard_source_neutral_first_phase() -> None:
    rows = [
        row("a" * 64, "task-a", call="get_item"),
        row("b" * 64, "task-b", call="update_item"),
    ]
    tokens = {"a" * 64: 10, "b" * 64: 10}
    candidates, _ = prepare_candidates(rows, supervised_tokens=tokens)
    selected, audit, ledger = select_global_candidates(
        candidates, token_budget=20, coverage_closure=True
    )
    assert len(selected) == 2
    assert audit["coverage_closure"] is True
    assert audit["coverage_closure_atoms"] == audit["capability_atoms_total"]
    assert audit["coverage_closure_selected_rows"] > 0
    assert all(item["phase"] == "capability_coverage_closure" for item in ledger if item["selected"])


def test_task_representative_closure_uses_one_universal_minimum() -> None:
    rows = [row("a" * 64, "task-a"), row("b" * 64, "task-a"),
            row("c" * 64, "task-b", call="update_item")]
    tokens = {item["trace_hash"]: 10 for item in rows}
    candidates, _ = prepare_candidates(rows, supervised_tokens=tokens)
    selected, audit, ledger = select_global_candidates(
        candidates, token_budget=30, minimum_per_task=1
    )
    assert {item["source_task_id"] for item in selected} == {"task-a", "task-b"}
    assert audit["task_universe"] == audit["tasks_represented"] == 2
    assert audit["task_closure_selected_rows"] == 2
    assert sum(item.get("phase") == "task_representative_closure" for item in ledger) == 2


def test_minimal_closure_does_not_fill_remaining_token_budget() -> None:
    rows = [row("a" * 64, "task-a"), row("b" * 64, "task-a"),
            row("c" * 64, "task-b", call="update_item"),
            row("d" * 64, "task-b", call="update_item")]
    tokens = {item["trace_hash"]: 4 for item in rows}
    candidates, _ = prepare_candidates(rows, supervised_tokens=tokens)
    selected, audit, _ = select_global_candidates(
        candidates, token_budget=100, minimum_per_task=1,
        coverage_closure=True, fill_remaining=False,
    )
    assert len(selected) == 2
    assert audit["fill_remaining"] is False
    assert audit["unused_token_budget"] == 92


def test_unknown_teacher_is_not_collapsed_when_explicit_provenance_is_required() -> None:
    rows = [row(f"{i:064x}", "task-a") for i in range(6)]
    for item in rows:
        item["source"] = "same-source-without-teacher-provenance"
        item["messages"][0]["content"] += item["trace_hash"]
    tokens = {item["trace_hash"]: 1 for item in rows}
    candidates, _ = prepare_candidates(rows, supervised_tokens=tokens)
    selected, audit, _ = select_global_candidates(
        candidates, token_budget=20, max_per_task=8, max_per_task_teacher=4,
        minimum_per_task=6, fill_remaining=False,
        teacher_cap_requires_explicit_provenance=True,
    )
    assert len(selected) == 6
    assert audit["candidates_with_explicit_teacher_provenance"] == 0


def test_semantic_trajectory_cap_is_hard() -> None:
    rows = [row(f"{i:064x}", f"task-{i}") for i in range(3)]
    for item in rows:
        item["messages"][0]["content"] += item["source_task_id"]
    tokens = {item["trace_hash"]: 1 for item in rows}
    candidates, _ = prepare_candidates(rows, supervised_tokens=tokens)
    selected, audit, _ = select_global_candidates(
        candidates, token_budget=20, minimum_per_task=1,
        max_selected_trajectories=2, fill_remaining=False,
    )
    assert len(selected) == 2
    assert audit["max_selected_trajectories"] == 2
    assert audit["tasks_represented"] == 2


def test_semantically_identical_rows_are_deduplicated_across_trace_ids() -> None:
    first = row("a" * 64, "task-a")
    second = copy.deepcopy(first)
    second["trace_hash"] = "b" * 64
    second["task_id"] = "task-a:b"
    candidates, audit = prepare_candidates(
        [first, second],
        supervised_tokens={"a" * 64: 8, "b" * 64: 8},
    )
    assert len(candidates) == 1
    assert audit["duplicate_semantic_rows_removed"] == 1


def test_capability_path_family_cap_is_task_and_benchmark_neutral() -> None:
    rows = [row(f"{index:064x}", f"task-{index}") for index in range(4)]
    for index, item in enumerate(rows):
        item["domain"] = f"benchmark-{index}/category-{index}"
        item["messages"][0]["content"] = f"Find item {index + 1}"
        item["messages"][1]["tool_calls"][0]["function"]["arguments"] = (
            '{"item_id": "%d"}' % (index + 1)
        )
    tokens = {item["trace_hash"]: 1 for item in rows}
    candidates, prep = prepare_candidates(
        rows,
        supervised_tokens=tokens,
        path_family_mode="capability_path_v2",
    )
    assert prep["unique_path_families"] == 1
    selected, audit, ledger = select_global_candidates(
        candidates,
        token_budget=20,
        max_per_capability_path=2,
    )
    assert len(selected) == 2
    assert audit["max_per_capability_path"] == 2
    assert audit["selected_capability_paths"] == 1
    assert sum(
        item.get("reason") == "capability_path_cap" for item in ledger
    ) == 2


def test_task_closure_precedes_capability_path_redundancy_cap() -> None:
    rows = [row(f"{index:064x}", f"task-{index}") for index in range(4)]
    for index, item in enumerate(rows):
        item["messages"][0]["content"] = f"Find item {index + 1}"
        item["messages"][1]["tool_calls"][0]["function"]["arguments"] = (
            '{"item_id": "%d"}' % (index + 1)
        )
    tokens = {item["trace_hash"]: 1 for item in rows}
    candidates, _ = prepare_candidates(
        rows,
        supervised_tokens=tokens,
        path_family_mode="capability_path_v2",
    )
    selected, audit, _ = select_global_candidates(
        candidates,
        token_budget=20,
        minimum_per_task=1,
        max_per_capability_path=1,
    )
    assert len(selected) == 4
    assert audit["task_universe"] == audit["tasks_represented"] == 4
    assert audit["capability_path_cap_applies_after_task_closure"] is True


def test_multiscale_gap_must_be_explicit_when_required() -> None:
    candidate = row("a" * 64, "task-a")
    with pytest.raises(ValueError, match="multi_scale_policy_gap"):
        prepare_candidates(
            [candidate],
            supervised_tokens={"a" * 64: 10},
            require_multiscale_policy_gap=True,
        )


def test_masked_error_followed_by_supervised_turn_is_recovery() -> None:
    candidate = row("c" * 64, "task-c")
    candidate["messages"] = [
        {"role": "user", "content": "Fix it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "update_item", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "Error: rejected", "error": True},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "get_item", "arguments": "{}"}}],
        },
        {"role": "tool", "content": '{"value": 1}'},
    ]
    candidate["supervised_assistant_turn_indices"] = [1]
    assert "recovery:after_error" in capability_atoms(candidate)


def test_error_action_must_remain_masked_after_recovery() -> None:
    candidate = row("a" * 64, "task-a", error=True)
    with pytest.raises(ValueError, match="error-producing"):
        prepare_candidates(
            [candidate], supervised_tokens={"a" * 64: 10}
        )

    candidate["messages"].extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "get_item", "arguments": '{"item_id": "2"}'}}
                ],
            },
            {"role": "tool", "content": '{"value": 2}'},
        ]
    )
    candidate["supervised_assistant_turn_indices"] = [1]
    prepared, _ = prepare_candidates(
        [candidate], supervised_tokens={"a" * 64: 10}
    )
    assert len(prepared) == 1

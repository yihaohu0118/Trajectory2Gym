"""CPU-only selection demonstration; all records and token counts are synthetic."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analysis.select_structural_support_scaling_views import select_views
from trajectory2gym.training.global_sft_selection import prepare_candidates


def main():
    rows = []
    for i, turns in enumerate((["done"], ["finished"], ["first", "second"], ["one", "two"])):
        messages = [{"role": "user", "content": "Synthetic task"}]
        for j, text in enumerate(turns):
            if j:
                messages.append({"role": "user", "content": "continue"})
            messages.append({"role": "assistant", "content": text})
        rows.append({"trace_hash": f"demo-{i}", "source_task_id": f"task-{len(turns)}",
                     "source_corpus": "synthetic", "domain": "synthetic/demo",
                     "success": True, "enable_thinking": False, "messages": messages,
                     "tools": [], "supervised_assistant_turn_indices": list(range(len(turns))),
                     "verified_assistant_turn_indices": []})
    candidates, _ = prepare_candidates(
        rows, supervised_tokens={r["trace_hash"]: 10 + i for i, r in enumerate(rows)},
        require_multiscale_policy_gap=False,
        loop_redundancy_mode="certified_exact_adjacent_v2", path_family_mode="capability_path_v2")
    views, _ = select_views(candidates, reference_traces={"demo-0"},
                            scale_depths={"depth-1": 1, "depth-2": 2}, max_per_task=8)
    selected = {key: [r["trace_hash"] for r in value] for key, value in views.items()}
    assert set(selected["depth-1"]) <= set(selected["depth-2"])
    print(json.dumps({"synthetic_demo_only": True, "selected": selected}, indent=2))


if __name__ == "__main__":
    main()

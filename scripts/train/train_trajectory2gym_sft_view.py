from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_jsonl(path: Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def domain_temperature_sampling_weights(
    rows: list[dict[str, Any]], *, alpha: float
) -> list[float]:
    """Interpolate between equal-domain and natural example sampling.

    The number of distinct benchmark tasks is often highly imbalanced (for
    example Tau1 Retail has ten times as many train tasks as Airline). Domain
    mass is proportional to ``count ** alpha``: alpha=0 gives equal mass,
    alpha=1 reproduces natural example-proportional sampling, and alpha=0.5 is
    the auditable square-root compromise. Within-domain rows stay uniform.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("domain sampling alpha must be in [0, 1]")
    domains = [str(row.get("domain") or "").strip() for row in rows]
    if any(not domain for domain in domains):
        raise ValueError("domain sampling requires domain on every row")
    counts = Counter(domains)
    return [float(counts[domain]) ** (alpha - 1.0) for domain in domains]


def domain_balanced_sampling_weights(rows: list[dict[str, Any]]) -> list[float]:
    """Backward-compatible alpha=0 equal-domain sampling weights."""

    return domain_temperature_sampling_weights(rows, alpha=0.0)


def parse_domain_sampling_masses(values: list[str]) -> dict[str, float]:
    """Parse repeated ``domain=mass`` declarations into positive masses."""

    masses: dict[str, float] = {}
    for value in values:
        domain, separator, raw_mass = value.partition("=")
        domain = domain.strip()
        if not separator or not domain or not raw_mass.strip():
            raise ValueError(
                "domain sampling mass must use non-empty domain=mass syntax"
            )
        if domain in masses:
            raise ValueError(f"duplicate domain sampling mass: {domain}")
        try:
            mass = float(raw_mass)
        except ValueError as exc:
            raise ValueError(f"invalid domain sampling mass: {value}") from exc
        if not mass > 0.0:
            raise ValueError(f"domain sampling mass must be positive: {value}")
        masses[domain] = mass
    return masses


def domain_target_sampling_weights(
    rows: list[dict[str, Any]], *, masses: dict[str, float]
) -> list[float]:
    """Give every declared domain its requested expected sampling mass.

    This is a generic curriculum control: all observed domains must be
    declared and receive positive mass.  A domain's mass is distributed
    uniformly over its rows, so dataset size cannot silently override the
    requested curriculum.
    """

    domains = [str(row.get("domain") or "").strip() for row in rows]
    if any(not domain for domain in domains):
        raise ValueError("domain sampling requires domain on every row")
    counts = Counter(domains)
    observed = set(counts)
    declared = set(masses)
    if declared != observed:
        missing = sorted(observed - declared)
        unknown = sorted(declared - observed)
        raise ValueError(
            "domain sampling masses must exactly cover observed domains: "
            f"missing={missing} unknown={unknown}"
        )
    if any(not float(mass) > 0.0 for mass in masses.values()):
        raise ValueError("domain sampling masses must all be positive")
    total = sum(float(mass) for mass in masses.values())
    normalized = {domain: float(mass) / total for domain, mass in masses.items()}
    return [normalized[domain] / counts[domain] for domain in domains]


def domain_coverage_capped_sampling_indices(
    rows: list[dict[str, Any]],
    *,
    repeat_cap: int,
    seed: int,
    task_balanced: bool = False,
) -> list[int]:
    """Cover every row once, then repeat small domains by a bounded amount.

    Equal-domain sampling with replacement can silently leave a large fraction
    of the biggest domain unseen while replaying a small domain many times.
    This schedule gives every selected trajectory one visit per epoch.  Smaller
    domains receive additional shuffled passes, capped at ``repeat_cap`` total
    visits per row and at the size of the largest domain.  The rule depends
    only on declared domain identity and is therefore benchmark-neutral.
    """

    if repeat_cap < 1:
        raise ValueError("domain coverage repeat cap must be at least 1")
    by_domain: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        domain = str(row.get("domain") or "").strip()
        if not domain:
            raise ValueError("domain coverage sampling requires domain on every row")
        by_domain.setdefault(domain, []).append(index)
    if not by_domain:
        return []
    largest = max(len(indices) for indices in by_domain.values())
    generator = random.Random(seed)
    schedule: list[int] = []
    for domain in sorted(by_domain):
        source = by_domain[domain]
        target = min(largest, repeat_cap * len(source))
        if not task_balanced:
            selected: list[int] = []
            while len(selected) < target:
                cycle = list(source)
                generator.shuffle(cycle)
                selected.extend(cycle[: target - len(selected)])
        else:
            # Start with a complete source-data pass.  Additional visits are
            # allocated to the least-exposed task first and to the
            # least-exposed row inside that task.  This prevents a task with
            # eight successful traces from receiving eight times the mass of
            # a task with one trace, without ever repeating a single row more
            # than ``repeat_cap`` times in one sampler epoch.
            by_task: dict[str, list[int]] = {}
            for index in source:
                task = str(rows[index].get("source_task_id") or "").strip()
                if not task:
                    raise ValueError(
                        "task-balanced domain coverage sampling requires "
                        "source_task_id on every row"
                    )
                by_task.setdefault(task, []).append(index)
            selected = list(source)
            row_visits = Counter(source)
            task_visits = Counter(
                {
                    task: sum(row_visits[index] for index in indices)
                    for task, indices in by_task.items()
                }
            )
            while len(selected) < target:
                candidates = [
                    task
                    for task, indices in by_task.items()
                    if any(row_visits[index] < repeat_cap for index in indices)
                ]
                if not candidates:
                    raise AssertionError(
                        "coverage schedule exhausted before declared target"
                    )
                least_task_visits = min(task_visits[task] for task in candidates)
                candidate_tasks = [
                    task
                    for task in candidates
                    if task_visits[task] == least_task_visits
                ]
                task = generator.choice(sorted(candidate_tasks))
                available = [
                    index
                    for index in by_task[task]
                    if row_visits[index] < repeat_cap
                ]
                least_row_visits = min(row_visits[index] for index in available)
                candidate_rows = [
                    index
                    for index in available
                    if row_visits[index] == least_row_visits
                ]
                index = generator.choice(sorted(candidate_rows))
                selected.append(index)
                row_visits[index] += 1
                task_visits[task] += 1
        schedule.extend(selected)
    generator.shuffle(schedule)
    return schedule


def summarize_sampling_exposure(
    rows: list[dict[str, Any]],
    *,
    epochs: float,
    max_steps: int,
    world_size: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    seed: int,
    domain_coverage_repeat_cap: int | None = None,
    domain_coverage_task_balanced: bool = False,
) -> dict[str, Any]:
    """Describe what the optimizer will actually see before training.

    ``num_train_epochs`` is silently ignored when ``max_steps`` is positive in
    Transformers.  That produced a historical checkpoint labelled as a
    multi-epoch SFT run even though it consumed only a small fraction of the
    selected trajectory corpus.  This audit makes the effective exposure and
    domain mass explicit and gives launchers a fail-closed contract.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if max_steps == 0 or max_steps < -1:
        raise ValueError("max_steps must be -1 or positive")
    if world_size <= 0 or per_device_train_batch_size <= 0:
        raise ValueError("world size and per-device batch size must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient accumulation steps must be positive")
    if not rows:
        raise ValueError("sampling exposure requires at least one row")

    if domain_coverage_repeat_cap is None:
        schedule = list(range(len(rows)))
        sampler = "natural"
    else:
        schedule = domain_coverage_capped_sampling_indices(
            rows,
            repeat_cap=domain_coverage_repeat_cap,
            seed=seed,
            task_balanced=domain_coverage_task_balanced,
        )
        sampler = "domain_coverage_capped"

    global_batch = (
        world_size
        * per_device_train_batch_size
        * gradient_accumulation_steps
    )
    if max_steps > 0:
        planned_presentations = max_steps * global_batch
        control = "max_steps"
    else:
        planned_presentations = math.ceil(len(schedule) * epochs)
        control = "epochs"

    full_cycles, partial = divmod(planned_presentations, len(schedule))
    planned_indices = schedule * full_cycles + schedule[:partial]
    raw_by_domain = Counter(str(row.get("domain") or "") for row in rows)
    if any(not domain for domain in raw_by_domain):
        raise ValueError("sampling exposure requires domain on every row")
    planned_by_domain = Counter(
        str(rows[index].get("domain") or "") for index in planned_indices
    )
    schedule_by_domain = Counter(
        str(rows[index].get("domain") or "") for index in schedule
    )
    row_visits = Counter(planned_indices)
    task_visits: Counter[str] = Counter()
    for index in planned_indices:
        task = str(rows[index].get("source_task_id") or "").strip()
        if task:
            task_visits[task] += 1

    domain_mass = {
        domain: planned_by_domain[domain] / planned_presentations
        for domain in sorted(raw_by_domain)
    }
    return {
        "format": "trajectory2gym_sft_exposure_audit_v1",
        "control": control,
        "source_rows": len(rows),
        "source_rows_by_domain": dict(sorted(raw_by_domain.items())),
        "sampler": sampler,
        "domain_coverage_repeat_cap": domain_coverage_repeat_cap,
        "domain_coverage_task_balanced": bool(
            domain_coverage_task_balanced
        ),
        "sampler_rows_per_epoch": len(schedule),
        "sampler_rows_per_epoch_by_domain": dict(
            sorted(schedule_by_domain.items())
        ),
        "declared_epochs": float(epochs),
        "declared_max_steps": int(max_steps),
        "global_optimization_batch_size": global_batch,
        "planned_optimizer_steps": math.ceil(
            planned_presentations / global_batch
        ),
        "planned_sample_presentations": planned_presentations,
        "planned_presentations_by_domain": dict(
            sorted(planned_by_domain.items())
        ),
        "planned_domain_mass": domain_mass,
        "base_dataset_equivalent_epochs": planned_presentations / len(rows),
        "sampler_equivalent_epochs": planned_presentations / len(schedule),
        "all_source_rows_seen": len(row_visits) == len(rows),
        "source_row_visit_min": min(row_visits.values(), default=0),
        "source_row_visit_max": max(row_visits.values(), default=0),
        "observed_source_tasks": len(
            {
                str(row.get("source_task_id") or "").strip()
                for row in rows
                if str(row.get("source_task_id") or "").strip()
            }
        ),
        "planned_source_tasks_seen": len(task_visits),
        "source_task_visit_min": min(task_visits.values(), default=0),
        "source_task_visit_max": max(task_visits.values(), default=0),
        "max_steps_overrides_epochs": max_steps > 0,
    }


def learning_value_core_tail_sampling_weights(
    rows: list[dict[str, Any]],
    *,
    core_rank: int,
    core_mass: float,
    domain_alpha: float = 0.0,
    domain_masses: dict[str, float] | None = None,
    task_balanced: bool = False,
) -> list[float]:
    """Mix high-value per-task cores with a non-zero broad coverage tail.

    ``learning_value`` curation assigns a deterministic rank inside each task.
    This sampler keeps every selected trajectory eligible, while allocating a
    declared fraction of each domain's mass to rows whose rank is inside the
    core.  Domain mass is resolved before the core/tail mixture, so a small
    domain cannot disappear merely because another domain has more traces.

    With ``task_balanced=True``, each source task receives equal expected mass
    inside its domain before the core/tail split.  This prevents easy tasks
    with many successful trajectories from drowning out rare successful
    demonstrations from hard tasks.  It is a data-coverage policy, not a
    benchmark-specific reward or environment rule.
    """

    if core_rank <= 0:
        raise ValueError("learning-value core rank must be positive")
    if not 0.0 <= core_mass < 1.0:
        raise ValueError("learning-value core mass must be in [0, 1)")
    domains = [str(row.get("domain") or "").strip() for row in rows]
    if any(not domain for domain in domains):
        raise ValueError("learning-value sampling requires domain on every row")
    counts = Counter(domains)
    if domain_masses is not None:
        # Reuse the strict validation contract for explicit target masses.
        domain_target_sampling_weights(rows, masses=domain_masses)
        total = sum(float(value) for value in domain_masses.values())
        target_mass = {
            domain: float(domain_masses[domain]) / total for domain in counts
        }
    else:
        if not 0.0 <= domain_alpha <= 1.0:
            raise ValueError("domain sampling alpha must be in [0, 1]")
        unnormalized = {
            domain: float(count) ** domain_alpha
            for domain, count in counts.items()
        }
        total = sum(unnormalized.values())
        target_mass = {
            domain: value / total for domain, value in unnormalized.items()
        }

    task_keys: list[tuple[str, str]] = []
    rows_by_task: Counter[tuple[str, str]] = Counter()
    tasks_by_domain: dict[str, set[tuple[str, str]]] = {
        domain: set() for domain in counts
    }
    core_by_domain: Counter[str] = Counter()
    core_by_task: Counter[tuple[str, str]] = Counter()
    is_core: list[bool] = []
    for row, domain in zip(rows, domains):
        curation = row.get("curation")
        if not isinstance(curation, dict) or curation.get("mode") != "learning_value":
            raise ValueError(
                "learning-value core-tail sampling requires learning_value curation"
            )
        rank = curation.get("rank_within_task")
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(
                "learning-value core-tail sampling requires positive rank_within_task"
            )
        source_task_id = row.get("source_task_id")
        if task_balanced and (
            source_task_id is None or not str(source_task_id).strip()
        ):
            raise ValueError(
                "task-balanced learning-value sampling requires source_task_id"
            )
        task_key = (domain, str(source_task_id))
        task_keys.append(task_key)
        rows_by_task[task_key] += 1
        tasks_by_domain[domain].add(task_key)
        core = rank <= core_rank
        is_core.append(core)
        core_by_domain[domain] += int(core)
        core_by_task[task_key] += int(core)
    missing_core = sorted(domain for domain in counts if not core_by_domain[domain])
    if missing_core:
        raise ValueError(
            f"learning-value core-tail sampling found no core rows: {missing_core}"
        )
    if task_balanced:
        missing_task_core = sorted(
            f"{domain}:{task_id}"
            for (domain, task_id), row_count in rows_by_task.items()
            if row_count and not core_by_task[(domain, task_id)]
        )
        if missing_task_core:
            raise ValueError(
                "task-balanced learning-value sampling found tasks without "
                f"core rows: {missing_task_core}"
            )

    weights: list[float] = []
    for domain, task_key, core in zip(domains, task_keys, is_core):
        if task_balanced:
            task_mass = target_mass[domain] / len(tasks_by_domain[domain])
            tail_weight = (
                task_mass * (1.0 - core_mass) / rows_by_task[task_key]
            )
            core_weight = (
                task_mass * core_mass / core_by_task[task_key]
                if core
                else 0.0
            )
        else:
            tail_weight = target_mass[domain] * (1.0 - core_mass) / counts[domain]
            core_weight = (
                target_mass[domain] * core_mass / core_by_domain[domain]
                if core
                else 0.0
            )
        weights.append(tail_weight + core_weight)
    return weights


def render_messages(
    messages: list[dict[str, Any]],
    tokenizer: Any | None = None,
    *,
    add_generation_prompt: bool = False,
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool | None = None,
) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": add_generation_prompt,
            }
            if tools:
                kwargs["tools"] = tools
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
            if isinstance(rendered, str):
                return rendered
        except Exception:
            pass
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "unknown")
        content = message.get("content", "")
        tool_calls = message.get("tool_calls")
        if tool_calls:
            content = f"{content}\n{json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)}".strip()
        parts.append(f"<|{role}|>\n{content}")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "\n".join(parts)


def row_to_text(row: dict[str, Any], tokenizer: Any | None = None) -> str:
    if isinstance(row.get("preformatted_prompt"), str) and isinstance(
        row.get("preformatted_completion"), str
    ):
        return (
            row["preformatted_prompt"]
            + row["preformatted_completion"]
            + str(row.get("preformatted_end") or "")
        )
    # ``prompt``/``response`` are the decision-level training contract and may
    # be deliberately transformed after ``messages`` was materialized.  Use
    # the explicit pair whenever present so prompt rendering and supervised
    # target rendering cannot silently read different versions of a row.
    # ``row_prompt_response_messages`` retains ``messages`` as the legacy
    # fallback for older corpora.
    prompt, response = row_prompt_response_messages(row)
    messages = prompt + response
    tools = row.get("tools")
    return render_messages(
        messages,
        tokenizer,
        tools=tools if isinstance(tools, list) else None,
        enable_thinking=row.get("enable_thinking"),
    )


def row_prompt_response_messages(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt = row.get("prompt")
    response = row.get("response")
    if isinstance(prompt, list) and isinstance(response, list):
        return list(prompt), list(response)
    if isinstance(prompt, list) and isinstance(response, dict):
        return list(prompt), [response]
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return list(messages[:-1]), [messages[-1]]
    return [], []


def row_to_prompt_text(row: dict[str, Any], tokenizer: Any | None = None) -> str:
    prompt, _ = row_prompt_response_messages(row)
    tools = row.get("tools")
    return render_messages(
        prompt,
        tokenizer,
        add_generation_prompt=True,
        tools=tools if isinstance(tools, list) else None,
        enable_thinking=row.get("enable_thinking"),
    )


def row_to_response_text(row: dict[str, Any], tokenizer: Any | None = None) -> str:
    _, response = row_prompt_response_messages(row)
    return render_messages(response, tokenizer)


def common_token_prefix(left: Any, right: Any) -> int:
    """Return the exact common-prefix length of two 1-D token tensors."""

    shared = min(int(left.numel()), int(right.numel()))
    index = 0
    while index < shared and int(left[index]) == int(right[index]):
        index += 1
    return index


def _scan_chatml_spans(
    ids: list[int],
    *,
    start_marker: list[int],
    end_marker: list[int],
) -> list[tuple[int, int]]:
    """Locate non-overlapping ChatML assistant bodies for one marker form."""

    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor <= len(ids) - len(start_marker):
        if ids[cursor : cursor + len(start_marker)] != start_marker:
            cursor += 1
            continue
        content_start = cursor + len(start_marker)
        end = content_start
        while (
            end <= len(ids) - len(end_marker)
            and ids[end : end + len(end_marker)] != end_marker
        ):
            end += 1
        if end > len(ids) - len(end_marker):
            raise RuntimeError("assistant ChatML turn has no closing marker")
        spans.append((content_start, end))
        cursor = end + len(end_marker)
    return spans


def _assistant_chatml_spans(
    ids: list[int], tokenizer: Any, *, expected_turns: int
) -> tuple[list[tuple[int, int]], list[int]]:
    """Return assistant spans despite BPE merges at the header/content seam.

    Normally ``<|im_start|>assistant\n`` tokenizes independently of the
    following content.  A leading newline in the assistant body can make a
    BPE tokenizer merge the header's newline with body whitespace, so the
    full token marker no longer occurs even though the rendered ChatML is
    valid.  Fall back to the role marker without its separator; when the
    separator remains an independent token, exclude it from the loss span.
    """

    end_marker = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    full_start = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    if not full_start or not end_marker:
        raise RuntimeError(
            "chat template exposes neither assistant masks nor ChatML markers"
        )
    spans = _scan_chatml_spans(
        ids, start_marker=full_start, end_marker=end_marker
    )
    if len(spans) == expected_turns:
        return spans, end_marker

    bare_start = tokenizer.encode(
        "<|im_start|>assistant", add_special_tokens=False
    )
    if not bare_start:
        return spans, end_marker
    spans = _scan_chatml_spans(
        ids, start_marker=bare_start, end_marker=end_marker
    )
    separator = tokenizer.encode("\n", add_special_tokens=False)
    if separator:
        corrected = []
        for start, end in spans:
            if ids[start : start + len(separator)] == separator:
                start += len(separator)
            elif start < end and tokenizer.decode(
                [ids[start]], skip_special_tokens=False
            ).startswith("\n"):
                # A BPE token can merge the ChatML header separator with
                # leading assistant-body newlines. Supervising that token
                # would also supervise a role-header byte unavailable as a
                # student target. Mask the whole mixed token; later body
                # tokens and the native stop marker remain supervised.
                start += 1
            corrected.append((start, end))
        spans = corrected
    return spans, end_marker


def response_only_tokenized_row(
    row: dict[str, Any], tokenizer: Any, *, max_length: int
) -> dict[str, Any]:
    """Serialize one decision exactly as a native inference call.

    Qwen3 appends its no-thinking generation prefix only when
    ``add_generation_prompt=True``. Rendering ``prompt + response`` in one
    pass therefore does *not* reproduce the bytes seen before an API-time
    decision. Build the example as ``generation_prefix + assistant_body``:
    the prefix comes from the inference template and the body comes from the
    tokenizer's canonical assistant serialization.
    """

    import torch

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("response-only SFT requires a non-empty messages list")
    if str(messages[-1].get("role") or "") != "assistant":
        raise RuntimeError("response-only SFT target must be the last assistant turn")

    prompt_text = row_to_prompt_text(row, tokenizer)
    canonical_full_text = row_to_text(row, tokenizer)
    prompt_ids = tokenizer(
        prompt_text, truncation=False, padding=False, return_tensors="pt"
    )["input_ids"].squeeze(0)
    canonical_ids = tokenizer(
        canonical_full_text, truncation=False, padding=False, return_tensors="pt"
    )["input_ids"].squeeze(0)

    ids = canonical_ids.tolist()
    canonical_prompt, canonical_response = row_prompt_response_messages(row)
    canonical_messages = [*canonical_prompt, *canonical_response]
    expected_assistant_turns = sum(
        str(message.get("role") or "") == "assistant"
        for message in canonical_messages
    )
    spans, _ = _assistant_chatml_spans(
        ids, tokenizer, expected_turns=expected_assistant_turns
    )
    if not spans:
        raise RuntimeError("canonical target has no assistant ChatML turn")
    body_start = spans[-1][0]
    response_ids = canonical_ids[body_start:]
    # Some collected trajectories persist the empty no-thinking marker inside
    # assistant content. The native generation prompt has already supplied
    # that marker, so remove exactly one duplicate from the target body.
    empty_thinking = tokenizer.encode(
        "<think>\n\n</think>\n\n", add_special_tokens=False
    )
    if (
        empty_thinking
        and response_ids.tolist()[: len(empty_thinking)] == empty_thinking
    ):
        response_ids = response_ids[len(empty_thinking) :]
    if int(response_ids.numel()) <= 0:
        raise RuntimeError("canonical target has no assistant response tokens")
    if int(response_ids.numel()) > max_length:
        raise RuntimeError(
            "SFT response exceeds max length: "
            f"response_tokens={int(response_ids.numel())} max_length={max_length}"
        )

    retained_prompt = min(
        int(prompt_ids.numel()), max_length - int(response_ids.numel())
    )
    input_ids = torch.cat((prompt_ids[-retained_prompt:], response_ids))
    labels = torch.cat(
        (
            torch.full((retained_prompt,), -100, dtype=torch.long),
            response_ids.clone(),
        )
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "prompt_tokens": retained_prompt,
    }


def assistant_only_tokenized_row(
    row: dict[str, Any], tokenizer: Any, *, max_length: int
) -> dict[str, Any]:
    """Tokenize one full trace and supervise its certified assistant decisions.

    Each assistant span is located using the tokenizer's own native chat
    template.  This avoids hand-written Qwen/tool-call serialization and gives
    full-trajectory SFT the same causal context that the agent sees at
    inference time. System, user, tool-observation, historical-assistant, and
    explicitly excluded decision tokens are masked.
    """

    import torch

    preformatted_prompt = row.get("preformatted_prompt")
    preformatted_completion = row.get("preformatted_completion")
    if isinstance(preformatted_prompt, str) or isinstance(
        preformatted_completion, str
    ):
        if not (
            isinstance(preformatted_prompt, str)
            and isinstance(preformatted_completion, str)
            and row.get("transport_protocol")
            == "bfcl_official_qwenfc_raw_completion_v1"
        ):
            raise RuntimeError("incomplete or unsupported preformatted SFT row")
        full_text = (
            preformatted_prompt
            + preformatted_completion
            + str(row.get("preformatted_end") or "")
        )
        prompt_ids = tokenizer.encode(
            preformatted_prompt, add_special_tokens=False
        )
        full_ids_list = tokenizer.encode(full_text, add_special_tokens=False)
        if full_ids_list[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError("preformatted prompt is not a token-exact prefix")
        if len(full_ids_list) > max_length:
            raise RuntimeError(
                "official transport SFT row exceeds max length: "
                f"tokens={len(full_ids_list)} max_length={max_length}"
            )
        if len(full_ids_list) <= len(prompt_ids):
            raise RuntimeError("preformatted SFT row has no completion tokens")
        full_ids = torch.tensor(full_ids_list, dtype=torch.long)
        labels = torch.full_like(full_ids, -100)
        labels[len(prompt_ids) :] = full_ids[len(prompt_ids) :]
        return {
            "input_ids": full_ids,
            "attention_mask": torch.ones_like(full_ids),
            "labels": labels,
            "prompt_tokens": len(prompt_ids),
        }

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("assistant-only SFT requires a non-empty messages list")
    # Prefer a tokenizer-provided assistant mask when its template declares
    # generation spans.  Qwen3's stock template currently does not, so retain
    # an exact ChatML boundary fallback below.
    supports_native_mask = "{% generation" in str(
        getattr(tokenizer, "chat_template", "") or ""
    )
    template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
    }
    if supports_native_mask:
        template_kwargs["return_assistant_tokens_mask"] = True
    tools = row.get("tools")
    if isinstance(tools, list) and tools:
        template_kwargs["tools"] = tools
    if "enable_thinking" in row:
        template_kwargs["enable_thinking"] = bool(row["enable_thinking"])
    templated = tokenizer.apply_chat_template(messages, **template_kwargs)
    full_ids = torch.tensor(templated["input_ids"], dtype=torch.long)
    context_assistant_turns = int(row.get("context_assistant_turns") or 0)
    if context_assistant_turns < 0:
        raise RuntimeError("context_assistant_turns must be non-negative")
    supervised_turns_raw = row.get("supervised_assistant_turn_indices")
    supervised_turns: set[int] | None = None
    if supervised_turns_raw is not None:
        if not isinstance(supervised_turns_raw, list):
            raise RuntimeError("supervised_assistant_turn_indices must be a list")
        supervised_turns = {int(index) for index in supervised_turns_raw}
        if any(index < 0 for index in supervised_turns):
            raise RuntimeError(
                "supervised_assistant_turn_indices must be non-negative"
            )
    raw_mask = (
        templated.get("assistant_masks") or templated.get("assistant_tokens_mask")
        if supports_native_mask
        else None
    )
    # Native masks expose only the union of assistant spans.  When a trace
    # starts from a mid-dialogue frontier we must distinguish historical
    # assistant context from compiler-certified decisions, so use the exact
    # ChatML span path below.
    if (
        raw_mask
        and any(raw_mask)
        and context_assistant_turns == 0
        and supervised_turns is None
    ):
        assistant_mask = torch.tensor(raw_mask, dtype=torch.bool)
    else:
        ids = full_ids.tolist()
        expected_turns = sum(
            1
            for message in messages
            if str(message.get("role") or "") == "assistant"
        )
        spans, end_marker = _assistant_chatml_spans(
            ids, tokenizer, expected_turns=expected_turns
        )
        assistant_mask = torch.zeros_like(full_ids, dtype=torch.bool)
        for located_turns, (content_start, end) in enumerate(spans):
            # The model should learn both the assistant content/tool call and
            # its native stop token, but never role headers or observations.
            should_supervise = (
                located_turns >= context_assistant_turns
                if supervised_turns is None
                else located_turns in supervised_turns
            )
            if should_supervise:
                assistant_mask[content_start : end + len(end_marker)] = True
        located_turns = len(spans)
        if located_turns != expected_turns:
            raise RuntimeError(
                "assistant ChatML turn-count mismatch: "
                f"located={located_turns} expected={expected_turns}"
            )
        if context_assistant_turns >= located_turns:
            raise RuntimeError(
                "context_assistant_turns leaves no supervised policy turn: "
                f"context={context_assistant_turns} total={located_turns}"
            )
        if supervised_turns is not None:
            unknown = sorted(index for index in supervised_turns if index >= located_turns)
            if unknown:
                raise RuntimeError(
                    "supervised_assistant_turn_indices names absent turns: "
                    + ",".join(str(index) for index in unknown)
                )
    tool_only_raw = row.get("supervised_tool_call_only_turn_indices") or []
    if tool_only_raw:
        if not isinstance(tool_only_raw, list):
            raise RuntimeError("supervised_tool_call_only_turn_indices must be a list")
        tool_only = {int(index) for index in tool_only_raw}
        if len(tool_only) != len(tool_only_raw) or supervised_turns is None or not tool_only <= supervised_turns:
            raise RuntimeError("tool-call-only turns must be unique supervised assistant turns")
        assistant_messages = [message for message in messages if message.get("role") == "assistant"]
        if any(index < 0 or index >= len(assistant_messages)
               or not assistant_messages[index].get("tool_calls")
               or not str(assistant_messages[index].get("content") or "").strip()
               for index in tool_only):
            raise RuntimeError("tool-call-only target requires a mixed assistant action")
        rendered_kwargs = {"tokenize": False, "add_generation_prompt": False}
        if isinstance(tools, list) and tools:
            rendered_kwargs["tools"] = tools
        if "enable_thinking" in row:
            rendered_kwargs["enable_thinking"] = bool(row["enable_thinking"])
        rendered = tokenizer.apply_chat_template(messages, **rendered_kwargs)
        with_offsets = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        if with_offsets["input_ids"] != full_ids.tolist():
            raise RuntimeError("tool-call-only character offsets do not match ChatML tokenization")
        assistant_matches = [match for match in re.finditer(
            r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", rendered, re.DOTALL
        )]
        if len(assistant_matches) != len(assistant_messages):
            raise RuntimeError("tool-call-only assistant character-span mismatch")
        for ordinal in sorted(tool_only):
            match = assistant_matches[ordinal]
            tool_start = rendered.find("<tool_call>", match.start(1), match.end(1))
            if tool_start < 0:
                raise RuntimeError("mixed assistant target lacks rendered tool-call marker")
            removed = 0
            for index, (left, right) in enumerate(with_offsets["offset_mapping"]):
                if right > left and left >= match.start(1) and right <= tool_start:
                    removed += bool(assistant_mask[index])
                    assistant_mask[index] = False
            if not removed:
                raise RuntimeError("mixed assistant narration contributed no target tokens")
    labels = torch.where(assistant_mask, full_ids, torch.full_like(full_ids, -100))

    # Preserve the end of long traces, which contains the terminal policy
    # decisions, while retaining every assistant label that remains visible.
    if int(full_ids.numel()) > max_length:
        offset = int(full_ids.numel()) - max_length
        full_ids = full_ids[offset:]
        labels = labels[offset:]
    if not torch.any(labels != -100):
        raise RuntimeError("full trace has no supervised assistant tokens")
    return {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids),
        "labels": labels,
    }


def summarize_rows(rows: list[dict[str, Any]], *, tokenizer: Any | None = None, max_preview: int = 2) -> dict[str, Any]:
    by_mode: dict[str, int] = {}
    by_class: dict[str, int] = {}
    by_label: dict[str, int] = {}
    lengths: list[int] = []
    previews: list[dict[str, Any]] = []
    for row in rows:
        by_mode[str(row.get("mode", ""))] = by_mode.get(str(row.get("mode", "")), 0) + 1
        by_class[str(row.get("action_class", ""))] = by_class.get(str(row.get("action_class", "")), 0) + 1
        by_label[str(row.get("branch_label", ""))] = by_label.get(str(row.get("branch_label", "")), 0) + 1
        text = row_to_text(row, tokenizer)
        lengths.append(len(text))
        if len(previews) < max_preview:
            previews.append(
                {
                    "task_id": row.get("task_id"),
                    "mode": row.get("mode"),
                    "branch_label": row.get("branch_label"),
                    "chars": len(text),
                    "text_preview": text[:500],
                }
            )
    for counts in (by_mode, by_class, by_label):
        counts.pop("", None)
    return {
        "rows": len(rows),
        "by_mode": dict(sorted(by_mode.items())),
        "by_action_class": dict(sorted(by_class.items())),
        "by_branch_label": dict(sorted(by_label.items())),
        "hallucinated_observations": sum(1 for row in rows if row.get("hallucinated_observation")),
        "text_chars_min": min(lengths) if lengths else 0,
        "text_chars_max": max(lengths) if lengths else 0,
        "text_chars_mean": float(sum(lengths) / len(lengths)) if lengths else 0.0,
        "previews": previews,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def sparse_supervised_causal_lm_loss(
    model: Any,
    inputs: dict[str, Any],
    *,
    num_items_in_batch: Any | None = None,
    activation_offload_cpu: bool = False,
    activation_offload_pin_memory: bool = True,
    ce_checkpoint_chunk_size: int = 0,
) -> tuple[Any, Any]:
    """Compute causal-LM loss without materializing logits for masked tokens.

    Agent trajectories can contain tens of thousands of context tokens while
    assistant-only supervision applies to only a small set of action/answer
    tokens.  A normal causal-LM forward still projects every hidden state onto
    the full vocabulary; for a 29k-token Qwen3 trace that temporary logits
    tensor is larger than the model itself.  Qwen3's ``logits_to_keep`` API
    accepts arbitrary sequence positions, so retain the complete transformer
    context but project only positions that predict at least one supervised
    next token in the micro-batch.

    The union-of-positions construction is correct for batch sizes above one:
    targets not supervised in a particular row remain ``ignore_index``.  The
    returned model outputs intentionally contain sparse logits and therefore
    must not be used by a metric expecting dense per-token predictions.
    """

    import torch
    import torch.nn.functional as functional

    if "labels" not in inputs:
        raise ValueError("sparse supervised loss requires labels")
    labels = inputs["labels"]
    if labels.ndim != 2 or labels.shape[-1] < 2:
        raise ValueError("sparse supervised loss requires [batch, sequence] labels")
    shift_labels = labels[..., 1:].contiguous()
    positions = torch.nonzero(
        shift_labels.ne(-100).any(dim=0), as_tuple=False
    ).flatten()
    if positions.numel() == 0:
        raise ValueError("sparse supervised loss found no causal target tokens")

    loss_weights = inputs.get("loss_weights")
    model_inputs = {
        key: value
        for key, value in inputs.items()
        if key not in {"labels", "loss_weights"}
    }
    saved_tensor_context = (
        torch.autograd.graph.save_on_cpu(
            pin_memory=activation_offload_pin_memory
        )
        if activation_offload_cpu
        else nullcontext()
    )
    with saved_tensor_context:
        outputs = model(**model_inputs, logits_to_keep=positions)
    logits = outputs.logits
    targets = shift_labels.index_select(-1, positions)
    if logits.shape[:-1] != targets.shape:
        raise RuntimeError(
            "model logits_to_keep contract mismatch: "
            f"logits={tuple(logits.shape)} targets={tuple(targets.shape)}"
        )
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1).to(flat_logits.device)
    valid = targets.ne(-100)
    token_weights = torch.ones(targets.shape, device=logits.device, dtype=torch.float32)
    if loss_weights is not None:
        weights = torch.as_tensor(
            loss_weights, device=logits.device, dtype=torch.float32
        ).flatten()
        if weights.numel() != targets.shape[0] or not torch.isfinite(weights).all():
            raise ValueError("loss_weights must contain one finite value per row")
        if not torch.all(weights > 0):
            raise ValueError("loss_weights must be positive")
        token_weights = token_weights * weights[:, None]
    def chunk_ce(z, y, w):
        return (functional.cross_entropy(z.float(), y, ignore_index=-100, reduction='none') * w).sum()
    if ce_checkpoint_chunk_size > 0:
        from torch.utils.checkpoint import checkpoint
        flat_weights = token_weights.reshape(-1)
        losses = []
        for start in range(0, flat_targets.numel(), ce_checkpoint_chunk_size):
            stop = start + ce_checkpoint_chunk_size
            losses.append(checkpoint(chunk_ce, flat_logits[start:stop], flat_targets[start:stop],
                                     flat_weights[start:stop], use_reentrant=False))
        numerator = torch.stack(losses).sum()
    else:
        numerator = chunk_ce(flat_logits, flat_targets, token_weights.reshape(-1))
    if num_items_in_batch is None:
        denominator = valid.sum().to(device=numerator.device, dtype=numerator.dtype)
    else:
        denominator = torch.as_tensor(
            num_items_in_batch, device=numerator.device, dtype=numerator.dtype
        )
        if denominator.numel() != 1 or float(denominator.detach().cpu()) <= 0:
            raise ValueError("num_items_in_batch must be a positive scalar")
    loss = numerator / denominator
    return loss, outputs


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import Dataset, Sampler, SequentialSampler, WeightedRandomSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from scripts.train.certified_loss_contract import sparse_loss_mixin

    if args.fsdp_cpu_offload_unpinned:
        # PyTorch 2.6 FSDP pins multi-GiB CPU shards unconditionally. On the
        # current GCP H100 hosts that CUDA host-registration call returns
        # invalid argument before the first forward. Keep those exact tensor
        # bytes in pageable CPU memory; only the H2D transport changes.
        original_pin_memory = torch.Tensor.pin_memory

        def fsdp_pageable_shard(self, device=None):  # type: ignore[no-untyped-def]
            if device is not None and self.device.type == "cpu":
                return self
            if device is None:
                return original_pin_memory(self)
            return original_pin_memory(self, device=device)

        torch.Tensor.pin_memory = fsdp_pageable_shard

    loss_normalization = getattr(args, "loss_normalization", "legacy_local_token_mean")
    sync_each_microbatch = bool(getattr(args, "sync_each_microbatch", False))
    cuda_memory_fraction = getattr(args, 'cuda_memory_fraction', None)
    if cuda_memory_fraction is not None:
        if not 0 < cuda_memory_fraction <= 1:
            raise ValueError('cuda memory fraction must be in (0, 1]')
        device = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction, device)
    if loss_normalization == "global_batch_token_mean" and args.row_loss_weight_field:
        raise ValueError("global_batch_token_mean requires unweighted certified targets")
    if loss_normalization == "global_batch_parent_mean":
        if args.per_device_train_batch_size != 1 or not args.row_loss_weight_field:
            raise ValueError("global_batch_parent_mean requires one row and audited parent weights")
    if getattr(args, "ce_checkpoint_chunk_size", 0) < 0:
        raise ValueError("CE checkpoint chunk size must be nonnegative")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = load_jsonl(args.train_jsonl, max_rows=args.max_train_rows)
    val_rows = load_jsonl(args.val_jsonl, max_rows=args.max_val_rows)

    def row_loss_weight(row: dict[str, Any]) -> float:
        if not args.row_loss_weight_field:
            return 1.0
        value: Any = row
        for component in args.row_loss_weight_field.split("."):
            if not isinstance(value, dict) or component not in value:
                raise ValueError(
                    f"row is missing loss-weight field {args.row_loss_weight_field!r}"
                )
            value = value[component]
        weight = float(value)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("row loss weights must be finite and positive")
        return weight

    train_loss_weights = [row_loss_weight(row) for row in train_rows]

    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if args.fsdp_full_shard:
        if not distributed:
            raise ValueError("--fsdp-full-shard requires a distributed launch")
        if args.lora:
            raise ValueError(
                "--fsdp-full-shard is reserved for full-parameter SFT; "
                "omit --lora"
            )
    device_map = args.device_map
    if isinstance(device_map, str) and device_map.lower() in {"", "none", "null"}:
        device_map = None
    if distributed and device_map is not None:
        raise ValueError(
            "distributed SFT requires --device-map none so Trainer assigns "
            "one model replica to each local rank"
        )
    if args.fsdp_cpu_ram_efficient_loading:
        import torch.distributed as dist

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"
        os.environ["FSDP_SYNC_MODULE_STATES"] = "true"
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype_map[args.dtype],
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.model_loader == "causal_lm":
        model_loader = AutoModelForCausalLM
    else:
        from transformers import AutoModelForImageTextToText

        model_loader = AutoModelForImageTextToText
    model = model_loader.from_pretrained(args.model, **model_kwargs)
    frozen_vision_parameters = 0
    frozen_vision_parameter_names: list[str] = []
    if args.freeze_vision_tower:
        for name, parameter in model.named_parameters():
            if name.startswith("model.visual.") or ".visual." in name:
                parameter.requires_grad_(False)
                frozen_vision_parameters += parameter.numel()
                frozen_vision_parameter_names.append(name)
        if not frozen_vision_parameter_names:
            raise ValueError(
                "--freeze-vision-tower was requested but no visual parameters were found"
            )
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        checkpoint_kwargs = {"use_reentrant": False} if distributed else None
        if checkpoint_kwargs is None:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=checkpoint_kwargs
            )
        for config in (
            model.config,
            getattr(model.config, "text_config", None),
        ):
            if config is not None and hasattr(config, "use_cache"):
                config.use_cache = False

    if args.lora:
        if args.lora_init_adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, str(args.lora_init_adapter), is_trainable=True
            )
        else:
            from peft import LoraConfig, get_peft_model

            target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules or None,
            )
            model = get_peft_model(model, config)

    class SFTDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]):
            self.rows = rows
            self.loss_weights = [row_loss_weight(row) for row in rows]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            text = row_to_text(self.rows[index], tokenizer)
            if args.assistant_only_loss:
                encoded_row = assistant_only_tokenized_row(
                    self.rows[index], tokenizer, max_length=args.max_length
                )
                input_ids = encoded_row["input_ids"]
                attention_mask = encoded_row["attention_mask"]
                labels = encoded_row["labels"]
            elif args.response_only_loss:
                encoded_row = response_only_tokenized_row(
                    self.rows[index], tokenizer, max_length=args.max_length
                )
                input_ids = encoded_row["input_ids"]
                attention_mask = encoded_row["attention_mask"]
                labels = encoded_row["labels"]
            else:
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=args.max_length,
                    padding=False if args.dynamic_padding else "max_length",
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].squeeze(0)
                attention_mask = encoded["attention_mask"].squeeze(0)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
            if not torch.any(labels != -100):
                raise RuntimeError(f"SFT row {index} has no supervised tokens")
            if not args.dynamic_padding and int(input_ids.numel()) < args.max_length:
                width = args.max_length - int(input_ids.numel())
                input_ids = torch.nn.functional.pad(
                    input_ids, (0, width), value=int(tokenizer.pad_token_id)
                )
                attention_mask = torch.nn.functional.pad(
                    attention_mask, (0, width), value=0
                )
                labels = torch.nn.functional.pad(labels, (0, width), value=-100)
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            if args.row_loss_weight_field:
                result["loss_weights"] = torch.tensor(
                    self.loss_weights[index], dtype=torch.float32
                )
            return result

    def dynamic_padding_collator(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Pad only to the longest sequence in the current micro-batch.

        The original SFT pilot padded every example to ``max_length``.  That
        is useful as a conservative compatibility default, but it wastes most
        of the compute on heterogeneous agent trajectories.  Keep the default
        unchanged and expose dynamic padding explicitly for full-data runs.
        """

        if not features:
            raise ValueError("cannot collate an empty SFT batch")
        target_length = max(int(item["input_ids"].shape[-1]) for item in features)
        pad_token_id = int(tokenizer.pad_token_id)
        padding_side = str(getattr(tokenizer, "padding_side", "right"))
        batch: dict[str, list[torch.Tensor]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        weighted = ["loss_weights" in item for item in features]
        if any(weighted) and not all(weighted):
            raise ValueError("loss_weights must be present on every row in a batch")
        if all(weighted):
            batch["loss_weights"] = []
        for item in features:
            length = int(item["input_ids"].shape[-1])
            width = target_length - length
            if padding_side == "left":
                input_ids = torch.nn.functional.pad(
                    item["input_ids"], (width, 0), value=pad_token_id
                )
                attention_mask = torch.nn.functional.pad(
                    item["attention_mask"], (width, 0), value=0
                )
                labels = torch.nn.functional.pad(
                    item["labels"], (width, 0), value=-100
                )
            else:
                input_ids = torch.nn.functional.pad(
                    item["input_ids"], (0, width), value=pad_token_id
                )
                attention_mask = torch.nn.functional.pad(
                    item["attention_mask"], (0, width), value=0
                )
                labels = torch.nn.functional.pad(
                    item["labels"], (0, width), value=-100
                )
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
            if all(weighted):
                batch["loss_weights"].append(item["loss_weights"])
        return {key: torch.stack(values) for key, values in batch.items()}

    fsdp_config = None
    if args.fsdp_full_shard:
        fsdp_config = {
            "transformer_layer_cls_to_wrap": [
                args.fsdp_transformer_layer_cls_to_wrap
            ],
            # Full-parameter SFT uses a conventional flat parameter view while
            # FSDP shards parameters, gradients, and optimizer state. These
            # settings are model-family neutral; only the decoder-layer class
            # is supplied by the launcher.
            # PyTorch 2.6 cannot pin the multi-GiB original-parameter shard
            # used by CPU offload on this host.  Flat wrapped decoder shards
            # keep each pinned allocation bounded while preserving the same
            # full-parameter update.
            "use_orig_params": not (
                args.fsdp_cpu_offload or args.fsdp_flat_parameter
            ),
            "sync_module_states": True,
            "limit_all_gathers": True,
            "cpu_ram_efficient_loading": bool(
                args.fsdp_cpu_ram_efficient_loading
            ),
        }
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        bf16=args.dtype == "bf16",
        fp16=args.dtype == "fp16",
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if distributed else None,
        fsdp=(
            "full_shard offload auto_wrap"
            if args.fsdp_full_shard and args.fsdp_cpu_offload
            else "full_shard auto_wrap"
            if args.fsdp_full_shard
            else ""
        ),
        fsdp_config=fsdp_config,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False}
            if distributed and args.gradient_checkpointing
            else None
        ),
        # SparseSupervisedLossMixin normalizes its local token sum itself.
        # Transformers 4.57 changed this default to True, which gathers a
        # world-size denominator before our override and silently scales an
        # 8-GPU loss/gradient by 1/8.  Pin the contract across versions.
        average_tokens_across_devices=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    SparseSupervisedLossMixin = sparse_loss_mixin(
        sparse_supervised_causal_lm_loss,
        enabled=args.sparse_supervised_logits,
        normalization=loss_normalization,
        activation_offload_cpu=args.activation_offload_cpu,
        activation_offload_pin_memory=not args.activation_offload_unpinned,
        ce_checkpoint_chunk_size=getattr(args,'ce_checkpoint_chunk_size',0),
        activation_offload_min_input_tokens=getattr(args,'activation_offload_min_input_tokens',0),
    )

    from trajectory2gym.training.checkpoint_storage import CheckpointStorageMixin
    storage_dtype = getattr(args, "checkpoint_storage_dtype", "native")
    if storage_dtype == "bf16" and not (
        args.dtype == "bf16" and args.save_only_model and args.save_strategy == "no"
        and not args.lora and args.resume_from_checkpoint is None
    ):
        raise ValueError("BF16 model storage is only supported for fresh full-parameter final model-only export")

    class DefaultTrainer(CheckpointStorageMixin, SparseSupervisedLossMixin, Trainer):
        pass

    class OrderedTrainer(CheckpointStorageMixin, SparseSupervisedLossMixin, Trainer):
        def _get_train_sampler(self, train_dataset=None):
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            return SequentialSampler(dataset)

    class DomainWeightedTrainer(CheckpointStorageMixin, SparseSupervisedLossMixin, Trainer):
        def _get_train_sampler(self, train_dataset=None):
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            generator = torch.Generator()
            generator.manual_seed(args.seed)
            if args.learning_value_core_mass is not None:
                masses = (
                    parse_domain_sampling_masses(args.domain_sampling_mass)
                    if args.domain_sampling_mass
                    else None
                )
                alpha = (
                    0.0
                    if args.domain_balanced_sampler
                    else (
                        float(args.domain_sampling_alpha)
                        if args.domain_sampling_alpha is not None
                        else 1.0
                    )
                )
                weights = learning_value_core_tail_sampling_weights(
                    dataset.rows,
                    core_rank=args.learning_value_core_rank,
                    core_mass=args.learning_value_core_mass,
                    domain_alpha=alpha,
                    domain_masses=masses,
                    task_balanced=args.learning_value_task_balanced,
                )
            elif args.domain_sampling_mass:
                weights = domain_target_sampling_weights(
                    dataset.rows,
                    masses=parse_domain_sampling_masses(args.domain_sampling_mass),
                )
            else:
                alpha = (
                    0.0
                    if args.domain_balanced_sampler
                    else float(args.domain_sampling_alpha)
                )
                weights = domain_temperature_sampling_weights(
                    dataset.rows, alpha=alpha
                )
            return WeightedRandomSampler(
                weights,
                num_samples=len(dataset),
                replacement=True,
                generator=generator,
            )

    class DomainCoverageTrainer(CheckpointStorageMixin, SparseSupervisedLossMixin, Trainer):
        def _get_train_sampler(self, train_dataset=None):
            dataset = train_dataset if train_dataset is not None else self.train_dataset

            class CoverageSampler(Sampler):
                def __init__(self):
                    self.epoch = 0
                    self.length = len(
                        domain_coverage_capped_sampling_indices(
                            dataset.rows,
                            repeat_cap=args.domain_coverage_repeat_cap,
                            seed=args.seed,
                            task_balanced=args.domain_coverage_task_balanced,
                        )
                    )

                def __iter__(self):
                    indices = domain_coverage_capped_sampling_indices(
                        dataset.rows,
                        repeat_cap=args.domain_coverage_repeat_cap,
                        seed=args.seed + self.epoch,
                        task_balanced=args.domain_coverage_task_balanced,
                    )
                    self.epoch += 1
                    return iter(indices)

                def __len__(self):
                    return self.length

            return CoverageSampler()

    if args.sequential_train_sampler:
        trainer_class = OrderedTrainer
    elif args.domain_coverage_repeat_cap is not None:
        trainer_class = DomainCoverageTrainer
    elif (
        args.domain_balanced_sampler
        or args.domain_sampling_alpha is not None
        or args.domain_sampling_mass
        or args.learning_value_core_mass is not None
    ):
        trainer_class = DomainWeightedTrainer
    else:
        trainer_class = DefaultTrainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=SFTDataset(train_rows),
        eval_dataset=SFTDataset(val_rows) if val_rows else None,
        data_collator=dynamic_padding_collator if args.dynamic_padding else None,
    )
    trainer.checkpoint_storage_dtype = storage_dtype
    if sync_each_microbatch:
        if args.fsdp_cpu_offload and args.gradient_accumulation_steps > 1:
            raise ValueError("CPU-shard offload with accumulated gradients is not validated")
        # HF 4.57 explicitly enters accelerator.no_sync for nonfinal
        # microbatches, retaining full unsharded FSDP gradients. Synchronize
        # each backward instead; GPU-resident shard gradients still accumulate
        # until the same optimizer boundary. This changes communication only.
        trainer.accelerator.no_sync = lambda model: nullcontext()
    resume_checkpoint = (
        str(args.resume_from_checkpoint.resolve())
        if args.resume_from_checkpoint is not None
        else None
    )
    if args.resume_from_checkpoint is not None and not args.resume_from_checkpoint.is_dir():
        raise ValueError("--resume-from-checkpoint must name an existing directory")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    metrics = dict(train_result.metrics)
    if val_rows:
        metrics.update({f"eval_{key}": value for key, value in trainer.evaluate().items()})
    if not args.skip_final_save:
        if args.release_training_state_before_save:
            trainer.accelerator.wait_for_everyone()
            rank = int(os.environ.get("RANK", "0"))
            before_free, _ = torch.cuda.mem_get_info()
            optimizer = trainer.optimizer
            if optimizer is None:
                raise RuntimeError("pre-save optimizer release requires a completed optimizer")
            optimizer.zero_grad(set_to_none=True)
            underlying = getattr(optimizer, "optimizer", optimizer)
            state = getattr(underlying, "state", None)
            if not isinstance(state, dict):
                raise RuntimeError("pre-save optimizer state is not a mutable mapping")
            optimizer_state_entries = len(state)
            state.clear()
            trainer.optimizer = None
            prepared_optimizers = getattr(trainer.accelerator, "_optimizers", None)
            if isinstance(prepared_optimizers, list):
                prepared_optimizers.clear()
            del optimizer, underlying, state
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            after_free, _ = torch.cuda.mem_get_info()
            memory_receipt = {
                "format": "trajectory2gym_pre_save_hbm_release_v1",
                "status": "released_pending_global_gate",
                "rank": rank,
                "before_free_bytes": before_free,
                "after_free_bytes": after_free,
                "optimizer_state_entries_cleared": optimizer_state_entries,
                "optimizer_updates_changed": False,
                "model_parameters_changed": False,
            }
            args.output_dir.mkdir(parents=True, exist_ok=True)
            receipt_path = args.output_dir / f"pre_save_hbm_rank{rank}.json"
            with receipt_path.open("x", encoding="utf-8") as stream:
                json.dump(memory_receipt, stream, indent=2, sort_keys=True)
                stream.write("\n")
            minimum_free = after_free
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                free_tensor = torch.tensor([after_free], device=torch.cuda.current_device(), dtype=torch.int64)
                torch.distributed.all_reduce(free_tensor, op=torch.distributed.ReduceOp.MIN)
                minimum_free = int(free_tensor.item())
            print(
                "PRE_SAVE_HBM_RELEASE "
                + json.dumps({"rank": rank, "before_free_mib": before_free // 2**20,
                              "after_free_mib": after_free // 2**20,
                              "minimum_rank_free_mib": minimum_free // 2**20}, sort_keys=True),
                flush=True,
            )
            if minimum_free < int(args.pre_save_min_free_gib * 2**30):
                raise RuntimeError(
                    "pre-save FSDP allgather headroom below frozen minimum: "
                    f"{minimum_free} bytes < {args.pre_save_min_free_gib} GiB"
                )
            trainer.accelerator.wait_for_everyone()
        if args.fsdp_sharded_final_save:
            from torch.distributed.fsdp import (
                ShardedOptimStateDictConfig,
                ShardedStateDictConfig,
            )

            plugin = trainer.accelerator.state.fsdp_plugin
            plugin.state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
            plugin.optim_state_dict_config = ShardedOptimStateDictConfig(
                offload_to_cpu=True
            )
            plugin.set_state_dict_type("SHARDED_STATE_DICT")
            trainer.accelerator.save_state(str(args.output_dir / "dcp"))
            trainer.accelerator.wait_for_everyone()
            if trainer.is_world_process_zero():
                unwrapped = trainer.accelerator.unwrap_model(trainer.model)
                unwrapped.config.save_pretrained(str(args.output_dir))
        else:
            trainer.save_model(str(args.output_dir))
        trainer.accelerator.wait_for_everyone()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(str(args.output_dir))
    else:
        trainer.accelerator.wait_for_everyone()
    merged_lora_output_dir = None
    if args.merge_lora_output_dir is not None:
        if not args.lora:
            raise ValueError("--merge-lora-output-dir requires --lora")
        if not hasattr(model, "merge_and_unload"):
            raise ValueError("loaded LoRA model does not support merge_and_unload()")
        if trainer.is_world_process_zero():
            merged_model = model.merge_and_unload()
            args.merge_lora_output_dir.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(str(args.merge_lora_output_dir))
            tokenizer.save_pretrained(str(args.merge_lora_output_dir))
        trainer.accelerator.wait_for_everyone()
        merged_lora_output_dir = str(args.merge_lora_output_dir)
    world_size = int(getattr(trainer.accelerator, "num_processes", 1) or 1)
    effective_global_batch_size = (
        int(args.per_device_train_batch_size)
        * int(args.gradient_accumulation_steps)
        * world_size
    )
    sampling_exposure = summarize_sampling_exposure(
        train_rows,
        epochs=float(args.num_train_epochs),
        max_steps=int(args.max_steps),
        world_size=world_size,
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        seed=int(args.seed),
        domain_coverage_repeat_cap=args.domain_coverage_repeat_cap,
        domain_coverage_task_balanced=args.domain_coverage_task_balanced,
    )
    sampling_weights: list[float] | None = None
    sampling_name: str | None = None
    if args.learning_value_core_mass is not None:
        sampling_weights = learning_value_core_tail_sampling_weights(
            train_rows,
            core_rank=args.learning_value_core_rank,
            core_mass=args.learning_value_core_mass,
            domain_alpha=(
                0.0
                if args.domain_balanced_sampler
                else float(args.domain_sampling_alpha or 1.0)
            ),
            domain_masses=(
                parse_domain_sampling_masses(args.domain_sampling_mass)
                if args.domain_sampling_mass
                else None
            ),
            task_balanced=args.learning_value_task_balanced,
        )
        sampling_name = "learning_value_weighted_replacement"
    elif args.domain_sampling_mass:
        sampling_weights = domain_target_sampling_weights(
            train_rows,
            masses=parse_domain_sampling_masses(args.domain_sampling_mass),
        )
        sampling_name = "domain_mass_weighted_replacement"
    elif args.domain_balanced_sampler:
        sampling_weights = domain_balanced_sampling_weights(train_rows)
        sampling_name = "domain_balanced_weighted_replacement"
    elif args.domain_sampling_alpha is not None:
        sampling_weights = domain_temperature_sampling_weights(
            train_rows, alpha=float(args.domain_sampling_alpha)
        )
        sampling_name = "domain_temperature_weighted_replacement"
    if sampling_weights is not None:
        total_weight = sum(sampling_weights)
        expected_mass: Counter[str] = Counter()
        for row, weight in zip(train_rows, sampling_weights):
            expected_mass[str(row.get("domain") or "")] += weight / total_weight
        presentations = int(sampling_exposure["planned_sample_presentations"])
        sampling_exposure.update(
            {
                "sampler": sampling_name,
                "weighted_sampling_with_replacement": True,
                "planned_domain_mass": dict(sorted(expected_mass.items())),
                "planned_presentations_by_domain": {
                    domain: mass * presentations
                    for domain, mass in sorted(expected_mass.items())
                },
                # Expected exposure is auditable by domain, but replacement
                # sampling cannot promise that every individual row appears.
                "all_source_rows_seen": False,
                "source_row_visit_min": None,
                "source_row_visit_max": None,
                "planned_source_tasks_seen": None,
                "source_task_visit_min": None,
                "source_task_visit_max": None,
            }
        )
    return {
        "format": "trajectory2gym_sft_train_result_v1",
        "model": args.model,
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "output_dir": str(args.output_dir),
        "merged_lora_output_dir": merged_lora_output_dir,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "training_contract": {
            "train_mode": "lora" if args.lora else "full",
            "model_loader": args.model_loader,
            "freeze_vision_tower": bool(args.freeze_vision_tower),
            "frozen_vision_parameters": int(frozen_vision_parameters),
            "frozen_vision_parameter_tensors": len(frozen_vision_parameter_names),
            "fsdp_full_shard": bool(args.fsdp_full_shard),
            "fsdp_cpu_offload": bool(args.fsdp_cpu_offload),
            "fsdp_cpu_offload_unpinned": bool(args.fsdp_cpu_offload_unpinned),
            "fsdp_flat_parameter": bool(args.fsdp_flat_parameter),
            "fsdp_cpu_ram_efficient_loading": bool(
                args.fsdp_cpu_ram_efficient_loading
            ),
            "activation_offload_cpu": bool(args.activation_offload_cpu),
            "activation_offload_min_input_tokens": getattr(args,'activation_offload_min_input_tokens',0),
            "rank_local_activation_placement_counts": getattr(trainer,'_activation_placement_counts',None),
            "activation_offload_pin_memory": (
                not bool(args.activation_offload_unpinned)
                if args.activation_offload_cpu
                else None
            ),
            "fsdp_transformer_layer_cls_to_wrap": (
                args.fsdp_transformer_layer_cls_to_wrap
                if args.fsdp_full_shard
                else None
            ),
            "world_size": world_size,
            "per_device_train_batch_size": int(args.per_device_train_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "effective_global_batch_size": effective_global_batch_size,
            "num_train_epochs": float(args.num_train_epochs),
            "max_steps": int(args.max_steps),
            "learning_rate": float(args.learning_rate),
            "warmup_steps": int(args.warmup_steps),
            "warmup_ratio": float(args.warmup_ratio),
            "weight_decay": float(args.weight_decay),
            "optimizer": str(args.optim),
            "lr_scheduler_type": str(args.lr_scheduler_type),
            "max_grad_norm": float(args.max_grad_norm),
            "max_length": int(args.max_length),
            "dtype": str(args.dtype),
            "attention_implementation": args.attn_implementation,
            "seed": int(args.seed),
            "save_steps": int(args.save_steps),
            "save_strategy": str(args.save_strategy),
            "save_total_limit": int(args.save_total_limit),
            "save_only_model": bool(args.save_only_model),
            "checkpoint_storage_dtype": storage_dtype,
            "release_training_state_before_save": bool(args.release_training_state_before_save),
            "pre_save_min_free_gib": float(args.pre_save_min_free_gib),
            "fsdp_sharded_final_save": bool(args.fsdp_sharded_final_save),
            "skip_final_save": bool(args.skip_final_save),
            "resume_from_checkpoint": resume_checkpoint,
            "sparse_supervised_logits": bool(args.sparse_supervised_logits),
            "row_loss_weight_field": args.row_loss_weight_field,
            "loss_normalization": (
                loss_normalization if loss_normalization != "legacy_local_token_mean"
                else "per_device_row_target_token_mean"
            ),
            "trainer_num_items_in_batch_ignored": loss_normalization == "legacy_local_token_mean",
            "causal_shifted_global_denominator": loss_normalization == "global_batch_token_mean",
            "sync_each_microbatch": sync_each_microbatch,
            "cuda_memory_fraction": cuda_memory_fraction,
            "ce_checkpoint_chunk_size": getattr(args,'ce_checkpoint_chunk_size',0),
            "accelerator_gradient_accumulation_steps": trainer.accelerator.gradient_accumulation_steps,
            "row_loss_weight_min": min(train_loss_weights, default=1.0),
            "row_loss_weight_max": max(train_loss_weights, default=1.0),
            "row_loss_weight_mean": (
                sum(train_loss_weights) / len(train_loss_weights)
                if train_loss_weights
                else 1.0
            ),
        },
        "lora": args.lora,
        "lora_contract": (
            {
                "rank": int(args.lora_rank),
                "alpha": int(args.lora_alpha),
                "dropout": float(args.lora_dropout),
                "target_modules": [
                    item.strip()
                    for item in str(args.lora_target_modules).split(",")
                    if item.strip()
                ],
            }
            if args.lora
            else None
        ),
        "lora_init_adapter": (
            str(args.lora_init_adapter) if args.lora_init_adapter is not None else None
        ),
        "response_only_loss": args.response_only_loss,
        "assistant_only_loss": args.assistant_only_loss,
        "dynamic_padding": args.dynamic_padding,
        "sequential_train_sampler": args.sequential_train_sampler,
        "domain_balanced_sampler": args.domain_balanced_sampler,
        "domain_sampling_alpha": args.domain_sampling_alpha,
        "domain_sampling_mass": (
            parse_domain_sampling_masses(args.domain_sampling_mass)
            if args.domain_sampling_mass
            else None
        ),
        "domain_coverage_repeat_cap": args.domain_coverage_repeat_cap,
        "domain_coverage_task_balanced": args.domain_coverage_task_balanced,
        "sampling_exposure": sampling_exposure,
        "learning_value_core_rank": args.learning_value_core_rank,
        "learning_value_core_mass": args.learning_value_core_mass,
        "learning_value_task_balanced": args.learning_value_task_balanced,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or preflight a causal-LM SFT run on trajectory2gym SFT views.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--model-loader",
        choices=("causal_lm", "image_text_to_text"),
        default="causal_lm",
        help=(
            "Explicit Transformers AutoModel class. Native multimodal checkpoints "
            "such as Qwen3.8 require image_text_to_text so the checkpoint prefix "
            "and visual tower are loaded rather than silently reinitialized."
        ),
    )
    parser.add_argument(
        "--freeze-vision-tower",
        action="store_true",
        help=(
            "Freeze the loaded visual tower for text-only SFT while retaining it "
            "in the saved checkpoint."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional Transformers attention backend, e.g. flash_attention_2.",
    )
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--checkpoint-storage-dtype", choices=("native", "bf16"), default="native")
    parser.add_argument("--release-training-state-before-save", action="store_true")
    parser.add_argument("--pre-save-min-free-gib", type=float, default=0.0)
    parser.add_argument("--fsdp-sharded-final-save", action="store_true")
    parser.add_argument("--activation-offload-min-input-tokens", type=int, default=0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Fraction of optimizer steps used for linear warmup.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--optim",
        choices=("adamw_torch", "adamw_torch_fused"),
        default="adamw_torch",
        help="Transformers optimizer backend; fused AdamW preserves the update rule with lower CUDA peak memory.",
    )
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Trainer and data-shuffle seed (default: Transformers' historical 42).",
    )
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument(
        "--save-strategy",
        choices=("no", "steps", "epoch"),
        default="steps",
        help="Transformers checkpoint strategy; final model saving is separate.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=1,
        help="Number of trainer checkpoints to retain (final merged output is separate).",
    )
    parser.add_argument(
        "--save-only-model",
        action="store_true",
        help="Do not persist optimizer/scheduler/RNG state in Trainer checkpoints.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume Trainer optimizer/scheduler/RNG state from an exact checkpoint.",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--sequential-train-sampler",
        action="store_true",
        help=(
            "Consume training rows in file order. Use with optimizer-balanced "
            "contiguous blocks whose size equals the global accumulation batch."
        ),
    )
    parser.add_argument(
        "--domain-balanced-sampler",
        action="store_true",
        help=(
            "Sample rows with inverse domain-frequency weights so every "
            "domain has equal expected training mass."
        ),
    )
    parser.add_argument(
        "--domain-sampling-alpha",
        type=float,
        default=None,
        help=(
            "Optional generic domain-temperature sampler: 0 is equal-domain, "
            "1 is natural example mass, and 0.5 is square-root mass."
        ),
    )
    parser.add_argument(
        "--domain-sampling-mass",
        action="append",
        default=[],
        metavar="DOMAIN=MASS",
        help=(
            "Explicit positive target mass for one domain. Repeat once for "
            "every observed domain; mutually exclusive with alpha/balanced."
        ),
    )
    parser.add_argument(
        "--domain-coverage-repeat-cap",
        type=int,
        default=None,
        help=(
            "Visit every selected row once per epoch, then repeat smaller "
            "domains up to this many total passes without exceeding the "
            "largest domain's row count. Mutually exclusive with other "
            "sampler modes."
        ),
    )
    parser.add_argument(
        "--domain-coverage-task-balanced",
        action="store_true",
        help=(
            "Allocate repeat visits to the least-exposed source task and "
            "row inside each small domain. Requires coverage-capped sampling."
        ),
    )
    parser.add_argument(
        "--learning-value-core-rank",
        type=int,
        default=4,
        help=(
            "Rows at or below this per-task learning_value rank form the "
            "high-value core for core-tail sampling."
        ),
    )
    parser.add_argument(
        "--learning-value-core-mass",
        type=float,
        default=None,
        help=(
            "Optional expected within-domain mass assigned to the high-value "
            "core. The remaining non-zero mass samples the complete tail."
        ),
    )
    parser.add_argument(
        "--learning-value-task-balanced",
        action="store_true",
        help=(
            "Within each domain, give every source task equal expected mass "
            "before applying the learning-value core/tail split."
        ),
    )
    parser.add_argument(
        "--dynamic-padding",
        action="store_true",
        help="Pad each micro-batch to its longest sequence instead of max-length.",
    )
    parser.add_argument("--response-only-loss", action="store_true", help="Mask prompt tokens so loss is computed only on the target response.")
    parser.add_argument(
        "--assistant-only-loss",
        action="store_true",
        help=(
            "For full conversational traces, mask system/user/tool tokens and "
            "supervise every assistant turn using the native chat template."
        ),
    )
    parser.add_argument(
        "--sparse-supervised-logits",
        action="store_true",
        help=(
            "Project hidden states onto the vocabulary only where the shifted "
            "labels are supervised. This preserves full context while avoiding "
            "dense-logit OOMs on long masked trajectories."
        ),
    )
    parser.add_argument(
        "--row-loss-weight-field",
        default="",
        help=(
            "Optional dotted row field containing a positive scalar that "
            "multiplies that row's supervised-token loss. Requires sparse "
            "supervised logits; the unweighted token denominator is retained "
            "so the declared weight changes gradient mass."
        ),
    )
    parser.add_argument(
        "--loss-normalization",
        choices=("legacy_local_token_mean", "global_batch_token_mean", "global_batch_parent_mean"),
        default="legacy_local_token_mean",
        help="Versioned CE denominator. Global mode sums shifted targets across all ranks and accumulation microbatches.",
    )
    parser.add_argument(
        "--sync-each-microbatch", action="store_true",
        help="Reduce-scatter FSDP gradients after each backward to avoid retaining full gradients between microbatches.",
    )
    parser.add_argument('--cuda-memory-fraction',type=float,default=None)
    parser.add_argument('--ce-checkpoint-chunk-size',type=int,default=0)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument(
        "--fsdp-full-shard",
        action="store_true",
        help=(
            "Run full-parameter SFT with FSDP full sharding and transformer "
            "auto-wrap. Requires a distributed launch and cannot be combined "
            "with --lora."
        ),
    )
    parser.add_argument(
        "--fsdp-cpu-offload",
        action="store_true",
        help=(
            "Offload FSDP parameter and gradient shards to CPU between uses. "
            "Requires --fsdp-full-shard and preserves full-parameter updates."
        ),
    )
    parser.add_argument(
        "--fsdp-cpu-offload-unpinned",
        action="store_true",
        help=(
            "Use pageable FSDP CPU parameter/gradient shards when this host "
            "cannot register large pinned buffers; values and optimizer are unchanged."
        ),
    )
    parser.add_argument(
        "--fsdp-flat-parameter",
        action="store_true",
        help=(
            "Use the conventional flattened FSDP parameter representation "
            "without CPU offload. This is execution placement metadata and "
            "requires --fsdp-full-shard."
        ),
    )
    parser.add_argument(
        "--fsdp-cpu-ram-efficient-loading",
        action="store_true",
        help=(
            "Initialize distributed FSDP before model loading so only rank 0 "
            "reads checkpoint weights and other ranks start from meta tensors."
        ),
    )
    parser.add_argument(
        "--skip-final-save",
        action="store_true",
        help="Skip checkpoint serialization for a bounded training smoke test.",
    )
    parser.add_argument(
        "--activation-offload-cpu",
        action="store_true",
        help=(
            "Store tensors saved for backward on CPU during the sparse-loss "
            "model forward. This changes execution placement, not the loss."
        ),
    )
    parser.add_argument(
        "--activation-offload-unpinned",
        action="store_true",
        help=(
            "Use pageable CPU storage for saved-tensor activation offload. "
            "This is a memory-transport fallback and does not change the loss."
        ),
    )
    parser.add_argument(
        "--fsdp-transformer-layer-cls-to-wrap",
        default="Qwen3DecoderLayer",
        help=(
            "Decoder-layer class used by FSDP auto-wrap. This is an execution "
            "adapter, not a benchmark-specific training rule."
        ),
    )
    parser.add_argument(
        "--lora-init-adapter",
        type=Path,
        default=None,
        help="Optional existing LoRA adapter to continue training as a curriculum stage.",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument(
        "--merge-lora-output-dir",
        type=Path,
        default=None,
        help="Optional full-model output directory after merging a LoRA adapter. Use this path as TRAJECTORY2GYM_MODEL_PATH for RL.",
    )
    args = parser.parse_args()
    if args.response_only_loss and args.assistant_only_loss:
        parser.error("choose only one of --response-only-loss or --assistant-only-loss")
    if args.row_loss_weight_field and not args.sparse_supervised_logits:
        parser.error("--row-loss-weight-field requires --sparse-supervised-logits")
    if args.loss_normalization == "global_batch_token_mean":
        if not args.sparse_supervised_logits or args.row_loss_weight_field:
            parser.error("global_batch_token_mean requires sparse logits without row weights")
    if args.loss_normalization == "global_batch_parent_mean":
        if (not args.sparse_supervised_logits or not args.row_loss_weight_field
                or args.per_device_train_batch_size != 1):
            parser.error("global_batch_parent_mean requires sparse logits, audited weights, and microbatch 1")
    if args.ce_checkpoint_chunk_size < 0:
        parser.error("--ce-checkpoint-chunk-size must be nonnegative")
    if args.domain_sampling_alpha is not None and not 0.0 <= args.domain_sampling_alpha <= 1.0:
        parser.error("--domain-sampling-alpha must be in [0, 1]")
    if (
        args.domain_coverage_repeat_cap is not None
        and args.domain_coverage_repeat_cap < 1
    ):
        parser.error("--domain-coverage-repeat-cap must be at least 1")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup-ratio must be in [0, 1)")
    if args.warmup_steps and args.warmup_ratio:
        parser.error("choose only one of --warmup-steps or --warmup-ratio")
    if (
        args.domain_coverage_task_balanced
        and args.domain_coverage_repeat_cap is None
    ):
        parser.error(
            "--domain-coverage-task-balanced requires "
            "--domain-coverage-repeat-cap"
        )
    if args.learning_value_core_rank <= 0:
        parser.error("--learning-value-core-rank must be positive")
    if (
        args.learning_value_core_mass is not None
        and not 0.0 <= args.learning_value_core_mass < 1.0
    ):
        parser.error("--learning-value-core-mass must be in [0, 1)")
    if (
        args.learning_value_task_balanced
        and args.learning_value_core_mass is None
    ):
        parser.error(
            "--learning-value-task-balanced requires "
            "--learning-value-core-mass"
        )
    try:
        if args.domain_sampling_mass:
            parse_domain_sampling_masses(args.domain_sampling_mass)
    except ValueError as exc:
        parser.error(str(exc))
    sampler_modes = sum(
        (
            bool(args.domain_balanced_sampler),
            args.domain_sampling_alpha is not None,
            bool(args.domain_sampling_mass),
            args.domain_coverage_repeat_cap is not None,
        )
    )
    if sampler_modes > 1:
        parser.error(
            "choose one of --domain-balanced-sampler, "
            "--domain-sampling-alpha, --domain-sampling-mass, or "
            "--domain-coverage-repeat-cap"
        )
    if args.sequential_train_sampler and (
        args.domain_balanced_sampler
        or args.domain_sampling_alpha is not None
        or args.domain_sampling_mass
        or args.domain_coverage_repeat_cap is not None
        or args.learning_value_core_mass is not None
    ):
        parser.error("choose only one train sampler")
    if args.lora_init_adapter is not None and not args.lora:
        parser.error("--lora-init-adapter requires --lora")
    if args.fsdp_full_shard and args.lora:
        parser.error("--fsdp-full-shard cannot be combined with --lora")
    if args.fsdp_cpu_offload and not args.fsdp_full_shard:
        parser.error("--fsdp-cpu-offload requires --fsdp-full-shard")
    if args.fsdp_cpu_offload_unpinned and not args.fsdp_cpu_offload:
        parser.error("--fsdp-cpu-offload-unpinned requires --fsdp-cpu-offload")
    if args.release_training_state_before_save and not (
        args.fsdp_full_shard and args.save_only_model and args.save_strategy == "no"
        and args.resume_from_checkpoint is None and not args.skip_final_save
    ):
        parser.error("pre-save state release requires fresh FSDP model-only final export")
    if args.pre_save_min_free_gib < 0 or (
        args.pre_save_min_free_gib and not args.release_training_state_before_save
    ):
        parser.error("pre-save HBM minimum requires the state-release mode")
    if args.fsdp_sharded_final_save and not args.release_training_state_before_save:
        parser.error("FSDP sharded final save requires pre-save state release")
    if args.fsdp_flat_parameter and not args.fsdp_full_shard:
        parser.error("--fsdp-flat-parameter requires --fsdp-full-shard")
    if args.fsdp_cpu_ram_efficient_loading and not args.fsdp_full_shard:
        parser.error(
            "--fsdp-cpu-ram-efficient-loading requires --fsdp-full-shard"
        )

    if args.dry_run:
        train_rows = load_jsonl(args.train_jsonl, max_rows=args.max_train_rows)
        val_rows = load_jsonl(args.val_jsonl, max_rows=args.max_val_rows)
        result = {
            "format": "trajectory2gym_sft_train_preflight_v1",
            "train_jsonl": str(args.train_jsonl),
            "val_jsonl": str(args.val_jsonl),
            "output_dir": str(args.output_dir),
            "model": args.model,
            "response_only_loss": args.response_only_loss,
            "assistant_only_loss": args.assistant_only_loss,
            "train_summary": summarize_rows(train_rows),
            "val_summary": summarize_rows(val_rows),
        }
    else:
        if not args.model:
            parser.error("--model is required unless --dry-run is set")
        result = run_training(args)

    is_world_process_zero = int(os.environ.get("RANK", "0")) == 0
    if args.summary_json is not None and is_world_process_zero:
        write_json(args.summary_json, result)
    if is_world_process_zero:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _destroy_distributed_process_group() -> None:
    """Release NCCL explicitly after final FSDP serialization.

    Relying on interpreter teardown left nonzero ranks spinning in the NCCL
    watchdog after rank zero had already written a complete checkpoint. This
    is execution cleanup only and does not touch model/data/optimizer state.
    """

    try:
        import gc
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            # Trainer/FSDP retain Python cycles and reserved CUDA blocks after
            # run_training returns. NCCL teardown itself can allocate buffers;
            # free unreachable model state before destroying communicators.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            dist.destroy_process_group()
    except Exception as exc:  # pragma: no cover - environment-specific failure
        print(f"distributed teardown warning: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    try:
        main()
    finally:
        _destroy_distributed_process_group()

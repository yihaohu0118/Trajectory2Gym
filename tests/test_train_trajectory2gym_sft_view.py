from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.train.train_trajectory2gym_sft_view import (
    _destroy_distributed_process_group,
    assistant_only_tokenized_row,
    domain_balanced_sampling_weights,
    domain_coverage_capped_sampling_indices,
    domain_target_sampling_weights,
    domain_temperature_sampling_weights,
    learning_value_core_tail_sampling_weights,
    parse_domain_sampling_masses,
    render_messages,
    row_prompt_response_messages,
    row_to_prompt_text,
    row_to_response_text,
    row_to_text,
    response_only_tokenized_row,
    sparse_supervised_causal_lm_loss,
    summarize_sampling_exposure,
    summarize_rows,
)


class TrainTrajectory2GymSFTViewTest(unittest.TestCase):
    class RecordingTokenizer:
        chat_template = ""

        def __init__(self) -> None:
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(kwargs)
            body = "|".join(
                f"<|im_start|>{item['role']}\n{item.get('content', '')}<|im_end|>"
                for item in messages
            )
            if kwargs.get("add_generation_prompt"):
                body += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            return body

        def encode(self, text, add_special_tokens=False):
            return [ord(char) for char in text]

        def __call__(self, text, **kwargs):
            import torch

            return {"input_ids": torch.tensor([[ord(char) for char in text]])}

    def test_distributed_process_group_is_explicitly_destroyed(self) -> None:
        from unittest.mock import patch

        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.destroy_process_group") as destroy,
        ):
            _destroy_distributed_process_group()
        destroy.assert_called_once_with()

    class AssistantOnlyTokenizer(RecordingTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(kwargs)
            text = "".join(
                f"<|im_start|>{message['role']}\n{message.get('content', '')}<|im_end|>"
                for message in messages
            )
            return {"input_ids": [ord(character) for character in text]}

    class BoundaryMergingTokenizer(AssistantOnlyTokenizer):
        """Mimic a BPE merge across the assistant-header newline."""

        def decode(self, ids, skip_special_tokens=False):
            return "".join("\n\n" if token == 200_001 else chr(token) for token in ids)

        def apply_chat_template(self, messages, **kwargs):
            rendered = super().apply_chat_template(messages, **kwargs)
            text = "".join(chr(token) for token in rendered["input_ids"])
            needle = "<|im_start|>assistant\n\n"
            marker = "<|im_start|>assistant"
            self.assertion_target = needle in text
            tokens: list[int] = []
            cursor = 0
            while cursor < len(text):
                if text.startswith(needle, cursor):
                    tokens.extend(ord(character) for character in marker)
                    tokens.append(200_001)
                    cursor += len(needle)
                else:
                    tokens.append(ord(text[cursor]))
                    cursor += 1
            return {"input_ids": tokens}

    def test_sparse_supervised_loss_matches_dense_masked_causal_loss(self) -> None:
        import torch
        import torch.nn.functional as functional

        class TinyCausalLM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding(11, 5)
                self.lm_head = torch.nn.Linear(5, 11, bias=False)

            def forward(self, input_ids, logits_to_keep=0, **_kwargs):
                hidden = self.embedding(input_ids)
                positions = (
                    slice(-logits_to_keep, None)
                    if isinstance(logits_to_keep, int)
                    else logits_to_keep
                )
                return SimpleNamespace(logits=self.lm_head(hidden[:, positions, :]))

        torch.manual_seed(7)
        model = TinyCausalLM()
        input_ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        labels = torch.tensor(
            [[-100, -100, 3, -100, 5], [-100, 4, 2, -100, -100]]
        )
        dense_logits = model(input_ids=input_ids).logits[:, :-1, :]
        dense_loss = functional.cross_entropy(
            dense_logits.reshape(-1, 11),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        sparse_loss, sparse_outputs = sparse_supervised_causal_lm_loss(
            model,
            {"input_ids": input_ids, "labels": labels},
        )

        self.assertTrue(torch.allclose(sparse_loss, dense_loss, atol=1e-6))
        # The union contains positions 0, 1, and 3 rather than all four
        # causal prediction positions.
        self.assertEqual(tuple(sparse_outputs.logits.shape), (2, 3, 11))

    def test_sparse_supervised_loss_applies_row_gradient_weights(self) -> None:
        import torch
        import torch.nn.functional as functional

        class TinyCausalLM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding(11, 5)
                self.lm_head = torch.nn.Linear(5, 11, bias=False)

            def forward(self, input_ids, logits_to_keep=0, **_kwargs):
                hidden = self.embedding(input_ids)
                return SimpleNamespace(logits=self.lm_head(hidden[:, logits_to_keep, :]))

        torch.manual_seed(11)
        model = TinyCausalLM()
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        labels = torch.tensor([[-100, 2, -100, 4], [-100, 3, -100, 1]])
        positions = torch.tensor([0, 2])
        outputs = model(input_ids=input_ids, logits_to_keep=positions)
        targets = labels[:, 1:].index_select(-1, positions)
        per_token = functional.cross_entropy(
            outputs.logits.float().reshape(-1, 11),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(targets)
        expected = (
            per_token[0].sum() * 3.0 + per_token[1].sum()
        ) / targets.ne(-100).sum()

        loss, _ = sparse_supervised_causal_lm_loss(
            model,
            {
                "input_ids": input_ids,
                "labels": labels,
                "loss_weights": torch.tensor([3.0, 1.0]),
            },
        )
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

    def test_domain_balanced_weights_equalize_expected_mass(self) -> None:
        rows = [
            {"domain": "retail"},
            {"domain": "retail"},
            {"domain": "retail"},
            {"domain": "airline"},
        ]
        weights = domain_balanced_sampling_weights(rows)
        self.assertAlmostEqual(sum(weights[:3]), weights[3])
        self.assertEqual(weights[:3], [1 / 3, 1 / 3, 1 / 3])

    def test_preformatted_qwenfc_row_masks_exact_official_prompt(self) -> None:
        tokenizer = self.RecordingTokenizer()
        row = {
            "transport_protocol": "bfcl_official_qwenfc_raw_completion_v1",
            "preformatted_prompt": "official prompt\n<think>\n\n</think>\n\n",
            "preformatted_completion": '<tool_call>\n{"name":"f","arguments":{}}\n</tool_call>',
            "preformatted_end": "<|im_end|>\n",
        }
        encoded = assistant_only_tokenized_row(row, tokenizer, max_length=4096)
        prompt_tokens = len(row["preformatted_prompt"])
        self.assertTrue((encoded["labels"][:prompt_tokens] == -100).all())
        self.assertTrue((encoded["labels"][prompt_tokens:] != -100).all())
        self.assertEqual(
            row_to_text(row, tokenizer),
            row["preformatted_prompt"]
            + row["preformatted_completion"]
            + row["preformatted_end"],
        )

    def test_domain_balanced_weights_require_declared_domain(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires domain"):
            domain_balanced_sampling_weights([{"domain": "retail"}, {}])

    def test_domain_coverage_sampler_never_drops_rows_and_caps_repeats(self) -> None:
        rows = (
            [{"domain": "retail"}] * 10
            + [{"domain": "airline"}] * 2
            + [{"domain": "telecom"}]
        )
        indices = domain_coverage_capped_sampling_indices(
            rows, repeat_cap=3, seed=7
        )

        self.assertEqual(set(indices), set(range(len(rows))))
        self.assertEqual(len(indices), 10 + 6 + 3)
        counts = {index: indices.count(index) for index in range(len(rows))}
        self.assertTrue(all(counts[index] == 1 for index in range(10)))
        self.assertTrue(all(counts[index] == 3 for index in range(10, 13)))
        self.assertEqual(
            indices,
            domain_coverage_capped_sampling_indices(
                rows, repeat_cap=3, seed=7
            ),
        )

    def test_domain_coverage_sampler_validates_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            domain_coverage_capped_sampling_indices(
                [{"domain": "retail"}], repeat_cap=0, seed=0
            )
        with self.assertRaisesRegex(ValueError, "requires domain"):
            domain_coverage_capped_sampling_indices(
                [{"domain": "retail"}, {}], repeat_cap=2, seed=0
            )

    def test_task_balanced_coverage_prefers_rare_tasks_and_caps_rows(self) -> None:
        rows = [
            {"domain": "retail", "source_task_id": f"retail:{index}"}
            for index in range(8)
        ] + [
            {"domain": "airline", "source_task_id": "airline:rare"},
            {"domain": "airline", "source_task_id": "airline:common"},
            {"domain": "airline", "source_task_id": "airline:common"},
        ]
        indices = domain_coverage_capped_sampling_indices(
            rows,
            repeat_cap=3,
            seed=11,
            task_balanced=True,
        )

        self.assertEqual(set(indices), set(range(len(rows))))
        self.assertEqual(len(indices), 8 + 8)
        row_visits = {index: indices.count(index) for index in range(len(rows))}
        self.assertLessEqual(max(row_visits.values()), 3)
        rare_visits = row_visits[8]
        common_visits = row_visits[9] + row_visits[10]
        self.assertEqual(rare_visits, 3)
        self.assertEqual(common_visits, 5)

    def test_task_balanced_coverage_requires_task_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires source_task_id"):
            domain_coverage_capped_sampling_indices(
                [
                    {"domain": "large", "source_task_id": "large:0"},
                    {"domain": "large", "source_task_id": "large:1"},
                    {"domain": "small"},
                ],
                repeat_cap=2,
                seed=0,
                task_balanced=True,
            )

    def test_exposure_audit_reveals_max_steps_override(self) -> None:
        rows = (
            [
                {"domain": "retail", "source_task_id": f"retail:{index}"}
                for index in range(10)
            ]
            + [
                {"domain": "airline", "source_task_id": f"airline:{index}"}
                for index in range(2)
            ]
        )
        undertrained = summarize_sampling_exposure(
            rows,
            epochs=3,
            max_steps=1,
            world_size=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            seed=0,
            domain_coverage_repeat_cap=3,
            domain_coverage_task_balanced=True,
        )
        complete = summarize_sampling_exposure(
            rows,
            epochs=3,
            max_steps=-1,
            world_size=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            seed=0,
            domain_coverage_repeat_cap=3,
            domain_coverage_task_balanced=True,
        )

        self.assertTrue(undertrained["max_steps_overrides_epochs"])
        self.assertFalse(undertrained["all_source_rows_seen"])
        self.assertFalse(complete["max_steps_overrides_epochs"])
        self.assertTrue(complete["all_source_rows_seen"])
        self.assertEqual(complete["sampler_rows_per_epoch"], 16)
        self.assertEqual(complete["planned_sample_presentations"], 48)
        self.assertEqual(complete["planned_presentations_by_domain"], {
            "airline": 18,
            "retail": 30,
        })

    def test_domain_temperature_interpolates_expected_domain_mass(self) -> None:
        rows = [{"domain": "retail"}] * 9 + [{"domain": "airline"}]
        natural = domain_temperature_sampling_weights(rows, alpha=1.0)
        square_root = domain_temperature_sampling_weights(rows, alpha=0.5)
        balanced = domain_temperature_sampling_weights(rows, alpha=0.0)

        self.assertEqual(natural, [1.0] * 10)
        self.assertAlmostEqual(sum(balanced[:9]), balanced[9])
        self.assertAlmostEqual(sum(square_root[:9]) / square_root[9], 3.0)

    def test_domain_temperature_rejects_out_of_range_alpha(self) -> None:
        with self.assertRaisesRegex(ValueError, "alpha"):
            domain_temperature_sampling_weights(
                [{"domain": "retail"}], alpha=1.1
            )

    def test_explicit_domain_masses_set_expected_mass(self) -> None:
        rows = (
            [{"domain": "retail"}] * 4
            + [{"domain": "airline"}] * 2
            + [{"domain": "telecom"}]
        )
        weights = domain_target_sampling_weights(
            rows,
            masses={"retail": 0.1, "airline": 0.8, "telecom": 0.1},
        )

        self.assertAlmostEqual(sum(weights[:4]), 0.1)
        self.assertAlmostEqual(sum(weights[4:6]), 0.8)
        self.assertAlmostEqual(weights[6], 0.1)

    def test_explicit_domain_masses_require_complete_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing=.*telecom"):
            domain_target_sampling_weights(
                [{"domain": "airline"}, {"domain": "telecom"}],
                masses={"airline": 1.0},
            )

    def test_parse_domain_masses_rejects_duplicates_and_nonpositive(self) -> None:
        self.assertEqual(
            parse_domain_sampling_masses(["airline=8", "retail=1"]),
            {"airline": 8.0, "retail": 1.0},
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_domain_sampling_masses(["airline=1", "airline=2"])
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_domain_sampling_masses(["airline=0"])

    def test_learning_value_core_tail_preserves_domain_mass_and_tail(self) -> None:
        rows = [
            {
                "domain": "retail",
                "curation": {"mode": "learning_value", "rank_within_task": 1},
            },
            {
                "domain": "retail",
                "curation": {"mode": "learning_value", "rank_within_task": 5},
            },
            {
                "domain": "retail",
                "curation": {"mode": "learning_value", "rank_within_task": 6},
            },
            {
                "domain": "airline",
                "curation": {"mode": "learning_value", "rank_within_task": 2},
            },
            {
                "domain": "airline",
                "curation": {"mode": "learning_value", "rank_within_task": 8},
            },
        ]
        weights = learning_value_core_tail_sampling_weights(
            rows,
            core_rank=4,
            core_mass=0.75,
            domain_alpha=0.0,
        )

        self.assertAlmostEqual(sum(weights[:3]), 0.5)
        self.assertAlmostEqual(sum(weights[3:]), 0.5)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[3], weights[4])
        self.assertGreater(weights[1], 0.0)
        self.assertGreater(weights[4], 0.0)

    def test_learning_value_core_tail_requires_ranked_curation(self) -> None:
        with self.assertRaisesRegex(ValueError, "learning_value curation"):
            learning_value_core_tail_sampling_weights(
                [{"domain": "retail", "curation": {"mode": "other"}}],
                core_rank=4,
                core_mass=0.75,
            )

    def test_learning_value_task_balance_equalizes_tasks_and_keeps_tail(self) -> None:
        rows = [
            {
                "domain": "retail",
                "source_task_id": "easy",
                "curation": {"mode": "learning_value", "rank_within_task": 1},
            },
            {
                "domain": "retail",
                "source_task_id": "easy",
                "curation": {"mode": "learning_value", "rank_within_task": 2},
            },
            {
                "domain": "retail",
                "source_task_id": "easy",
                "curation": {"mode": "learning_value", "rank_within_task": 8},
            },
            {
                "domain": "retail",
                "source_task_id": "hard",
                "curation": {"mode": "learning_value", "rank_within_task": 1},
            },
            {
                "domain": "airline",
                "source_task_id": "only",
                "curation": {"mode": "learning_value", "rank_within_task": 1},
            },
            {
                "domain": "airline",
                "source_task_id": "only",
                "curation": {"mode": "learning_value", "rank_within_task": 7},
            },
        ]
        weights = learning_value_core_tail_sampling_weights(
            rows,
            core_rank=2,
            core_mass=0.75,
            domain_alpha=0.0,
            task_balanced=True,
        )

        self.assertAlmostEqual(sum(weights[:4]), 0.5)
        self.assertAlmostEqual(sum(weights[4:]), 0.5)
        self.assertAlmostEqual(sum(weights[:3]), 0.25)
        self.assertAlmostEqual(weights[3], 0.25)
        self.assertGreater(weights[0], weights[2])
        self.assertGreater(weights[4], weights[5])
        self.assertGreater(weights[2], 0.0)
        self.assertGreater(weights[5], 0.0)

    def test_learning_value_task_balance_requires_source_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires source_task_id"):
            learning_value_core_tail_sampling_weights(
                [
                    {
                        "domain": "retail",
                        "curation": {
                            "mode": "learning_value",
                            "rank_within_task": 1,
                        },
                    }
                ],
                core_rank=2,
                core_mass=0.75,
                task_balanced=True,
            )

    def test_render_messages_fallback_includes_tool_calls(self) -> None:
        text = render_messages(
            [
                {"role": "user", "content": "Lookup Paris"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "lookup_city", "arguments": {"city": "Paris"}},
                        }
                    ],
                },
            ]
        )

        self.assertIn("<|user|>", text)
        self.assertIn("Lookup Paris", text)
        self.assertIn("lookup_city", text)

    def test_row_to_text_uses_messages_when_present(self) -> None:
        text = row_to_text(
            {
                "messages": [
                    {"role": "user", "content": "A"},
                    {"role": "assistant", "content": "B"},
                ],
                "prompt": [{"role": "user", "content": "ignored"}],
            }
        )

        self.assertIn("A", text)
        self.assertIn("B", text)
        self.assertNotIn("ignored", text)

    def test_row_to_text_falls_back_to_prompt_response(self) -> None:
        text = row_to_text(
            {
                "prompt": [{"role": "user", "content": "A"}],
                "response": {"role": "assistant", "content": "B"},
            }
        )

        self.assertIn("A", text)
        self.assertIn("B", text)

    def test_row_to_text_prefers_explicit_response_over_stale_messages(self) -> None:
        text = row_to_text(
            {
                "messages": [
                    {"role": "user", "content": "A"},
                    {"role": "assistant", "content": "stale target"},
                ],
                "prompt": [{"role": "user", "content": "A"}],
                "response": {"role": "assistant", "content": "updated target"},
            }
        )

        self.assertIn("updated target", text)
        self.assertNotIn("stale target", text)

    def test_row_prompt_response_messages_prefers_explicit_prompt_response(self) -> None:
        prompt, response = row_prompt_response_messages(
            {
                "messages": [{"role": "user", "content": "ignored"}],
                "prompt": [{"role": "user", "content": "A"}],
                "response": [{"role": "assistant", "content": "B"}],
            }
        )

        self.assertEqual(prompt, [{"role": "user", "content": "A"}])
        self.assertEqual(response, [{"role": "assistant", "content": "B"}])

    def test_row_prompt_response_messages_falls_back_to_last_message(self) -> None:
        prompt, response = row_prompt_response_messages(
            {"messages": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]}
        )

        self.assertEqual(prompt, [{"role": "user", "content": "A"}])
        self.assertEqual(response, [{"role": "assistant", "content": "B"}])

    def test_row_to_prompt_text_omits_response(self) -> None:
        text = row_to_prompt_text(
            {
                "prompt": [{"role": "user", "content": "A"}],
                "response": [{"role": "assistant", "content": "B"}],
            }
        )

        self.assertIn("A", text)
        self.assertNotIn("B", text)

    def test_no_thinking_response_only_matches_inference_prefix(self) -> None:
        tokenizer = self.RecordingTokenizer()
        row = {
            "messages": [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
            ],
            "enable_thinking": False,
        }
        encoded = response_only_tokenized_row(row, tokenizer, max_length=4096)
        supervised = encoded["input_ids"][encoded["labels"] != -100]
        decoded = "".join(chr(int(token)) for token in supervised)
        prefix = "".join(
            chr(int(token))
            for token in encoded["input_ids"][: encoded["prompt_tokens"]]
        )
        self.assertTrue(prefix.endswith("<think>\n\n</think>\n\n"))
        self.assertEqual(decoded, "B<|im_end|>")
        self.assertTrue(
            any(call.get("enable_thinking") is False for call in tokenizer.calls)
        )

    def test_no_thinking_response_only_deduplicates_persisted_empty_think(self) -> None:
        tokenizer = self.RecordingTokenizer()
        row = {
            "messages": [
                {"role": "user", "content": "A"},
                {
                    "role": "assistant",
                    "content": "<think>\n\n</think>\n\nB",
                },
            ],
            "enable_thinking": False,
        }
        encoded = response_only_tokenized_row(row, tokenizer, max_length=4096)
        supervised = encoded["input_ids"][encoded["labels"] != -100]
        decoded = "".join(chr(int(token)) for token in supervised)
        self.assertEqual(decoded, "B<|im_end|>")

    def test_no_thinking_assistant_only_is_forwarded_to_native_template(self) -> None:
        tokenizer = self.AssistantOnlyTokenizer()
        row = {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ],
            "enable_thinking": False,
            "context_assistant_turns": 0,
            "supervised_assistant_turn_indices": [0],
        }

        encoded = assistant_only_tokenized_row(row, tokenizer, max_length=4096)

        self.assertTrue(any(call.get("enable_thinking") is False for call in tokenizer.calls))
        self.assertTrue(any(call.get("tools") == row["tools"] for call in tokenizer.calls))
        supervised = encoded["input_ids"][encoded["labels"] != -100]
        decoded = "".join(chr(int(token)) for token in supervised)
        self.assertEqual(decoded, "answer<|im_end|>")

    def test_assistant_only_handles_header_body_bpe_merge(self) -> None:
        tokenizer = self.BoundaryMergingTokenizer()
        row = {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "\nanswer"},
            ],
            "enable_thinking": False,
            "context_assistant_turns": 0,
            "supervised_assistant_turn_indices": [0],
        }

        encoded = assistant_only_tokenized_row(row, tokenizer, max_length=4096)

        self.assertTrue(tokenizer.assertion_target)
        self.assertGreater(int((encoded["labels"] != -100).sum()), 0)
        merged_position = (encoded["input_ids"] == 200_001).nonzero().flatten().tolist()
        self.assertEqual(len(merged_position), 1)
        self.assertEqual(int(encoded["labels"][merged_position[0]]), -100)

    def test_response_only_supervises_explicit_updated_response(self) -> None:
        tokenizer = self.RecordingTokenizer()
        row = {
            "messages": [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "stale"},
            ],
            "prompt": [{"role": "user", "content": "A"}],
            "response": {"role": "assistant", "content": "updated"},
            "enable_thinking": False,
        }
        encoded = response_only_tokenized_row(row, tokenizer, max_length=4096)
        supervised = encoded["input_ids"][encoded["labels"] != -100]
        decoded = "".join(chr(int(token)) for token in supervised)
        self.assertEqual(decoded, "updated<|im_end|>")

    def test_row_to_response_text_omits_prompt(self) -> None:
        text = row_to_response_text(
            {
                "prompt": [{"role": "user", "content": "A"}],
                "response": [{"role": "assistant", "content": "B"}],
            }
        )

        self.assertNotIn("A", text)
        self.assertIn("B", text)

    def test_summarize_rows_reports_distribution_and_lengths(self) -> None:
        summary = summarize_rows(
            [
                {
                    "messages": [{"role": "user", "content": "A"}],
                    "mode": "partial",
                    "action_class": "valid",
                    "branch_label": "valid_replay_exact",
                },
                {
                    "messages": [{"role": "user", "content": "B"}],
                    "mode": "partial",
                    "action_class": "ambiguous",
                    "branch_label": "ambiguous_unseen_query",
                    "hallucinated_observation": True,
                },
            ],
            max_preview=1,
        )

        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["by_mode"], {"partial": 2})
        self.assertEqual(summary["by_action_class"], {"ambiguous": 1, "valid": 1})
        self.assertEqual(summary["hallucinated_observations"], 1)
        self.assertEqual(len(summary["previews"]), 1)
        self.assertGreater(summary["text_chars_max"], 0)


if __name__ == "__main__":
    unittest.main()

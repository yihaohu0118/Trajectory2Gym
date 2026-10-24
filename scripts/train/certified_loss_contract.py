"""Sparse ordinary CE with an explicit distributed accumulation denominator."""

from __future__ import annotations

from typing import Any, Callable


def sparse_loss_mixin(
    loss_function: Callable,
    *,
    enabled: bool,
    normalization: str = "legacy_local_token_mean",
    activation_offload_cpu: bool = False,
    activation_offload_pin_memory: bool = True,
    ce_checkpoint_chunk_size: int = 0,
    activation_offload_min_input_tokens: int = 0,
):
    if normalization not in {"legacy_local_token_mean", "global_batch_token_mean", "global_batch_parent_mean"}:
        raise ValueError(f"unsupported SFT normalization: {normalization}")
    if activation_offload_min_input_tokens < 0:
        raise ValueError('offload input-token threshold must be nonnegative')

    class SparseSupervisedLossMixin:
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if enabled and normalization != "legacy_local_token_mean":
                # Trainer must not divide again by accumulation steps: the
                # denominator below already spans that entire optimizer step.
                self.model_accepts_loss_kwargs = True
                if self.accelerator.gradient_accumulation_steps != 1:
                    raise ValueError("Trainer, not Accelerator, must own accumulation")

        def _get_num_items_in_batch(self, batch_samples, device):
            if not enabled or normalization == "legacy_local_token_mean":
                return super()._get_num_items_in_batch(batch_samples, device)
            if not batch_samples:
                return None
            import torch

            # Count prediction targets AFTER causal shifting. Label column 0
            # has no predecessor and must not silently enter the denominator.
            if normalization == "global_batch_parent_mean":
                if any(batch['labels'].shape[0] != 1 for batch in batch_samples):
                    raise ValueError('parent control requires one rendered row per microbatch')
                count = len(batch_samples)
            else:
                count = sum(int(batch["labels"][..., 1:].ne(-100).sum()) for batch in batch_samples)
            total = torch.tensor(count, device=device, dtype=torch.int64)
            total = self.accelerator.gather(total).sum()
            if total.item() <= 0:
                raise ValueError("optimizer batch contains no certified causal targets")
            return total

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if not enabled:
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            denominator: Any = None
            multiplier = 1
            if normalization == "global_batch_token_mean":
                if "loss_weights" in inputs:
                    raise ValueError("uniform certified-token CE cannot use row loss weights")
                denominator = num_items_in_batch
                if denominator is not None:
                    # DDP/FSDP average rank gradients. Compensate exactly once
                    # because the token denominator was summed across ranks.
                    multiplier = self.accelerator.num_processes
            elif normalization == 'global_batch_parent_mean':
                if inputs['labels'].shape[0] != 1 or 'loss_weights' not in inputs:
                    raise ValueError('parent control requires one row and an audited parent weight')
                if num_items_in_batch is not None:
                    multiplier = self.accelerator.num_processes / num_items_in_batch
            use_offload = activation_offload_cpu and (
                activation_offload_min_input_tokens == 0
                or inputs['input_ids'].numel() >= activation_offload_min_input_tokens
            )
            counts = getattr(self,'_activation_placement_counts',{'cpu':0,'gpu':0})
            counts['cpu' if use_offload else 'gpu'] += 1
            self._activation_placement_counts = counts
            loss, outputs = loss_function(
                model, inputs, num_items_in_batch=denominator,
                activation_offload_cpu=use_offload,
                activation_offload_pin_memory=activation_offload_pin_memory,
                **({'ce_checkpoint_chunk_size':ce_checkpoint_chunk_size} if ce_checkpoint_chunk_size else {}),
            )
            loss = loss * multiplier
            return (loss, outputs) if return_outputs else loss

    return SparseSupervisedLossMixin

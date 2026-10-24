from __future__ import annotations

import re
from typing import Sequence


_THINKING_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_THINKING_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE | re.DOTALL)


def strip_visible_thinking(content: str) -> str:
    """Remove model-private Qwen-style thinking from externally visible text."""

    text = str(content or "")
    stripped = _THINKING_BLOCK_RE.sub("", text)
    if "</think" in stripped.lower():
        stripped = re.sub(r"(?is)^.*?</think\s*>", "", stripped)
    if _THINKING_OPEN_RE.search(stripped):
        stripped = _THINKING_OPEN_RE.split(stripped, maxsplit=1)[0]
    return stripped.strip()


def reasoning_token_loss_mask(token_ids: list[int], tokenizer) -> list[int]:
    """Return a mask that excludes ``<think>`` spans and orphan close tags.

    The function operates on the exact sampled token IDs rather than
    re-tokenizing the visible completion.  That keeps PPO log-prob alignment
    intact while preventing a non-thinking run from reinforcing accidental
    private-reasoning syntax.  Unterminated blocks fail closed by masking the
    remainder of the sampled response; visible tokens after a close tag remain
    trainable.
    """

    ids = list(token_ids)

    def encode(text: str) -> list[int]:
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))

    opening = encode("<think>")
    closing = encode("</think>")
    if not opening or not closing:
        raise ValueError("tokenizer produced an empty reasoning delimiter")

    mask = [1] * len(ids)
    inside = False
    index = 0
    while index < len(ids):
        if ids[index : index + len(opening)] == opening:
            for position in range(index, min(len(ids), index + len(opening))):
                mask[position] = 0
            inside = True
            index += len(opening)
            continue
        if ids[index : index + len(closing)] == closing:
            for position in range(index, min(len(ids), index + len(closing))):
                mask[position] = 0
            inside = False
            index += len(closing)
            continue
        if inside:
            mask[index] = 0
        index += 1
    return mask


def align_sampled_completion_tokens(
    *,
    template_increment: Sequence[int],
    generation_prompt: Sequence[int],
    sampled_tokens: Sequence[int],
    eos_token_id: int,
) -> tuple[list[int], int | None, list[int]]:
    """Insert sampled tokens into a chat-template completion increment.

    Returns ``(token_ids, sampled_start, sampled_content)``.  A ``None`` start
    means the tokenizer's generation wrapper could not be authenticated in
    the completed-message template.  In that case the structurally valid
    re-tokenized template increment is returned unchanged and callers must
    mask the whole turn from policy optimization.  This fail-closed fallback
    keeps one malformed role/template boundary from aborting an otherwise
    valid rollout batch or, worse, training on misaligned vLLM tokens.
    """

    template = list(template_increment)
    prompt = list(generation_prompt)
    sampled = list(sampled_tokens)

    if sampled[-1:] == [eos_token_id]:
        sampled = sampled[:-1]
    if prompt and sampled[: len(prompt)] == prompt:
        sampled = sampled[len(prompt) :]

    def find_subsequence(
        sequence: list[int], subsequence: list[int], *, reverse: bool = False
    ) -> int:
        if not subsequence:
            return -1
        indices = range(len(sequence) - len(subsequence) + 1)
        if reverse:
            indices = reversed(indices)
        for index in indices:
            if sequence[index : index + len(subsequence)] == subsequence:
                return index
        return -1

    prompt_index = find_subsequence(template, prompt)
    eos_index = find_subsequence(template, [eos_token_id], reverse=True)
    sampled_start = prompt_index + len(prompt)
    if (
        prompt_index < 0
        or eos_index < 0
        or eos_index < sampled_start
    ):
        return template, None, []

    aligned = template[:sampled_start] + sampled + template[eos_index:]
    return aligned, sampled_start, sampled

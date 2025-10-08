"""
DoLa runtime utilities (static, no Early Exit).

This module provides minimal, model-agnostic functions to apply DoLa on top of
an existing head (e.g., LM `lm_head` or image-token `gen_head`) without
modifying any backbone architecture/weights.

All functions are pure tensor ops; callers pass in logits/hidden as tensors and
the appropriate projection head callable.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _as_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Numerically-stable log_softmax on last dim.

    Args:
        logits: [..., vocab]
    Returns:
        [..., vocab] log-probabilities that sum to 1 in prob space.
    """
    return F.log_softmax(logits, dim=-1)


@torch.no_grad()
def contrast_with_apc_from_logits(
    final_logits: torch.Tensor,
    mid_logits: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Apply DoLa-static on two logits tensors with APC masking.

    final_logits and mid_logits must have identical shapes [..., vocab].

    Steps per row:
      - q_N = softmax(final)
      - head set V = {x: q_N(x) >= alpha * max(q_N)}
      - F = log_softmax(final) - log_softmax(mid)
      - outside V -> -inf (large negative), keep inside V unchanged

    Returns:
        F_masked logits (same shape), suitable to be used as logits downstream.
    """
    assert final_logits.shape == mid_logits.shape, "shape mismatch"

    # log-probs for ratio; using log_softmax is more stable than softmax then log
    log_qN = _as_log_softmax(final_logits)
    log_qJ = _as_log_softmax(mid_logits)
    F_contrast = log_qN - log_qJ  # [..., vocab]

    # APC head mask computed from q_N per row
    # Threshold: alpha * max(q_N) <=> log threshold = log max(q_N) + log(alpha)
    with torch.enable_grad():
        # torch.no_grad above, but we explicitly don't need autograd here.
        pass
    qN = log_qN.exp()
    max_qN, _ = qN.max(dim=-1, keepdim=True)
    thr = alpha * max_qN
    head_mask = qN >= thr  # [..., vocab] boolean

    # Mask outside head to a large negative number to emulate -inf
    very_neg = torch.finfo(final_logits.dtype).min if final_logits.dtype.is_floating_point else -1e9
    # In practice, using a finite large negative stabilizes downstream softmax
    very_neg = -1e9 if not torch.isfinite(torch.tensor(very_neg)) else max(very_neg, -1e9)

    F_masked = torch.where(head_mask, F_contrast, torch.tensor(very_neg, device=F_contrast.device, dtype=F_contrast.dtype))
    return F_masked


@torch.no_grad()
def dola_static_from_hidden(
    hidden_states: tuple[torch.Tensor, ...],
    head_module,  # callable: (hidden) -> logits
    j_index: int,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Compute DoLa-static logits from a tuple of hidden states.

    Args:
        hidden_states: tuple length L (+1 if embeddings are included). Each element
            is [B, T, H]. The last element corresponds to the final layer output.
        head_module: projection head that maps hidden [B, H] -> logits [B, V].
        j_index: index into `hidden_states` for the fixed mid layer (supports
            negative indexing). If `hidden_states` includes the embeddings as the
            first element, choose accordingly (e.g., in HF LLaMA it's len=L+1).
        alpha: APC threshold in probability space.

    Returns:
        DoLa-static logits (F masked) for the last position: [B, V]
    """
    # final layer last token
    hN_last = hidden_states[-1][:, -1, :]  # [B, H]

    # fixed mid-layer last token
    hj_last = hidden_states[j_index][:, -1, :]  # [B, H]

    final_logits = head_module(hN_last)  # [B, V]
    mid_logits = head_module(hj_last)    # [B, V]

    F_masked = contrast_with_apc_from_logits(final_logits, mid_logits, alpha=alpha)
    return F_masked


def pick_static_layer_index(num_hidden_states: int, has_embedding: bool, ratio: float = 0.8) -> int:
    """Pick a fixed mid-layer index j* given tuple length and whether the first
    element is the embeddings output.

    Args:
        num_hidden_states: len(hidden_states) tuple at runtime.
        has_embedding: True if hidden_states[0] is embeddings (HF style).
        ratio: relative position (0<ratio<1). 0.8 means near the back.

    Returns:
        j_index to use for `hidden_states[j_index]`.
    """
    assert 0.0 < ratio < 1.0
    # If includes embeddings, layers count = L = n_layers+1; valid mid layers are 1..L-2
    if has_embedding:
        first, last = 1, num_hidden_states - 2
    else:
        first, last = 0, num_hidden_states - 2
    if last < first:
        # Fallback: use penultimate element
        return num_hidden_states - 2
    j = int(first + ratio * (last - first))
    j = max(first, min(last, j))
    return j


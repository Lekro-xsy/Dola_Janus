"""
DoLa runtime utilities (Early-Exit + APC + contrastive fusion) for both
vision and language sides. These functions do not modify backbone modules.

Notes:
- Language side expects access to the language model's final norm and head.
- Vision side expects already-computed representations and a text bank.
- We purposely keep this file self-contained and import-free (beyond torch)
  so it can be reused in different scripts without touching model code.
"""

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically-stable log-softmax wrapper."""
    return F.log_softmax(logits, dim=dim)


def _safe_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.softmax(logits, dim=dim)


def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Jensen-Shannon divergence between two probability distributions.

    Args:
        p: [..., V]
        q: [..., V]
    Returns:
        jsd value(s): [...]
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_p = (p * (p.log() - m.log())).sum(dim=-1)
    kl_q = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_p + kl_q)


def select_early_exit(qN: torch.Tensor, q_list: List[torch.Tensor]) -> int:
    """Select the early-exit layer that maximizes JSD(qN || qj).

    Args:
        qN: [B, V]
        q_list: list of [B, V]
    Returns:
        index into q_list
    """
    assert len(q_list) > 0, "q_list must be non-empty"
    # Stack to [L, B, V]
    Q = torch.stack(q_list, dim=0)
    # Compute JSD per layer: [L, B]
    jsd_vals = jsd(qN.unsqueeze(0).expand_as(Q), Q)
    # Choose the max per batch, then use the most frequent index (or 0th) —
    # for simplicity we take argmax on the mean across batch.
    mean_jsd = jsd_vals.mean(dim=1)
    idx = int(torch.argmax(mean_jsd).item())
    return idx


def contrast_and_mask(
    logqN: torch.Tensor,
    logqM: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """APC mask + contrastive logit (log qN - log qM).

    Args:
        logqN: [B, V] log-probs from the final layer
        logqM: [B, V] log-probs from the selected early-exit layer
        alpha: APC threshold relative to max prob of qN
    Returns:
        F_masked: [B, V] masked contrastive logits
    """
    # Contrastive score in log domain
    F_contrast = logqN - logqM

    # APC mask computed from qN
    # threshold = log(alpha * max p) = log(alpha) + max(log p)
    max_logqN, _ = logqN.max(dim=-1, keepdim=True)
    thr = torch.log(torch.tensor(alpha, device=logqN.device)) + max_logqN
    mask = logqN >= thr

    # Set -inf where masked out
    neg_inf = torch.tensor(-1e9, device=F_contrast.device, dtype=F_contrast.dtype)
    F_masked = torch.where(mask, F_contrast, neg_inf)
    return F_masked


def compute_probs_from_hidden(
    h: torch.Tensor,
    head: nn.Module,
    ln_out: Optional[nn.Module] = None,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute probabilities and logits from a hidden state.

    Args:
        h: [B, D]
        head: mapping to vocabulary (e.g., lm_head or gen_head)
        ln_out: optional final layer norm; if None, h is used directly
        temperature: softmax temperature
    Returns:
        (probs, logits): both [B, V]
    """
    if ln_out is not None:
        h = ln_out(h)
    logits = head(h)
    if temperature != 1.0:
        logits = logits / temperature
    probs = _safe_softmax(logits, dim=-1)
    return probs, logits


def apply_dola_on_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    head: nn.Module,
    ln_out: Optional[nn.Module],
    candidate_layer_ids: Iterable[int],
    temperature: float = 1.0,
    alpha: float = 0.1,
    rV_log_bias: Optional[torch.Tensor] = None,
    lambda_fuse: float = 0.0,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """Apply DoLa on a set of hidden states for the last position.

    Args:
        hidden_states: HF-style hidden_states tuple, len = N_layers+1, each [B, T, D]
        head: vocabulary projection module
        ln_out: final layer norm (or None if not needed)
        candidate_layer_ids: indices in [0..N_layers-1] to compare against the last layer
        temperature: softmax temperature
        alpha: APC threshold factor
        rV_log_bias: optional visual prior in log domain, shape [V] or [B, V]
        lambda_fuse: fusion strength for rV
    Returns:
        probs_joint: [B, V], selected_idx, logqN: [B, V], logqM: [B, V]
    """
    assert len(hidden_states) >= 2, "Expect embeddings + at least one layer"
    # Final layer (last element) at last position
    hN = hidden_states[-1][:, -1, :]
    qN, logitsN = compute_probs_from_hidden(hN, head=head, ln_out=ln_out, temperature=temperature)
    logqN = _safe_log_softmax(logitsN, dim=-1)

    # Collect candidate early-exit distributions
    q_list: List[torch.Tensor] = []
    logq_list: List[torch.Tensor] = []
    # hidden_states index: 0 -> embeddings, 1..N -> layer outputs
    for j in candidate_layer_ids:
        hs = hidden_states[1 + j][:, -1, :]
        qj, logits_j = compute_probs_from_hidden(hs, head=head, ln_out=ln_out, temperature=temperature)
        q_list.append(qj)
        logq_list.append(_safe_log_softmax(logits_j, dim=-1))

    # Early-exit selection by JSD
    sel_idx = select_early_exit(qN, q_list)
    logqM = logq_list[sel_idx]

    # Contrastive logit + APC mask
    F_masked = contrast_and_mask(logqN, logqM, alpha=alpha)

    # Optional visual fusion in log domain (additive bias)
    if rV_log_bias is not None and lambda_fuse != 0.0:
        if rV_log_bias.dim() == 1:
            F_masked = F_masked + lambda_fuse * rV_log_bias.unsqueeze(0)
        else:
            F_masked = F_masked + lambda_fuse * rV_log_bias

    probs_joint = _safe_softmax(F_masked, dim=-1)
    return probs_joint, sel_idx, logqN, logqM


def build_probs_from_rep(
    z: torch.Tensor,
    text_bank: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Build vision-side distribution q(c) from image rep z and text bank.

    Args:
        z: [B, D]
        text_bank: [C, D]
        tau: temperature (scale for dot product)
    Returns:
        probs: [B, C]
    """
    # cosine-like by scaling; assume z and text_bank already normalized if desired
    scores = (z @ text_bank.T) / tau
    return _safe_softmax(scores, dim=-1)


def pick_candidate_layers(n_layers: int, k: int = 5) -> List[int]:
    """Heuristic: select k approximately-evenly spaced, even-numbered layers in [0, n_layers-1]."""
    k = max(1, min(k, n_layers))
    # Evenly spaced indices
    idxs = torch.linspace(0, n_layers - 1, steps=k).round().to(torch.int64).tolist()
    # Prefer even
    idxs = [int(i - (i % 2)) for i in idxs]
    # Deduplicate & clamp
    idxs = sorted(set(max(0, min(n_layers - 1, i)) for i in idxs))
    return idxs


__all__ = [
    "jsd",
    "select_early_exit",
    "contrast_and_mask",
    "compute_probs_from_hidden",
    "apply_dola_on_hidden_states",
    "build_probs_from_rep",
    "pick_candidate_layers",
]


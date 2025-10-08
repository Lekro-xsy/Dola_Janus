"""
DoLa runtime utilities (pure functions).

Implements the math described in modify1.md for Decoding by Contrasting Layers
with Early Exit selection. All functions operate on tensors and do not depend
on Janus model classes to keep coupling minimal.

Notes:
- We interpret the "vocabulary" as the candidate class/text set C.
- Inputs are expected as torch.Tensor on the same device and dtype.
- All softmax/log/ratio operations are numerically stabilized.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def build_probs_from_rep(
    z: torch.Tensor, text_bank: torch.Tensor, tau: float = 1.0
) -> torch.Tensor:
    """
    Compute q(c) = softmax(<z, t_c> / tau) over candidates.

    Args:
        z: [B, D] image representation after pooling/head pre-logits.
        text_bank: [C, D] candidate (text) embeddings; precomputed with the
            paired text encoder so the space matches the vision head.
        tau: temperature.

    Returns:
        probs: [B, C] probability distribution for each sample in the batch.
    """
    assert z.dim() == 2 and text_bank.dim() == 2, "z and text_bank must be 2D"
    assert z.size(1) == text_bank.size(1), "rep/text dim mismatch"

    # Scores s = <z, t_c> / tau
    scores = (z @ text_bank.t()) / max(tau, 1e-6)
    probs = F.softmax(scores, dim=-1)
    return probs


def _kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Element-wise KL divergence sum_i p_i * log(p_i/q_i). Returns per-sample KL.

    p, q: [B, C] valid distributions.
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Jensen–Shannon divergence between two distributions.

    Args:
        p, q: [B, C] distributions (sum to 1 across -1).
        eps: numerical floor.

    Returns:
        jsd: [B] per-sample JSD.
    """
    m = 0.5 * (p + q)
    return 0.5 * _kl_div(p, m, eps) + 0.5 * _kl_div(q, m, eps)


def select_early_exit(qN: torch.Tensor, q_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Select M = argmax_j JSD(qN || qj) for each sample.

    Args:
        qN: [B, C] distribution from the final layer.
        q_list: list of [B, C] distributions for each candidate layer j.

    Returns:
        m_idx: [B] long tensor of chosen indices into q_list.
    """
    assert len(q_list) > 0, "q_list should not be empty"
    # Stack: [J, B, C]
    q_stack = torch.stack(q_list, dim=0)
    # Compute JSD per layer per sample → [J, B]
    jsd_vals = []
    for j in range(q_stack.size(0)):
        jsd_vals.append(jsd(qN, q_stack[j]))
    jsd_mat = torch.stack(jsd_vals, dim=0)  # [J, B]
    m_idx = torch.argmax(jsd_mat, dim=0)  # [B]
    return m_idx


def contrast_and_mask(
    qN: torch.Tensor,
    qM: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Apply APC head mask and DoLa contrast: F(c) = log qN(c) - log qM(c),
    then softmax over the head-visible set.

    Args:
        qN: [B, C] final layer distribution.
        qM: [B, C] early-exit chosen layer distribution (aligned with qN).
        alpha: head threshold factor in [0, 1].

    Returns:
        p_hat: [B, C] normalized over visible candidates; masked entries are 0.
    """
    eps = 1e-12
    # Head set mask: qN(c) >= alpha * max(qN)
    max_qN = qN.max(dim=-1, keepdim=True).values
    visible = qN >= (alpha * max_qN)

    # Log-ratio with numerical floor
    Fv = torch.log(qN.clamp_min(eps)) - torch.log(qM.clamp_min(eps))
    Fv = torch.where(visible, Fv, torch.full_like(Fv, float("-inf")))

    # Normalize within visible set; fill masked with 0 after softmax
    p_hat = F.softmax(Fv, dim=-1)
    p_hat = torch.where(visible, p_hat, torch.zeros_like(p_hat))
    # Re-normalize in case of numerical drift (sum over visible only)
    z = p_hat.sum(dim=-1, keepdim=True).clamp_min(eps)
    p_hat = p_hat / z
    return p_hat


# Convenience API for single-sample DoLa given qN and candidate qj list.
def dola_from_distributions(
    qN: torch.Tensor,
    q_candidates: List[torch.Tensor],
    alpha: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        qN: [B, C]
        q_candidates: list of [B, C]
        alpha: float

    Returns:
        p_hat: [B, C]
        m_idx: [B]
    """
    m_idx = select_early_exit(qN, q_candidates)
    # Gather qM per sample
    # Stack candidates [J, B, C] → index with m_idx for each B
    q_stack = torch.stack(q_candidates, dim=0)
    J, B, C = q_stack.shape
    arangeB = torch.arange(B, device=q_stack.device)
    qM = q_stack[m_idx, arangeB, :]
    p_hat = contrast_and_mask(qN, qM, alpha=alpha)
    return p_hat, m_idx


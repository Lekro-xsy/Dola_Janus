"""
DoLa-static runtime utilities for SigLIP ViT.

This module adds small, self-contained helpers as specified in modify2.md:
- build_probs_from_rep(z, text_bank, tau): compute class distribution from a
  representation and a bank of text embeddings.
- contrast_and_mask(qN, qJ, alpha): DoLa APC head filtering and log-ratio
  fusion restricted to head set.
- rerank_with_dola_static_siglip(...): convenience wrapper wiring SigLIP ViT
  intermediate layer and final layer through forward_head to produce the
  DoLa-static re-scored distribution over text candidates.

We only add logic; no changes to existing model classes/weights.
"""

from typing import Iterable, Tuple

import torch
import torch.nn.functional as F


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is [B, C] or [B, D]; add batch dim if needed."""
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


@torch.no_grad()
def build_probs_from_rep(
    z: torch.Tensor,
    text_bank: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Compute q(c) = softmax(<z, t_c>/tau) over a bank of text embeddings.

    Args:
        z: image representation, shape [D] or [B, D]. It is L2-normalized inside.
        text_bank: text candidate embeddings, shape [C, D]. It is L2-normalized inside.
        tau: temperature scalar.

    Returns:
        probs: [B, C] probability distribution(s) over candidates.
    """
    z = _ensure_2d(z)
    assert text_bank.dim() == 2, "text_bank must be [C, D]"

    # Normalize to cosine similarity space
    z = F.normalize(z, dim=-1)
    t = F.normalize(text_bank, dim=-1)

    # Scores and probabilities
    s = (z @ t.t()) / max(tau, 1e-6)  # [B, C]
    q = F.softmax(s, dim=-1)
    return q


@torch.no_grad()
def contrast_and_mask(
    qN: torch.Tensor,
    qJ: torch.Tensor,
    alpha: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DoLa head filtering (APC) + log-ratio fusion restricted to the head set.

    F(c) = log qN(c) - log qJ(c) on head set V_head = { c | qN(c) >= alpha * max(qN) };
    outside V_head is masked to -inf; softmax over V_head to get p_hat.

    Args:
        qN: [B, C] mature (final layer) distribution(s)
        qJ: [B, C] early (fixed layer j*) distribution(s)
        alpha: head filtering ratio (0.1 by default)

    Returns:
        p_hat: [B, C] final DoLa-static distribution(s)
        mask:  [B, C] boolean mask indicating V_head per-row
    """
    qN = _ensure_2d(qN)
    qJ = _ensure_2d(qJ)
    assert qN.shape == qJ.shape, "qN and qJ must have the same shape"

    B, C = qN.shape
    eps = 1e-9

    # Head set per sample
    max_qN = qN.max(dim=-1, keepdim=True).values  # [B, 1]
    thresh = alpha * max_qN
    mask = qN >= thresh  # [B, C]

    # Log-ratio only on the head set; others set to -inf
    log_qN = torch.log(qN.clamp_min(eps))
    log_qJ = torch.log(qJ.clamp_min(eps))
    Fscore = log_qN - log_qJ  # [B, C]

    # Masking: set outside head to -inf to exclude from softmax
    neg_inf = torch.finfo(Fscore.dtype).min
    Fmasked = torch.where(mask, Fscore, torch.full_like(Fscore, neg_inf))

    # Normalize on the head set
    p_hat = F.softmax(Fmasked, dim=-1)
    return p_hat, mask


@torch.no_grad()
def rerank_with_dola_static_siglip(
    vit,  # janus.models.siglip_vit.VisionTransformer (vision tower)
    image_tensor: torch.Tensor,
    text_bank: torch.Tensor,
    fixed_layer: int = 18,
    alpha: float = 0.1,
    tau: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience end-to-end DoLa-static for SigLIP ViT.

    Steps:
      1) Final-layer features -> forward_head(pre_logits=True) -> z_N
      2) get_intermediate_layers(..., indices=[j*], norm=True, return_prefix_tokens=True)
         reconstruct full sequence -> forward_head(pre_logits=True) -> z_j*
      3) q_N, q_j* via build_probs_from_rep; fuse with contrast_and_mask.

    Args:
        vit: VisionTransformer instance (ignore_head=True is fine)
        image_tensor: [B, 3, H, W]
        text_bank: [C, D] text embeddings
        fixed_layer: 0-based or 1-based layer index (1..N). If in [1..N], it will
                     be converted to 0-based automatically.
        alpha: APC head ratio
        tau: temperature

    Returns:
        p_hat: [B, C] DoLa-static fused distribution(s)
        qN:    [B, C] final-layer baseline distribution(s)
        qJ:    [B, C] early-layer distribution(s) at j*
    """
    vit.eval()

    # Final layer features -> pooled rep
    xN = vit.forward_features(image_tensor)  # [B, T, D]
    zN = vit.forward_head(xN, pre_logits=True)  # [B, D]

    # Resolve j* to 0-based index within [0, depth-1]
    depth = len(vit.blocks)
    j = fixed_layer
    if j >= 1 and j <= depth:
        j = j - 1  # convert 1-based -> 0-based
    j = max(0, min(j, depth - 1))

    # Intermediate layer features; need prefix tokens to feed forward_head
    outs = vit.get_intermediate_layers(
        image_tensor, n=[j], reshape=False, return_prefix_tokens=True, norm=True
    )
    # outs is a tuple of length 1; each item is (out_wo_prefix, prefix_tokens)
    out_wo_prefix, prefix_tokens = outs[0]
    # Reconstruct full sequence
    xJ = torch.cat([prefix_tokens, out_wo_prefix], dim=1)  # [B, T, D]
    zJ = vit.forward_head(xJ, pre_logits=True)  # [B, D]

    # Build distributions and fuse
    qN = build_probs_from_rep(zN, text_bank, tau=tau)
    qJ = build_probs_from_rep(zJ, text_bank, tau=tau)
    p_hat, _ = contrast_and_mask(qN, qJ, alpha=alpha)
    return p_hat, qN, qJ


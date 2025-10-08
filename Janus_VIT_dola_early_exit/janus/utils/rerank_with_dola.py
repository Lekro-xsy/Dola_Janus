"""
SigLIP-specific DoLa reranking utilities.

This module wires VisionTransformer intermediate layers to DoLa runtime, without
modifying the ViT class itself. It expects that the provided VisionTransformer
implements:
- get_intermediate_layers(x, n|indices, norm=True, return_prefix_tokens=True)
- forward_features(x)
- forward_head(x, pre_logits=True)

The primary entry point is `rerank_with_dola_siglip` which computes the final
DoLa-distribution over the provided text_bank for each input image.
"""

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch

from .dola_runtime import build_probs_from_rep, dola_from_distributions


def _pool_rep_from_tokens(vit, tokens: torch.Tensor) -> torch.Tensor:
    """Pool token sequence into a single representation using model's head path.

    Args:
        vit: VisionTransformer instance
        tokens: [B, T, D]
    Returns:
        reps: [B, D]
    """
    # Use pre_logits=True to get pooled representation before any classification head.
    return vit.forward_head(tokens, pre_logits=True)


def _final_rep(vit, images: torch.Tensor) -> torch.Tensor:
    """Compute final layer pooled representation z_N for images.

    Args:
        vit: VisionTransformer
        images: [B, 3, H, W]
    Returns:
        reps: [B, D]
    """
    tokens = vit.forward_features(images)  # already norm-ed inside
    reps = _pool_rep_from_tokens(vit, tokens)
    return reps


def _intermediate_reps(
    vit,
    images: torch.Tensor,
    candidate_layers: Sequence[int],
) -> List[torch.Tensor]:
    """Collect pooled representations z_j for candidate layers.

    Args:
        vit: VisionTransformer
        images: [B, 3, H, W]
        candidate_layers: 1-based indices into transformer blocks (per paper)

    Returns:
        reps_list: list of [B, D] for each candidate layer j
    """
    if len(candidate_layers) == 0:
        raise ValueError("candidate_layers is empty")

    # Map to 0-based block indices expected by get_intermediate_layers
    idx_0based = [max(0, i - 1) for i in candidate_layers]
    outs = vit.get_intermediate_layers(
        images, n=idx_0based, reshape=False, return_prefix_tokens=True, norm=True
    )
    reps_list: List[torch.Tensor] = []
    for out_wo_prefix, prefix in outs:
        # Reattach prefix tokens so forward_head sees the original layout
        tokens = torch.cat([prefix, out_wo_prefix], dim=1)
        reps_list.append(_pool_rep_from_tokens(vit, tokens))
    return reps_list


@torch.inference_mode()
def rerank_with_dola_siglip(
    vit,
    images: torch.Tensor,
    text_bank: torch.Tensor,
    *,
    alpha: float = 0.1,
    tau: float = 1.0,
    candidate_layers: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute DoLa-contrasted distribution over candidates for each image.

    Args:
        vit: SigLIP VisionTransformer instance (no modifications needed).
        images: [B, 3, H, W] normalized as expected by the vision tower.
        text_bank: [C, D] candidate embeddings (paired text space).
        alpha: APC threshold.
        tau: temperature for similarity softmax.
        candidate_layers: sequence of 1-based layer indices to consider for early exit.
            If None, defaults to bucket [17, 24] with even layers {18, 20, 22}.

    Returns:
        p_hat: [B, C] DoLa final probabilities.
        m_idx: [B] chosen candidate-layer index position (0..len(candidate_layers)-1).
    """
    if candidate_layers is None:
        candidate_layers = (18, 20, 22)

    # Final layer distribution q_N
    zN = _final_rep(vit, images)
    qN = build_probs_from_rep(zN, text_bank, tau=tau)

    # Candidate early-exit distributions {q_j}
    reps_list = _intermediate_reps(vit, images, candidate_layers)
    q_list = [build_probs_from_rep(r, text_bank, tau=tau) for r in reps_list]

    # DoLa fusion
    p_hat, m_idx = dola_from_distributions(qN, q_list, alpha=alpha)
    return p_hat, m_idx


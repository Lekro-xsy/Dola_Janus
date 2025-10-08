"""
Cross-modal fusion utilities for DoLa: map visual candidates to LM tokens and
compute a log-domain prior r_V(x) that can be added to language-side DoLa score.

We keep mapping extremely simple by default: each candidate string maps to its
starting BPE token id (set of size 1). You can plug in a richer alias dictionary
or multi-token starts if needed.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def build_token_mapping(
    tokenizer,
    candidates: Sequence[str],
    use_bos: bool = False,
) -> Dict[str, List[int]]:
    """Map each candidate phrase to a list of starting token ids.

    Args:
        tokenizer: HF tokenizer
        candidates: list of candidate strings
        use_bos: if True, prepend BOS to avoid special merges affecting first id
    Returns:
        dict: {candidate -> [token_id, ...]}
    """
    mapping: Dict[str, List[int]] = {}
    for c in candidates:
        text = (tokenizer.bos_token or "") + c if use_bos else c
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        # By default only take the first token id as the phrase "starter".
        mapping[c] = [ids[0]]
    return mapping


def visual_prior_log_bias(
    vocab_size: int,
    candidates: Sequence[str],
    pV: torch.Tensor,
    token_map: Dict[str, List[int]],
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Compute r_V(x) in log-domain as log( sum_{c: x in T(c)} pV(c) ).

    Args:
        vocab_size: LM vocab size
        candidates: list of candidate strings, length C
        pV: [C] or [B, C] visual distribution
        token_map: mapping {candidate -> [token_id,...]}
        fill_value: log-bias for tokens not covered; 0.0 means neutral
    Returns:
        rV: [V] or [B, V]
    """
    device = pV.device
    if pV.dim() == 1:
        r = torch.full((vocab_size,), fill_value=fill_value, device=device)
        for i, c in enumerate(candidates):
            ids = token_map.get(c, [])
            if not ids:
                continue
            mass = pV[i].clamp_min(1e-20)
            val = torch.log(mass)
            for t in ids:
                r[t] = torch.logaddexp(r[t], val)
        return r
    else:
        B = pV.size(0)
        r = torch.full((B, vocab_size), fill_value=fill_value, device=device)
        for i, c in enumerate(candidates):
            ids = token_map.get(c, [])
            if not ids:
                continue
            mass = pV[:, i].clamp_min(1e-20).log().unsqueeze(-1)  # [B, 1]
            for t in ids:
                r[:, t] = torch.logaddexp(r[:, t], mass.squeeze(-1))
        return r


__all__ = [
    "build_token_mapping",
    "visual_prior_log_bias",
]


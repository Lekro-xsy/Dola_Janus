"""
DoLa-static runtime utilities for LLaMA (no early-exit, fixed contrast layer).

This module adds minimal, non-intrusive helpers to compute probabilities from
intermediate hidden states and to fuse distributions using the DoLa formula
with APC (adaptive head constraint). It does not modify any model classes.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compute_probs_from_hidden(
    hidden_last_pos: torch.Tensor,
    ln_out,
    phi_head,
    log_probs: bool = False,
) -> torch.Tensor:
    """
    Convert the last-position hidden state of one layer into a probability over the
    vocabulary using the same LN_out and head as the final layer.
    """
    h = ln_out(hidden_last_pos)
    logits = phi_head(h)
    if log_probs:
        return F.log_softmax(logits, dim=-1)
    return F.softmax(logits, dim=-1)


def contrast_with_apc(
    qN_or_logqN: torch.Tensor,
    qJ_or_logqJ: torch.Tensor,
    alpha: float = 0.1,
    inputs_are_log: bool = False,
    temperature: Optional[float] = None,
) -> torch.Tensor:
    """
    Apply APC head filtering and DoLa log-ratio fusion.
    """
    eps = 1e-12

    if inputs_are_log:
        log_qN = qN_or_logqN
        log_qJ = qJ_or_logqJ
        qN = log_qN.exp()
    else:
        qN = qN_or_logqN.clamp_min(eps)
        qJ = qJ_or_logqJ.clamp_min(eps)
        log_qN = qN.log()
        log_qJ = qJ.log()

    max_qN = qN.max(dim=-1, keepdim=True).values
    head_mask = qN >= (alpha * max_qN)

    F_logits = log_qN - log_qJ
    neg_inf = torch.finfo(F_logits.dtype).min
    F_logits = torch.where(head_mask, F_logits, torch.full_like(F_logits, neg_inf))

    if temperature is not None and temperature > 0 and not math.isclose(temperature, 1.0):
        F_logits = F_logits / temperature

    return F.softmax(F_logits, dim=-1)


@torch.inference_mode()
def generate_with_dola_static_imgtokens(
    mmgpt,
    prompt_ids: torch.LongTensor,
    parallel_size: int,
    cfg_weight: float,
    image_token_num_per_image: int,
    ln_out,
    phi_head,
    j_star: int = 24,
    alpha: float = 0.1,
    temperature: float = 1.0,
    pad_id: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Minimal DoLa-static loop for image-token generation."""
    device = prompt_ids.device
    input_ids = prompt_ids

    tokens = torch.zeros((parallel_size * 2, input_ids.numel()), dtype=torch.long, device=device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            if pad_id is None:
                tokens[i, 1:-1] = 0
            else:
                tokens[i, 1:-1] = pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.long, device=device)

    pkv = None
    last_hidden = None
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=pkv,
            output_hidden_states=True,
        )
        pkv = outputs.past_key_values
        last_hidden = outputs.last_hidden_state

        final_last = last_hidden[:, -1, :]
        logits_all = phi_head(final_last)

        hs = outputs.hidden_states
        if not (1 <= j_star < len(hs)):
            raise ValueError(f"j_star={j_star} is out of valid range 1..{len(hs)-1}")
        j_last = hs[j_star][:, -1, :]
        logits_j_all = phi_head(ln_out(j_last))

        logit_cond = logits_all[0::2, :]
        logit_uncond = logits_all[1::2, :]
        mixed_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

        j_logit_cond = logits_j_all[0::2, :]
        j_logit_uncond = logits_j_all[1::2, :]
        mixed_logits_j = j_logit_uncond + cfg_weight * (j_logit_cond - j_logit_uncond)

        log_qN = F.log_softmax(mixed_logits, dim=-1)
        log_qJ = F.log_softmax(mixed_logits_j, dim=-1)
        probs_hat = contrast_with_apc(log_qN, log_qJ, alpha=alpha, inputs_are_log=True, temperature=temperature)

        # Use provided RNG generator for reproducibility if given
        next_token = torch.multinomial(probs_hat, num_samples=1, generator=generator)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token_2x = torch.cat([next_token, next_token], dim=1).reshape(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_2x)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    return generated_tokens, last_hidden

import torch


def compute_logprobs_from_hidden(h_lastpos: torch.Tensor, ln_out, phi) -> torch.Tensor:
    """
    Given last-position hidden state(s) h_lastpos [B, D], apply the same
    output-layer norm and head to obtain log-probabilities over the vocab.

    h_lastpos: [B, D]
    ln_out: a callable nn.Module, e.g., LLaMA final RMSNorm
    phi: the output head, e.g., lm_head or gen_head
    Returns log_probs: [B, V]
    """
    x = ln_out(h_lastpos)
    logits = phi(x)
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs


@torch.no_grad()
def jsd_from_logprobs(log_p: torch.Tensor, log_q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Jensen–Shannon divergence between two distributions given as log-probs.
    log_p, log_q: [B, V]
    Return: [B] per-sample JSD
    """
    # Convert to probs in a stable way
    p = torch.exp(log_p)
    q = torch.exp(log_q)
    m = 0.5 * (p + q)
    # Avoid zeros
    m = torch.clamp(m, min=eps)
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)

    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd


@torch.no_grad()
def select_layer_by_jsd(log_pN: torch.Tensor, cand_logprobs: dict) -> int:
    """
    Select the layer id (key of cand_logprobs) with maximum JSD to log_pN.
    cand_logprobs: {layer_idx(int): log_probs [B,V]}
    Returns the chosen layer_idx (int).
    """
    best_layer = None
    best_jsd = None
    for j, log_pj in cand_logprobs.items():
        jsd = jsd_from_logprobs(log_pN, log_pj)  # [B]
        # aggregate per-batch (mean) to choose a single layer
        score = jsd.mean()
        if (best_jsd is None) or (score > best_jsd):
            best_jsd = score
            best_layer = j
    return best_layer


@torch.no_grad()
def contrast_with_apc_from_logprobs(
    log_pN: torch.Tensor,
    log_pM: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    DoLa contrastive fusion under APC mask.

    log_pN, log_pM: [B, V]
    Return p_hat: [B, V] (probabilities, masked and normalized over head set)
    """
    # Head set by threshold on q_N
    pN = torch.exp(log_pN)
    max_pN = pN.max(dim=-1, keepdim=True).values
    head_mask = pN >= (alpha * max_pN)

    # F = log pN - log pM, but only on head set
    F = (log_pN - log_pM)
    # mask out non-head tokens with -inf so softmax ignores them
    neg_inf = torch.finfo(F.dtype).min
    F_masked = torch.where(head_mask, F, torch.full_like(F, neg_inf))
    p_hat = torch.softmax(F_masked, dim=-1)
    return p_hat


def sanitize_candidate_layers(candidate_layers, n_layers: int):
    """
    Ensure candidate layers are valid indices for hidden_states tuple.
    For HF LLaMA, hidden_states[0] is embeddings, 1..n_layers are blocks, and -1 is final.
    We restrict to 1..n_layers-1.
    """
    valid = []
    for j in candidate_layers:
        if isinstance(j, int) and 1 <= j <= max(1, n_layers - 1):
            valid.append(j)
    # deduplicate and sort
    return sorted(set(valid))


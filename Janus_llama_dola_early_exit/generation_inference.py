# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image
from typing import List, Optional

try:
    import dola_runtime
except Exception:
    dola_runtime = None  # will raise at call time

# specify the path to the model
model_path = "/z_data/migration/syxin/janus/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

conversation = [
    {
        "role": "User",
        "content": "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.",
    },
    {"role": "Assistant", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt = sft_format + vl_chat_processor.image_start_tag


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None,
        )
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)


    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_samples', exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)


generate(
    vl_gpt,
    vl_chat_processor,
    prompt,
)


@torch.inference_mode()
def generate_with_dola(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1.0,
    parallel_size: int = 16,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    alpha: float = 0.1,
    candidate_layers: Optional[List[int]] = None,
):
    """
    Janus text-to-image generation with DoLa Early Exit over the LM hidden states.
    Only adds logic on top of the original generation; model weights/arch unchanged.
    """
    if dola_runtime is None:
        raise ImportError("dola_runtime not found; expected dola_runtime.py at repo root")

    # prepare prefix tokens (cond/uncond pairs as in CFG)
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros(
        (parallel_size, image_token_num_per_image), dtype=torch.int
    ).cuda()

    # get LN_out and head (phi)
    ln_out = mmgpt.language_model.model.norm
    phi = mmgpt.gen_head

    # layer count (HF: num_hidden_layers)
    n_layers = mmgpt.language_model.config.num_hidden_layers

    # default candidate layers if not provided; prefer later bucket for image gen
    if candidate_layers is None:
        # approximate last bucket for 30L/32L models
        rough = [n_layers - 8, n_layers - 6, n_layers - 4, n_layers - 2]
        candidate_layers = [max(1, int(j)) for j in rough]

    candidate_layers = dola_runtime.sanitize_candidate_layers(candidate_layers, n_layers)

    past = None
    for step in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past,
            output_hidden_states=True,
        )
        past = outputs.past_key_values
        hs_tuple = outputs.hidden_states  # tuple length = n_layers+1 (incl. embed)

        # last position hidden for final and candidate layers
        hN = hs_tuple[-1][:, -1, :]
        logpN = dola_runtime.compute_logprobs_from_hidden(hN, ln_out, phi)

        cand_logprobs = {}
        for j in candidate_layers:
            # hidden_states index j corresponds to after j-th block
            hj = hs_tuple[j][:, -1, :]
            cand_logprobs[j] = dola_runtime.compute_logprobs_from_hidden(hj, ln_out, phi)

        # classifier-free guidance on both N and candidates
        # split into cond/uncond pairs
        def cfg_combine(logp: torch.Tensor) -> torch.Tensor:
            logits = logp  # log-probs; convert to logits space by identity up to a const
            # Work in logits: recover logits by adding a constant doesn't change softmax.
            # We convert back to probs after mixing.
            # Compute pseudo logits from logp; directly mix in exp-space for stability.
            p = torch.exp(logits)
            p_cond = p[0::2, :]
            p_uncond = p[1::2, :]
            # mix in prob space, then renormalize
            mixed = p_uncond + cfg_weight * (p_cond - p_uncond)
            mixed = torch.clamp(mixed, min=0)
            mixed = mixed / torch.clamp(mixed.sum(dim=-1, keepdim=True), min=1e-12)
            return torch.log(torch.clamp(mixed, min=1e-12))

        logpN_mix = cfg_combine(logpN)
        cand_logprobs_mix = {j: cfg_combine(lp) for j, lp in cand_logprobs.items()}

        # select layer M by max JSD
        M = dola_runtime.select_layer_by_jsd(logpN_mix, cand_logprobs_mix)
        logpM_mix = cand_logprobs_mix[M]

        # DoLa contrast with APC
        p_hat = dola_runtime.contrast_with_apc_from_logprobs(logpN_mix, logpM_mix, alpha)

        # apply temperature on p_hat to keep baseline semantics
        if temperature is not None and abs(temperature - 1.0) > 1e-6:
            p_hat = torch.softmax(torch.log(torch.clamp(p_hat, min=1e-12)) / temperature, dim=-1)

        # sample next token for each of the parallel trajectories
        next_token = torch.multinomial(p_hat, num_samples=1)
        generated_tokens[:, step] = next_token.squeeze(dim=-1)

        # feed next image token embeds (duplicate for cond/uncond)
        next_token_pairs = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_pairs)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs("generated_samples", exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join("generated_samples", f"img_dola_{i}.jpg")
        PIL.Image.fromarray(visual_img[i]).save(save_path)

    return generated_tokens

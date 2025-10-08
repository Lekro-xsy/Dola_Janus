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
import torch.nn as nn
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.dola_runtime import (
    apply_dola_on_hidden_states,
    pick_candidate_layers,
)
import numpy as np
import os
import PIL.Image

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
    outputs = None

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None,
            output_hidden_states=True,
            return_dict=True,
        )

        # Hidden for last position across cond/uncond batch
        hs_tuple = outputs.hidden_states  # len = N_layers+1, each [2*B, T, D]
        last_hidden = outputs.last_hidden_state

        # Compute final layer logits per branch
        logitsN = mmgpt.gen_head(last_hidden[:, -1, :])  # [2*B, V]
        logit_cond_N = logitsN[0::2, :]
        logit_uncond_N = logitsN[1::2, :]
        logitsN_cfg = logit_uncond_N + cfg_weight * (logit_cond_N - logit_uncond_N)

        # Build candidate layer ids from current LM depth
        n_layers = len(hs_tuple) - 1
        cand_ids = pick_candidate_layers(n_layers, k=5)

        # For DoLa we need probs/logits for both final and early layers with the same head
        # Here the head is the image-token head; no final norm is used in baseline.
        # To match CFG, we combine cond/uncond at each layer before DoLa.
        # Build a pseudo hidden_states sequence with already-CFG-combined representations
        # by projecting to logits at each candidate and mixing at the logit level.
        # We re-compute within apply_dola_on_hidden_states by passing a custom head and ln=None.

        # Construct a tiny shim: we will pass the raw hidden states and handle CFG inside below.
        # To achieve that, we create a head wrapper that first applies gen_head and then extracts
        # cond/uncond slices to mix. We define it inline for clarity.
        class _CFGHead(nn.Module):
            def __init__(self, base_head: nn.Module, w: float):
                super().__init__()
                self.base_head = base_head
                self.w = w
            def forward(self, h: torch.Tensor) -> torch.Tensor:
                logits = self.base_head(h)  # [2*B, V]
                logit_c = logits[0::2, :]
                logit_u = logits[1::2, :]
                return logit_u + self.w * (logit_c - logit_u)

        # Create a view of hidden states only at the last time-step to save memory
        # but apply_dola_on_hidden_states expects full [B,T,D] per layer; we keep as-is.
        cfg_head = _CFGHead(mmgpt.gen_head, cfg_weight).to(last_hidden.device)

        # Build a fake batch where each hidden state is merged via head; we pass ln_out=None.
        # We can't pre-apply the head to hidden_states here since DoLa will handle that.
        # So we simply call apply_dola_on_hidden_states with the raw `hs_tuple` and our cfg_head.

        # apply DoLa
        probs_joint, sel_idx, logqN, logqM = apply_dola_on_hidden_states(
            hidden_states=hs_tuple,
            head=cfg_head,
            ln_out=None,
            candidate_layer_ids=cand_ids,
            temperature=temperature,
            alpha=0.1,
            rV_log_bias=None,
            lambda_fuse=0.0,
        )

        # Sample next token from DoLa-adjusted distribution
        next_token = torch.multinomial(probs_joint, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        # Prepare next-step embeds (duplicate for cond/uncond)
        next_token_2b = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_2b)
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

import argparse
import json
import os
import re
from typing import List

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.dola_runtime import apply_dola_on_hidden_states, pick_candidate_layers


def build_prompt(vl_chat_processor: VLChatProcessor, user_text: str) -> str:
    conversation = [
        {"role": "User", "content": user_text},
        {"role": "Assistant", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation, sft_format=vl_chat_processor.sft_format, system_prompt=""
    )
    return sft_format + vl_chat_processor.image_start_tag


@torch.inference_mode()
def generate_images_with_dola(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    parallel_size: int = 4,
    temperature: float = 1.0,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    apc_alpha: float = 0.1,
    layer_k: int = 5,
):
    # Build input ids for cond/uncond CFG batch
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    outputs = None

    class _CFGHead(nn.Module):
        def __init__(self, base_head: nn.Module, w: float):
            super().__init__()
            self.base_head = base_head
            self.w = w
        def forward(self, h: torch.Tensor) -> torch.Tensor:
            logits = self.base_head(h)
            logit_c = logits[0::2, :]
            logit_u = logits[1::2, :]
            return logit_u + self.w * (logit_c - logit_u)

    cfg_head = _CFGHead(mmgpt.gen_head, cfg_weight).to(inputs_embeds.device)

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None,
            output_hidden_states=True,
            return_dict=True,
        )

        hs_tuple = outputs.hidden_states
        n_layers = len(hs_tuple) - 1
        cand_ids = pick_candidate_layers(n_layers, k=layer_k)

        probs_joint, _, _, _ = apply_dola_on_hidden_states(
            hidden_states=hs_tuple,
            head=cfg_head,
            ln_out=None,
            candidate_layer_ids=cand_ids,
            temperature=temperature,
            alpha=apc_alpha,
            rV_log_bias=None,
            lambda_fuse=0.0,
        )

        next_token = torch.multinomial(probs_joint, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token_2b = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_2b)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    # Decode image tokens
    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    return visual_img


def main():
    parser = argparse.ArgumentParser(description="Batch generate images from prompts with DoLa (language-side)")
    parser.add_argument("--prompt-file", type=str, default="/z_data/migration/syxin/janus/prompt.json")
    parser.add_argument("--out-dir", type=str, default="generated_samples_batch")
    parser.add_argument("--model", type=str, default="/z_data/migration/syxin/janus/Janus-Pro-7B")
    parser.add_argument("--parallel-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-weight", type=float, default=5.0)
    parser.add_argument("--image-token-num", type=int, default=576)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--apc-alpha", type=float, default=0.1)
    parser.add_argument("--layer-k", type=int, default=5)
    args = parser.parse_args()

    # Load prompts
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts: List[str] = data.get("prompts", [])
    assert isinstance(prompts, list) and len(prompts) > 0, "No prompts found in JSON."

    # Init model
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model)
    mmgpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    mmgpt = mmgpt.to(torch.bfloat16).cuda().eval()

    os.makedirs(args.out_dir, exist_ok=True)

    for idx, p in enumerate(prompts):
        prompt_text = build_prompt(vl_chat_processor, p)
        imgs = generate_images_with_dola(
            mmgpt=mmgpt,
            vl_chat_processor=vl_chat_processor,
            prompt=prompt_text,
            parallel_size=args.parallel_size,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
            image_token_num_per_image=args.image_token_num,
            img_size=args.img_size,
            patch_size=args.patch_size,
            apc_alpha=args.apc_alpha,
            layer_k=args.layer_k,
        )

        # Sanitize
        base = re.sub(r"\W+", "_", p)[:60]
        for j in range(imgs.shape[0]):
            save_path = os.path.join(args.out_dir, f"{idx:05d}_{base}_{j}.jpg")
            PIL.Image.fromarray(imgs[j]).save(save_path)

        print(f"[{idx+1}/{len(prompts)}] Saved {imgs.shape[0]} images for prompt: {p}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Batch image generation from a JSON list of text prompts using Janus.

Implements DoLa-static (no Early Exit) on the generative head (image-token
vocabulary) for improved robustness. No backbone/weights are modified.

Usage example:
  python batch_generate_from_prompts.py \
      --prompt_json /z_data/migration/syxin/janus/prompt.json \
      --model_path deepseek-ai/Janus-1.3B \
      --output_dir generated_samples_dola \
      --parallel_size 8 \
      --cfg_weight 5.0 \
      --alpha 0.1 --layer_ratio 0.8

Note: parallel_size images are generated per prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.dola_runtime import (
    dola_static_from_hidden,
    pick_static_layer_index,
)


def sanitize_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9._-]", "", name)
    return name[:120] if len(name) > 120 else name


@torch.inference_mode()
def generate_one(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1.0,
    parallel_size: int = 8,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    use_dola_static: bool = True,
    alpha: float = 0.1,
    layer_ratio: float = 0.8,
):
    # Build classifier-free guidance two-stream tokens (cond/uncond interleaved)
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int64, device=mmgpt.language_model.device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            # unconditional branch: pad the content between BOS/EOS
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int64, device=mmgpt.language_model.device)

    past_kv = None
    j_index = None  # decide after seeing hidden_states shape

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_kv,
            output_hidden_states=True,
        )
        past_kv = outputs.past_key_values

        if use_dola_static:
            hidden_states = outputs.hidden_states  # tuple length = n_layers (+1 if embeddings)
            if j_index is None:
                has_emb = len(hidden_states) > mmgpt.language_model.config.num_hidden_layers
                j_index = pick_static_layer_index(len(hidden_states), has_embedding=has_emb, ratio=layer_ratio)

            # DoLa-static logits for last position, for both branches stacked
            F_logits = dola_static_from_hidden(hidden_states, mmgpt.gen_head, j_index=j_index, alpha=alpha)  # [B, V]

            # split cond/uncond
            logit_cond = F_logits[0::2, :]
            logit_uncond = F_logits[1::2, :]
        else:
            # Baseline: use final layer + gen_head
            hidden_states = outputs.last_hidden_state
            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

        # CFG combining in logit space
        logits_cfg = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits_cfg / max(temperature, 1e-6), dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        # Prepare next-step inputs: image embeddings for both streams
        next_token_pair = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).reshape(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_pair)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    # Decode VQ tokens to images
    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int64), shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = dec.astype(np.uint8)  # [N, H, W, 3]
    return visual_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_json", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--model_path", type=str, default="deepseek-ai/Janus-1.3B")
    ap.add_argument("--parallel_size", type=int, default=8)
    ap.add_argument("--cfg_weight", type=float, default=5.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--image_token_num_per_image", type=int, default=576)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.1, help="APC threshold for DoLa")
    ap.add_argument("--layer_ratio", type=float, default=0.8, help="Fixed mid-layer position ratio (0,1)")
    ap.add_argument("--no_dola", action="store_true", help="Disable DoLa (baseline)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model & processor
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    mm: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)  # type: ignore
    mm = mm.to(torch.bfloat16).cuda().eval()

    # Load prompts
    with open(args.prompt_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts: List[str] = data.get("prompts", [])
    if not prompts:
        raise SystemExit("No prompts found in JSON (expected key 'prompts').")

    for idx, text in enumerate(prompts):
        conv = [
            {"role": "User", "content": text},
            {"role": "Assistant", "content": ""},
        ]
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conv, sft_format=vl_chat_processor.sft_format, system_prompt=""
        )
        prompt = sft_format + vl_chat_processor.image_start_tag

        imgs = generate_one(
            mm,
            vl_chat_processor,
            prompt,
            temperature=args.temperature,
            parallel_size=args.parallel_size,
            cfg_weight=args.cfg_weight,
            image_token_num_per_image=args.image_token_num_per_image,
            img_size=args.img_size,
            patch_size=args.patch_size,
            use_dola_static=(not args.no_dola),
            alpha=args.alpha,
            layer_ratio=args.layer_ratio,
        )

        base = sanitize_filename(text) or f"prompt_{idx:04d}"
        out_dir = os.path.join(args.output_dir, f"{idx:04d}_{base}")
        os.makedirs(out_dir, exist_ok=True)

        for k in range(imgs.shape[0]):
            save_path = os.path.join(out_dir, f"img_{k:02d}.jpg")
            Image.fromarray(imgs[k]).save(save_path, quality=95)

        print(f"[OK] {idx+1}/{len(prompts)}: saved {imgs.shape[0]} images to {out_dir}")


if __name__ == "__main__":
    main()


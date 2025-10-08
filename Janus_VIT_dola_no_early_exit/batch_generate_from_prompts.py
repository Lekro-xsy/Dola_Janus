#!/usr/bin/env python3
"""
Batch Text-to-Image generation for Janus using a JSON prompts file.

Reads a JSON file with a key "prompts": ["..."] and generates images for each
prompt using the same sampling loop as generation_inference.py/interactivechat.py.

This script loads the model once and reuses it for all prompts.
"""
import argparse
import json
import os
import re
import time
from typing import List

import numpy as np
import PIL.Image
import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor


def _sanitize(name: str, max_len: int = 80) -> str:
    name = re.sub(r"\W+", "_", name.strip())
    if len(name) > max_len:
        name = name[:max_len]
    return name or "prompt"


def _build_prompt(vl_chat_processor: VLChatProcessor, text: str) -> str:
    conversation = [
        {"role": "User", "content": text},
        {"role": "Assistant", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return sft_format + vl_chat_processor.image_start_tag


@torch.inference_mode()
def generate_images(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    parallel_size: int = 4,
    temperature: float = 1.0,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    # Build conditional/unconditional tokens for classifier-free guidance
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
    outputs = None

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

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    return visual_img  # [N, H, W, 3] uint8


def main():
    ap = argparse.ArgumentParser(description="Batch T2I from prompts JSON")
    ap.add_argument(
        "--prompts_json",
        type=str,
        required=True,
        help="Path to JSON file containing {'prompts': [...]}",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="/z_data/migration/syxin/janus/Janus-Pro-7B",
        help="HF model id or local path",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="generated_samples_batch",
        help="Output directory",
    )
    ap.add_argument("--parallel_size", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cfg_weight", type=float, default=5.0)
    ap.add_argument("--image_token_num", type=int, default=576)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--patch_size", type=int, default=16)
    args = ap.parse_args()

    # Load prompts
    with open(args.prompts_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        prompts: List[str] = data.get("prompts", [])
    assert isinstance(prompts, list) and len(prompts) > 0, "No prompts found"

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model & processor once
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model)
    mmgpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True
    )
    mmgpt = mmgpt.to(torch.bfloat16).cuda().eval()

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    for idx, text in enumerate(prompts):
        short = _sanitize(text)
        prompt = _build_prompt(vl_chat_processor, text)
        imgs = generate_images(
            mmgpt,
            vl_chat_processor,
            prompt,
            parallel_size=args.parallel_size,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
            image_token_num_per_image=args.image_token_num,
            img_size=args.img_size,
            patch_size=args.patch_size,
        )  # [N,H,W,3]

        # Save into subdir per prompt
        subdir = os.path.join(args.out_dir, f"{idx:04d}_{short}")
        os.makedirs(subdir, exist_ok=True)
        for i in range(imgs.shape[0]):
            out_path = os.path.join(subdir, f"img_{timestamp}_{i:02d}.jpg")
            PIL.Image.fromarray(imgs[i]).save(out_path)

        print(f"Saved {imgs.shape[0]} images for prompt[{idx}]: {text} -> {subdir}")


if __name__ == "__main__":
    main()

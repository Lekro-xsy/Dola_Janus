#!/usr/bin/env python3
"""
Batch generate images from prompts in a JSON file using Janus generation path,
with DoLa-static (no Early Exit) applied on the generative head.

This script mirrors the argument interface used in sibling repos' scripts,
including fixed seed and batch controls.

Example:
  python scripts/generate_batch_from_prompt_json.py \
    --json /z_data/migration/syxin/janus/prompt.json \
    --out generated_prompts \
    --model /z_data/migration/syxin/janus/Janus-Pro-7B \
    --parallel-size 8 --cfg-weight 5 --temperature 1.0 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

# Ensure repo root import when running from scripts/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.dola_runtime import (
    dola_static_from_hidden,
    pick_static_layer_index,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to prompt.json with key 'prompts'")
    ap.add_argument("--out", required=True, help="Output directory to store generated images")
    ap.add_argument("--model", default="/z_data/migration/syxin/janus/Janus-Pro-7B", help="Model id or local path")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda", help="Device, e.g., cuda or cpu")
    ap.add_argument("--parallel-size", type=int, default=8, help="Number of images sampled in parallel")
    ap.add_argument("--cfg-weight", type=float, default=5.0, help="CFG guidance weight; ignored when --disable-cfg")
    ap.add_argument("--disable-cfg", action="store_true", help="Disable classifier-free guidance to halve compute")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--image-token-num", type=int, default=576, help="Tokens per image")
    ap.add_argument("--per-prompt", type=int, default=1, help="Number of images per prompt (<= parallel-size)")
    ap.add_argument("--max-prompts", type=int, default=0, help="Limit number of prompts to process (0 means all)")
    ap.add_argument("--seed", type=int, default=42, help="Global random seed for reproducible sampling")
    ap.add_argument("--deterministic", action="store_true", help="Enable deterministic CUDA kernels (may slow down)")
    # DoLa-static hyper-params
    ap.add_argument("--alpha", type=float, default=0.1, help="APC threshold (0~1)")
    ap.add_argument("--layer-ratio", type=float, default=0.8, help="Fixed mid-layer position ratio (0,1)")
    ap.add_argument("--no-dola", action="store_true", help="Disable DoLa-static (baseline)")
    return ap.parse_args()


def build_prompt(vl_chat_processor: VLChatProcessor, user_text: str) -> str:
    conversation = [
        {"role": "User", "content": user_text},
        {"role": "Assistant", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    # For generation we append image start tag so the model begins producing image tokens
    return sft_format + vl_chat_processor.image_start_tag


@torch.inference_mode()
def generate_images(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    *,
    temperature: float = 1.0,
    parallel_size: int = 8,
    cfg_weight: float = 5.0,
    disable_cfg: bool = False,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    seed: int = 42,
    use_dola: bool = True,
    alpha: float = 0.1,
    layer_ratio: float = 0.8,
) -> np.ndarray:
    """Return array of shape [parallel_size, H, W, 3] uint8."""
    device = mmgpt.device

    # Per-call RNG for reproducibility
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    # Build CFG two-stream tokens (cond/uncond interleaved) or single stream
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    pair = 1 if disable_cfg else 2
    tokens = torch.zeros((parallel_size * pair, len(input_ids)), dtype=torch.int64, device=device)
    for i in range(parallel_size * pair):
        tokens[i, :] = input_ids
        if (not disable_cfg) and (i % 2 != 0):
            # unconditional branch: pad contents between BOS/EOS
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int64, device=device)

    past = None
    j_index = None
    for step in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past,
            output_hidden_states=True,
        )
        past = outputs.past_key_values

        if use_dola:
            hs_tuple = outputs.hidden_states
            if j_index is None:
                has_emb = len(hs_tuple) > mmgpt.language_model.config.num_hidden_layers
                j_index = pick_static_layer_index(len(hs_tuple), has_embedding=has_emb, ratio=layer_ratio)

            # Compute DoLa-static logits F for last position (for both streams stacked)
            F_logits = dola_static_from_hidden(hs_tuple, mmgpt.gen_head, j_index=j_index, alpha=alpha)

            if disable_cfg:
                logits = F_logits
            else:
                logit_cond = F_logits[0::2, :]
                logit_uncond = F_logits[1::2, :]
                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        else:
            # Baseline: use final layer + gen_head
            hidden_states = outputs.last_hidden_state
            base_logits = mmgpt.gen_head(hidden_states[:, -1, :])
            if disable_cfg:
                logits = base_logits
            else:
                logit_cond = base_logits[0::2, :]
                logit_uncond = base_logits[1::2, :]
                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=gen)
        generated_tokens[:, step] = next_token.squeeze(dim=-1)

        if disable_cfg:
            next_token_flat = next_token.view(-1)
        else:
            next_token_flat = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_flat)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int32),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).detach().cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    visual_img = dec.astype(np.uint8)
    return visual_img


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed/determinism setup
    if args.deterministic:
        try:
            import random
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
    else:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

    # Load model & processor
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(dtype=dtype).to(args.device)
    vl_gpt.eval()

    # Load prompts
    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts: List[str] = data.get("prompts", [])
    assert isinstance(prompts, list) and len(prompts) > 0, "Invalid or empty prompts"

    if args.max_prompts and args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    for idx, text in enumerate(prompts):
        prompt = build_prompt(vl_chat_processor, text)
        imgs = generate_images(
            vl_gpt,
            vl_chat_processor,
            prompt,
            temperature=args.temperature,
            parallel_size=args.parallel_size,
            cfg_weight=args.cfg_weight,
            disable_cfg=args.disable_cfg,
            image_token_num_per_image=args.image_token_num,
            img_size=args.img_size,
            patch_size=args.patch_size,
            seed=args.seed,
            use_dola=(not args.no_dola),
            alpha=args.alpha,
            layer_ratio=args.layer_ratio,
        )

        # Select the first N images per prompt (<= parallel_size)
        n_keep = min(args.per_prompt, imgs.shape[0])
        prompt_dir = out_dir / f"prompt_{idx:04d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for k in range(n_keep):
            Image.fromarray(imgs[k]).save(prompt_dir / f"img_{k:02d}.jpg")

    print(f"Saved images under: {out_dir}")


if __name__ == "__main__":
    main()


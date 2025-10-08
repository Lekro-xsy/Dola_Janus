#!/usr/bin/env python3
"""
Batch generate images from prompts in a JSON file using Janus Pro with DoLa
(language-side: Early Exit + APC + contrastive).

Parameters are aligned with the existing VIT script you referenced, including
fixed seed and per-prompt batch control.
"""

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

# Ensure repo root is importable when running from scripts/ subdir
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.dola_runtime import apply_dola_on_hidden_states, pick_candidate_layers
import torch.nn as nn


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to prompt.json with key 'prompts'")
    ap.add_argument("--out", required=True, help="Output directory to store generated images")
    ap.add_argument("--model", default="/z_data/migration/syxin/janus/Janus-Pro-7B", help="Model id or local path")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda", help="Device, e.g., cuda or cpu")
    ap.add_argument("--parallel-size", type=int, default=1, help="Number of images sampled in parallel")
    ap.add_argument("--cfg-weight", type=float, default=5.0, help="CFG guidance weight; ignored when --disable-cfg")
    ap.add_argument("--disable-cfg", action="store_true", help="Disable classifier-free guidance to halve compute")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--image-token-num", type=int, default=576, help="Tokens per image")
    ap.add_argument("--per-prompt", type=int, default=1, help="Number of images per prompt (<= parallel-size)")
    ap.add_argument("--max-prompts", type=int, default=0, help="Limit number of prompts (0 means all)")
    ap.add_argument("--seed", type=int, default=42, help="Global random seed for reproducible sampling")
    ap.add_argument("--deterministic", action="store_true", help="Deterministic kernels; may slow down")
    return ap.parse_args()


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
    *,
    temperature: float,
    parallel_size: int,
    cfg_weight: float,
    disable_cfg: bool,
    image_token_num_per_image: int,
    img_size: int,
    patch_size: int,
    seed: int,
) -> np.ndarray:
    """Return array of shape [parallel_size, H, W, 3] uint8."""
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    pair = 1 if disable_cfg else 2
    tokens = torch.zeros((parallel_size * pair, len(input_ids)), dtype=torch.int64, device=mmgpt.device)
    for i in range(parallel_size * pair):
        tokens[i, :] = input_ids
        if (not disable_cfg) and (i % 2 != 0):
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int64, device=mmgpt.device)

    # Per-call RNG for reproducibility
    gen = torch.Generator(device=mmgpt.device)
    gen.manual_seed(int(seed))

    class _CFGHead(nn.Module):
        def __init__(self, base_head: nn.Module, w: float, disabled: bool):
            super().__init__()
            self.base_head = base_head
            self.w = w
            self.disabled = disabled
        def forward(self, h: torch.Tensor) -> torch.Tensor:
            logits = self.base_head(h)
            if self.disabled:
                return logits
            logit_c = logits[0::2, :]
            logit_u = logits[1::2, :]
            return logit_u + self.w * (logit_c - logit_u)

    cfg_head = _CFGHead(mmgpt.gen_head, cfg_weight, disable_cfg).to(inputs_embeds.device)

    outputs = None
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
        cand_ids = pick_candidate_layers(n_layers, k=5)

        # DoLa (language-side): APC alpha fixed to 0.1; fusion lambda=0 for text->image
        probs_joint, _, _, _ = apply_dola_on_hidden_states(
            hidden_states=hs_tuple,
            head=cfg_head,
            ln_out=None,
            candidate_layer_ids=cand_ids,
            temperature=temperature,
            alpha=0.1,
            rV_log_bias=None,
            lambda_fuse=0.0,
        )

        next_token = torch.multinomial(probs_joint, num_samples=1, generator=gen)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        if disable_cfg:
            next_token_flat = next_token.view(-1)
        else:
            next_token_flat = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_flat)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int32),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).detach().cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    return dec.astype(np.uint8)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts: List[str] = data.get("prompts", [])
    assert isinstance(prompts, list) and len(prompts) > 0, "Invalid or empty prompts"

    # Seeds and backend flags
    try:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    # Model and processor
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    mmgpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    mmgpt = mmgpt.to(dtype=dtype).to(args.device)
    mmgpt.eval()

    if args.max_prompts and args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    for idx, text in enumerate(prompts):
        prompt = build_prompt(vl_chat_processor, text)
        imgs = generate_images_with_dola(
            mmgpt,
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
        )

        # Only keep the first N images per prompt
        n_keep = min(args.per_prompt, imgs.shape[0])
        prompt_dir = out_dir / f"prompt_{idx:04d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for k in range(n_keep):
            Image.fromarray(imgs[k]).save(prompt_dir / f"img_{k:02d}.jpg")

    print(f"Saved images under: {out_dir}")


if __name__ == "__main__":
    main()

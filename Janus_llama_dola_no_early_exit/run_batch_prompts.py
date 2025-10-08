import argparse
import json
import os
import re
from typing import List

import numpy as np
import PIL.Image
import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.dola_runtime import generate_with_dola_static_imgtokens


def sanitize_filename(text: str, maxlen: int = 80) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^A-Za-z0-9_\- ]+", "", text)
    text = text.replace(" ", "_")
    return text[:maxlen]


def make_prompt(vl_chat_processor: VLChatProcessor, text: str) -> str:
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


def main():
    ap = argparse.ArgumentParser(description="Batch-generate images with DoLa-static (no early-exit)")
    ap.add_argument("--prompt_file", default="/z_data/migration/syxin/janus/prompt.json")
    ap.add_argument("--model", default="/z_data/migration/syxin/janus/Janus-Pro-7B")
    ap.add_argument("--outdir", default="generated_samples_batch")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=576)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--j_star", type=int, default=24)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_prompts", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts: List[str] = data.get("prompts", [])
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model)
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    # RNG for reproducibility
    gen = torch.Generator(device=vl_gpt.device)
    gen.manual_seed(int(args.seed))

    for idx, text in enumerate(prompts):
        prompt = make_prompt(vl_chat_processor, text)
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).to(vl_gpt.device)

        gen_tokens, _ = generate_with_dola_static_imgtokens(
            mmgpt=vl_gpt,
            prompt_ids=input_ids,
            parallel_size=args.parallel,
            cfg_weight=args.cfg,
            image_token_num_per_image=args.tokens,
            ln_out=vl_gpt.language_model.model.norm,
            phi_head=vl_gpt.gen_head,
            j_star=args.j_star,
            alpha=args.alpha,
            temperature=args.temperature,
            pad_id=vl_chat_processor.pad_id,
            generator=gen,
        )

        dec = vl_gpt.gen_vision_model.decode_code(
            gen_tokens.to(dtype=torch.int),
            shape=[args.parallel, 8, args.img_size // args.patch, args.img_size // args.patch],
        )
        # detach to avoid requires_grad when converting to numpy
        dec = dec.to(torch.float32).detach().cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

        base_dir = os.path.join(args.outdir, f"{idx:04d}_" + sanitize_filename(text[:50]))
        os.makedirs(base_dir, exist_ok=True)
        for i in range(args.parallel):
            PIL.Image.fromarray(dec[i]).save(os.path.join(base_dir, f"{i:02d}.jpg"))
        with open(os.path.join(base_dir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import re
from typing import List, Optional

import numpy as np
import PIL.Image
import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor

import dola_runtime


def sanitize_filename(text: str, maxlen: int = 80) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^A-Za-z0-9_\- ]+", "", text)
    text = text.replace(" ", "_")
    return text[:maxlen]


@torch.inference_mode()
def generate_with_dola(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1.0,
    parallel_size: int = 4,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    alpha: float = 0.1,
    candidate_layers: Optional[List[int]] = None,
):
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

    ln_out = mmgpt.language_model.model.norm
    phi = mmgpt.gen_head
    n_layers = mmgpt.language_model.config.num_hidden_layers

    if candidate_layers is None:
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
        hs_tuple = outputs.hidden_states

        hN = hs_tuple[-1][:, -1, :]
        logpN = dola_runtime.compute_logprobs_from_hidden(hN, ln_out, phi)

        cand_logprobs = {}
        for j in candidate_layers:
            hj = hs_tuple[j][:, -1, :]
            cand_logprobs[j] = dola_runtime.compute_logprobs_from_hidden(hj, ln_out, phi)

        def cfg_combine(logp: torch.Tensor) -> torch.Tensor:
            p = torch.exp(logp)
            p_cond = p[0::2, :]
            p_uncond = p[1::2, :]
            mixed = p_uncond + cfg_weight * (p_cond - p_uncond)
            mixed = torch.clamp(mixed, min=0)
            mixed = mixed / torch.clamp(mixed.sum(dim=-1, keepdim=True), min=1e-12)
            return torch.log(torch.clamp(mixed, min=1e-12))

        logpN_mix = cfg_combine(logpN)
        cand_logprobs_mix = {j: cfg_combine(lp) for j, lp in cand_logprobs.items()}

        M = dola_runtime.select_layer_by_jsd(logpN_mix, cand_logprobs_mix)
        logpM_mix = cand_logprobs_mix[M]

        p_hat = dola_runtime.contrast_with_apc_from_logprobs(logpN_mix, logpM_mix, alpha)

        if temperature is not None and abs(temperature - 1.0) > 1e-6:
            p_hat = torch.softmax(torch.log(torch.clamp(p_hat, min=1e-12)) / temperature, dim=-1)

        next_token = torch.multinomial(p_hat, num_samples=1)
        generated_tokens[:, step] = next_token.squeeze(dim=-1)

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

    return visual_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/z_data/migration/syxin/janus/Janus-Pro-7B")
    ap.add_argument("--prompt_json", default="/z_data/migration/syxin/janus/prompt.json")
    ap.add_argument("--outdir", default="generated_samples_batch")
    ap.add_argument("--parallel_size", type=int, default=4)
    ap.add_argument("--cfg_weight", type=float, default=5.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--candidate_layers", type=str, default="")
    ap.add_argument("--max_prompts", type=int, default=-1)
    args = ap.parse_args()

    # load prompts
    with open(args.prompt_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts", [])
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    # parse candidate layers
    cand = None
    if args.candidate_layers:
        try:
            cand = [int(x) for x in args.candidate_layers.split(",") if x.strip()]
        except Exception:
            cand = None

    # model and processor
    model_path = args.model
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    os.makedirs(args.outdir, exist_ok=True)

    for idx, raw_prompt in enumerate(prompts):
        conversation = [
            {"role": "<|User|>", "content": raw_prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + vl_chat_processor.image_start_tag

        imgs = generate_with_dola(
            vl_gpt,
            vl_chat_processor,
            prompt,
            temperature=args.temperature,
            parallel_size=args.parallel_size,
            cfg_weight=args.cfg_weight,
            alpha=args.alpha,
            candidate_layers=cand,
        )

        base = f"{idx:04d}_" + sanitize_filename(raw_prompt[:50])
        for i in range(imgs.shape[0]):
            save_path = os.path.join(args.outdir, f"{base}_p{i}.jpg")
            PIL.Image.fromarray(imgs[i]).save(save_path)


if __name__ == "__main__":
    main()

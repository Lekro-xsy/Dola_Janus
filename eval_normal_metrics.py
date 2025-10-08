#!/usr/bin/env python3
"""
Evaluate images under ./normal against prompt.json using the same metric stack as eval_janus_metrics.py.

Inputs:
- Folder: ./normal, files like 1.jpg, 2.jpg, ..., N.jpg
- Prompts: ./prompt.json (a list of strings; index assumed 0-based)

Mapping rule:
- For image name K.jpg, we map to prompt index (K - 1). Out-of-range indices are skipped with a warning.

Outputs (written into ./normal/):
- janus_eval_scores.csv, janus_eval_scores.json

Notes:
- Reuses logic from eval_janus_metrics.py (local-only weights, caches under dataset/, etc.).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass
from typing import List, Optional

import torch

# Import helpers from the main evaluator (ensures local-only behavior and patches)
from eval_janus_metrics import (
    ensure_path,
    load_prompts,
    ImageEvalItem,
    compute_metric_scores,
    save_results,
)


def collect_normal_items(normal_dir: str, prompts: List[str]) -> List[ImageEvalItem]:
    items: List[ImageEvalItem] = []
    for name in sorted(os.listdir(normal_dir), key=lambda x: (len(x), x)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(name)[0]
        try:
            k = int(stem)
        except Exception:
            print(f"[WARN] Skip non-numeric filename: {name}")
            continue
        idx = k - 1  # map 1.jpg -> prompt[0]
        if idx < 0 or idx >= len(prompts):
            print(f"[WARN] prompt idx {idx} out of range for {name}")
            continue
        items.append(ImageEvalItem(prompt_idx=idx, prompt=prompts[idx], image_path=os.path.join(normal_dir, name)))
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--cache_dir', default=ensure_path('dataset/hf_cache'))
    p.add_argument('--prompt_json', default=ensure_path('prompt.json'))
    p.add_argument('--normal_dir', default=ensure_path('normal'))
    p.add_argument('--vqa_model', default='clip-flant5-xxl', choices=['clip-flant5-xxl', 'clip-flant5-xl', 'blip2-flan-t5-xxl'])
    p.add_argument('--clip_model', default='openai:ViT-L-14-336')
    p.add_argument('--itm_model', default='blip2-itm', choices=['blip2-itm', 'blip2-itm-vitL', 'blip2-itm-coco'])
    p.add_argument('--hps_model', default='hpsv2', choices=['hpsv2'])
    p.add_argument('--pickscore_model', default='pickscore-v1', choices=['pickscore-v1'])
    p.add_argument('--imagereward_ckpt', default=ensure_path('dataset/ImageReward/ImageReward.pt'))
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"[Info] Device: {args.device}")
    print(f"[Info] Cache dir: {args.cache_dir}")

    # Route caches to dataset cache_dir
    for k in ['HF_HOME', 'TRANSFORMERS_CACHE', 'HUGGINGFACE_HUB_CACHE', 'XDG_CACHE_HOME', 'TORCH_HOME', 'OPENCLIP_CACHE_DIR']:
        os.environ.setdefault(k, args.cache_dir)

    # Load prompts
    prompts = load_prompts(args.prompt_json)
    print(f"[Info] Loaded {len(prompts)} prompts from {args.prompt_json}")

    # Collect items from normal dir
    normal_dir = ensure_path(args.normal_dir)
    if not os.path.isdir(normal_dir):
        print(f"[ERR] normal dir not found: {normal_dir}")
        sys.exit(1)
    items = collect_normal_items(normal_dir, prompts)
    if not items:
        print(f"[WARN] No images found in {normal_dir}")
        sys.exit(0)
    print(f"[Info] Found {len(items)} images in {normal_dir}")

    # Compute metrics one by one (reuse compute_metric_scores for local-only behavior)
    try:
        print('[Step] HPSv2Score...')
        scores = compute_metric_scores(items, metric='hpsv2', model_name=args.hps_model, device=args.device, cache_dir=args.cache_dir)
        for it, s in zip(items, scores):
            it.HPSv2Score = s
    except Exception:
        print('[ERR] HPSv2Score failed:')
        traceback.print_exc()

    try:
        print(f'[Step] VQAScore ({args.vqa_model})...')
        scores = compute_metric_scores(items, metric='vqa', model_name=args.vqa_model, device=args.device, cache_dir=args.cache_dir)
        for it, s in zip(items, scores):
            it.VQAScore = s
    except Exception:
        print('[ERR] VQAScore failed:')
        traceback.print_exc()

    try:
        print(f'[Step] CLIPScore ({args.clip_model})...')
        scores = compute_metric_scores(items, metric='clip', model_name=args.clip_model, device=args.device, cache_dir=args.cache_dir)
        for it, s in zip(items, scores):
            it.CLIPScore = s
    except Exception:
        print('[ERR] CLIPScore failed:')
        traceback.print_exc()

    try:
        print(f'[Step] ITMScore ({args.itm_model})...')
        scores = compute_metric_scores(items, metric='itm', model_name=args.itm_model, device=args.device, cache_dir=args.cache_dir)
        for it, s in zip(items, scores):
            it.ITMScore = s
    except Exception:
        print('[ERR] ITMScore failed:')
        traceback.print_exc()

    try:
        print(f'[Step] PickScore ({args.pickscore_model})...')
        scores = compute_metric_scores(items, metric='pickscore', model_name=args.pickscore_model, device=args.device, cache_dir=args.cache_dir)
        for it, s in zip(items, scores):
            it.PickScore = s
    except Exception:
        print('[ERR] PickScore failed:')
        traceback.print_exc()

    try:
        ckpt = args.imagereward_ckpt if os.path.isfile(args.imagereward_ckpt) else 'ImageReward-v1.0'
        print(f"[Step] ImageReward ({os.path.basename(ckpt) if os.path.isfile(ckpt) else ckpt})...")
        scores = compute_metric_scores(items, metric='imagereward', model_name=ckpt, device=args.device, cache_dir=args.cache_dir)
        for it, s in zip(items, scores):
            it.ImageReward = s
    except Exception:
        print('[ERR] ImageReward failed:')
        traceback.print_exc()

    # Save into the same normal folder
    save_results(normal_dir, items)
    print('\n[Done] normal set processed.')


if __name__ == '__main__':
    main()


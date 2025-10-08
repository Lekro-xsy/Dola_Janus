#!/usr/bin/env python3
"""
Evaluate Janus image generations against prompts using multiple metrics.

Scores computed per image: HPSv2Score, VQAScore, CLIPScore, ITMScore, PickScore, ImageReward.

Per the task spec in dataset/task.md, images are collected from these folders:
- Janus_combined_dola_early_exit/outputs_dola (pick each subfolder's img_00.jpg)
- Janus_combined_dola_no_early_exit/generated_prompts_dola_static (pick each subfolder's img_00.jpg)
- Janus_llama_dola_early_exit/out_dola (pick each subfolder's img_00.jpg)
- Janus_llama_dola_no_early_exit/generated_samples_batch (pick each subfolder's 00.jpg)
- Janus_VIT_dola_early_exit/generated_from_prompts_serial (pick each subfolder's img_00.jpg)
- Janus_VIT_dola_no_early_exit/generated_samples_batch (pick each subfolder's *_00.jpg)

Outputs:
- For each of the 6 parent folders above, a CSV and JSON file summarizing per-image scores and per-metric means.

Notes:
- Uses local models under dataset when possible; downloads (HF/open_clip/pip) cache to dataset by default.
- If ImageReward import fails due to missing 'clip' package, the script will attempt to `pip install git+https://github.com/openai/CLIP.git`.
- If hpsv2 is not importable, the script will add dataset/HPSv2_github to sys.path.

Usage example (from repo root):
  python eval_janus_metrics.py --device cuda --vqa_model clip-flant5-xxl --cache_dir dataset/hf_cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import torch


# Resolve repo root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def ensure_path(p: str) -> str:
    return os.path.abspath(os.path.join(REPO_ROOT, p))


def try_import_hpsv2(hps_root: str) -> None:
    """Force `import hpsv2` to use the vendored code in dataset/HPSv2_github.
    We prepend the local path so it shadows any pip-installed package, and set HPS_ROOT
    so checkpoints are resolved locally without downloads.
    """
    local_pkg = ensure_path('dataset/HPSv2_github')
    if local_pkg not in sys.path:
        sys.path.insert(0, local_pkg)
    os.environ.setdefault('HPS_ROOT', hps_root)
    try:
        import hpsv2  # noqa: F401
    except Exception:
        print('[WARN] Failed to import local hpsv2 from dataset/HPSv2_github; falling back may cause downloads.')
        traceback.print_exc()


def ensure_imagereward_available() -> Optional[str]:
    """Ensure ImageReward can be imported. Return error string on failure, else None."""
    try:
        import ImageReward  # noqa: F401
        return None
    except ModuleNotFoundError as e:
        # Missing dependency (commonly 'clip'). Attempt to install minimal deps.
        missing = str(e)
        print(f'[INFO] ImageReward import failed ({missing}). Trying local CLIP code in dataset/CLIP ...')
        # Prefer local CLIP module provided under dataset/CLIP to avoid external installs
        local_clip_path = ensure_path('dataset/CLIP')
        if os.path.isdir(local_clip_path):
            if local_clip_path not in sys.path:
                sys.path.insert(0, local_clip_path)
            try:
                import clip  # noqa: F401
                print('[INFO] Found local clip module in dataset/CLIP')
            except Exception:
                print('[WARN] Local dataset/CLIP present but import failed; will fallback to pip install if needed.')
        # Try importing again after adding local clip
        try:
            import ImageReward  # noqa: F401
            return None
        except Exception:
            pass
        # As a last resort, install minimal packages (code deps, not weights)
        print('[INFO] Falling back to installing image-reward and openai/CLIP (code only).')
        py = sys.executable
        cmds = [
            [py, '-m', 'pip', 'install', '--quiet', 'image-reward'],
            [py, '-m', 'pip', 'install', '--quiet', 'git+https://github.com/openai/CLIP.git'],
        ]
        for cmd in cmds:
            try:
                import subprocess
                subprocess.check_call(cmd)
            except Exception:
                pass
        try:
            import ImageReward  # noqa: F401
            return None
        except Exception as e2:
            return f'Import failed after install attempts: {e2}'
    except Exception as e:
        return f'Unexpected error importing ImageReward: {e}'


def load_prompts(prompt_json_path: str) -> List[str]:
    with open(prompt_json_path, 'r') as f:
        data = json.load(f)
    prompts = data.get('prompts') or data
    assert isinstance(prompts, list) and all(isinstance(x, str) for x in prompts)
    return prompts


def find_images_for_set(root: str) -> List[Tuple[int, str]]:
    """Find (prompt_idx, image_path) pairs for a given parent dir.
    Supports the six Janus output structures in the task spec.
    """
    pairs: List[Tuple[int, str]] = []

    # Patterns per known sub-structure
    candidates: List[Tuple[str, str, str]] = [
        # (glob_root, glob_pattern, mode)
        (os.path.join(root, 'outputs_dola'), 'prompt_*/img_00.jpg', 'prompt_dir'),
        (os.path.join(root, 'generated_prompts_dola_static'), 'prompt_*/img_00.jpg', 'prompt_dir'),
        (os.path.join(root, 'out_dola'), 'prompt_*/img_00.jpg', 'prompt_dir'),
        (os.path.join(root, 'generated_from_prompts_serial'), 'prompt_*/img_00.jpg', 'prompt_dir'),
        # llama/VIT no early-exit batch structures
        (os.path.join(root, 'generated_samples_batch'), '*/00.jpg', 'batch_dir_simple'),
        (os.path.join(root, 'generated_samples_batch'), '*/*_00.jpg', 'batch_dir_timestamp'),
    ]

    import glob
    for base, pattern, mode in candidates:
        if not os.path.isdir(base):
            continue
        for img in sorted(glob.glob(os.path.join(base, pattern))):
            try:
                if mode == 'prompt_dir':
                    # .../prompt_XXXX/img_00.jpg
                    dname = os.path.basename(os.path.dirname(img))
                    assert dname.startswith('prompt_')
                    idx = int(dname.split('_')[1])
                elif mode == 'batch_dir_simple':
                    # .../XXXX_.../00.jpg
                    dname = os.path.basename(os.path.dirname(img))
                    idx = int(dname.split('_', 1)[0])
                elif mode == 'batch_dir_timestamp':
                    # .../XXXX_.../img_YYYYMMDD-HHMMSS_00.jpg
                    dname = os.path.basename(os.path.dirname(img))
                    idx = int(dname.split('_', 1)[0])
                else:
                    continue
                pairs.append((idx, img))
            except Exception:
                # Skip malformed folder/file names
                continue

    # Remove duplicates keeping first occurrence
    seen = set()
    unique_pairs: List[Tuple[int, str]] = []
    for idx, path in pairs:
        key = (idx, os.path.basename(path))
        if key in seen:
            continue
        seen.add(key)
        unique_pairs.append((idx, path))

    # Sort by prompt index
    unique_pairs.sort(key=lambda x: x[0])
    return unique_pairs


@dataclass
class ImageEvalItem:
    prompt_idx: int
    prompt: str
    image_path: str
    HPSv2Score: Optional[float] = None
    VQAScore: Optional[float] = None
    CLIPScore: Optional[float] = None
    ITMScore: Optional[float] = None
    PickScore: Optional[float] = None
    ImageReward: Optional[float] = None


def batch_dataset(items: List[ImageEvalItem]) -> List[dict]:
    # t2v_metrics expects each sample dict to have 'images' and 'texts' lists
    return [
        {
            'images': [it.image_path],
            'texts': [it.prompt],
        }
        for it in items
    ]


def compute_metric_scores(
    items: List[ImageEvalItem],
    metric: str,
    model_name: str,
    device: str,
    cache_dir: str,
    question_tmpl: Optional[str] = None,
    answer_tmpl: Optional[str] = None,
) -> List[float]:
    """Compute scores for a given metric using t2v_metrics wrappers when possible.
    metric in {'vqa', 'clip', 'itm', 'hpsv2', 'pickscore', 'imagereward'}
    """
    if metric == 'imagereward':
        # Use ImageReward package directly
        err = ensure_imagereward_available()
        if err:
            print(f'[WARN] Skipping ImageReward due to import error: {err}')
            return [float('nan')] * len(items)
        import ImageReward as reward
        # The model_name here is a checkpoint path or canonical name "ImageReward-v1.0"
        med_cfg = ensure_path('dataset/ImageReward/med_config.json')
        model = reward.load(name=model_name, device=device, download_root=cache_dir, med_config=med_cfg if os.path.isfile(med_cfg) else None)
        scores: List[float] = []
        with torch.no_grad():
            for it in items:
                try:
                    s = float(model.score(it.prompt, it.image_path))
                except Exception:
                    s = float('nan')
                scores.append(s)
        return scores

    # t2v_metrics for the rest
    # Import t2v_metrics lazily to ensure ffmpeg check runs once
    sys.path.insert(0, ensure_path('dataset/t2v_metrics'))
    import t2v_metrics  # noqa: E402

    if metric == 'vqa':
        # If using CLIP-FlanT5, point to local checkpoints under dataset/
        if model_name in ['clip-flant5-xxl', 'clip-flant5-xl']:
            try:
                from t2v_metrics.models.vqascore_models import clip_t5_model as _ct5
                local_dir = 'dataset/clip-flant5-xxl' if model_name.endswith('xxl') else 'dataset/clip-flant5-xl'
                local_dir = ensure_path(local_dir)
                if os.path.isdir(local_dir):
                    _ct5.CLIP_T5_MODELS[model_name]['model']['path'] = local_dir
                    _ct5.CLIP_T5_MODELS[model_name]['tokenizer']['path'] = local_dir
                    print(f'[INFO] Using local VQA checkpoint at {local_dir}')
            except Exception:
                print('[WARN] Failed to remap VQA model to local path; will rely on HF cache in dataset.')
        score_model = t2v_metrics.get_score_model(
            model=model_name,
            device=device,
            cache_dir=cache_dir,
        )
        kwargs = {}
        if question_tmpl:
            kwargs['question_template'] = question_tmpl
        if answer_tmpl:
            kwargs['answer_template'] = answer_tmpl
        scores_tensor = score_model.batch_forward(batch_dataset(items), batch_size=8, **kwargs)
    elif metric == 'clip':
        # Prefer local HF CLIP ViT-L/14-336 if available, by patching CLIPScoreModel to use transformers
        local_hf_clip_l14_336 = ensure_path('dataset/clip-vit-large-patch14-336')
        hf_clip_patched = False
        if os.path.isdir(local_hf_clip_l14_336) and model_name.lower().startswith('openai:vit-l-14-336'):
            try:
                from t2v_metrics.models.clipscore_models import clip_model as _clip
                from transformers import CLIPProcessor, CLIPModel
                def _hf_load_model(self):
                    self.processor = CLIPProcessor.from_pretrained(local_hf_clip_l14_336)
                    self.model = CLIPModel.from_pretrained(local_hf_clip_l14_336).to(self.device).eval()
                def _hf_load_images(self, image: List[str]):
                    imgs = [self.image_loader(x) for x in image]
                    inputs = self.processor(images=imgs, return_tensors='pt')
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    return inputs
                @torch.no_grad()
                def _hf_forward(self, images: List[str], texts: List[str]) -> torch.Tensor:
                    assert len(images) == len(texts)
                    img_inputs = self.load_images(images)
                    text_inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors='pt')
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                    image_features = self.model.get_image_features(**img_inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = self.model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    return (image_features * text_features).sum(dim=-1)
                _clip.CLIPScoreModel.load_model = _hf_load_model
                _clip.CLIPScoreModel.load_images = _hf_load_images
                _clip.CLIPScoreModel.forward = _hf_forward
                hf_clip_patched = True
                print(f'[INFO] Using local HF CLIP at {local_hf_clip_l14_336} for CLIPScore')
            except Exception:
                print('[WARN] Failed to patch CLIPScore to use local HF CLIP; will fallback to open_clip.')
        # Ensure open_clip code is available when using open_clip pretrained ids (only if not patched to HF)
        if (':' in model_name) and not hf_clip_patched:
            try:
                import open_clip  # noqa: F401
            except Exception:
                print('[INFO] Installing open_clip_torch (code only) for CLIPScore...')
                import subprocess
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'open_clip_torch'])
        score_model = t2v_metrics.get_score_model(
            model=model_name,
            device=device,
            cache_dir=cache_dir,
        )
        scores_tensor = score_model.batch_forward(batch_dataset(items), batch_size=64)
    elif metric == 'itm':
        # Prefer local HF Transformers BLIP2-ITM weights under dataset/ to avoid any downloads
        # by monkey-patching the BLIP2ITMScoreModel to use AutoProcessor/Blip2ForImageTextRetrieval.
        try:
            local_itm_dir = ensure_path('dataset/blip2-itm-vit-g')
            if os.path.isdir(local_itm_dir):
                from t2v_metrics.models.itmscore_models import blip2_itm_model as _blip
                from transformers import AutoProcessor
                try:
                    from transformers import Blip2ForImageTextRetrieval as _Blip2ITM
                except Exception:
                    _Blip2ITM = None
                if _Blip2ITM is not None:
                    def _hf_itm_load_model(self):
                        self.processor = AutoProcessor.from_pretrained(local_itm_dir)
                        self.model = _Blip2ITM.from_pretrained(local_itm_dir).to(self.device).eval()
                    def _hf_itm_forward(self, images: List[str], texts: List[str]) -> torch.Tensor:
                        from PIL import Image
                        assert len(images) == len(texts), 'Number of images and texts must match'
                        pil_imgs = [self.image_loader(x) for x in images]
                        inputs = self.processor(text=texts, images=pil_imgs, return_tensors='pt', padding=True)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        with torch.no_grad():
                            out = self.model(**inputs)
                        # For retrieval, use logits_per_image diagonal as match score; squish to [0,1] via sigmoid
                        logits = out.logits_per_image
                        if logits.ndim == 2:
                            # take diagonal for aligned pairs
                            score_vec = torch.diag(logits)
                        else:
                            score_vec = logits.view(-1)
                        return torch.sigmoid(score_vec)
                    _blip.BLIP2ITMScoreModel.load_model = _hf_itm_load_model
                    _blip.BLIP2ITMScoreModel.forward = _hf_itm_forward
                    # load_images not needed; forward constructs inputs via processor
                    print(f'[INFO] Using local Transformers BLIP2-ITM at {local_itm_dir}')
        except Exception:
            print('[WARN] Failed to patch ITM to local Transformers weights; will use LAVIS default (may trigger downloads).')
        score_model = t2v_metrics.get_score_model(
            model=model_name,
            device=device,
            cache_dir=cache_dir,
        )
        scores_tensor = score_model.batch_forward(batch_dataset(items), batch_size=16)
    elif metric == 'hpsv2':
        # Ensure hpsv2 import works before t2v_metrics wrapper tries to import it
        try_import_hpsv2(hps_root=ensure_path('dataset/HPSv2'))
        score_model = t2v_metrics.get_score_model(
            model=model_name,  # 'hpsv2'
            device=device,
            cache_dir=cache_dir,
        )
        scores_tensor = score_model.batch_forward(batch_dataset(items), batch_size=8)
    elif metric == 'pickscore':
        # If local PickScore_v1 exists, patch the loader to use it directly
        try:
            local_pickscore = ensure_path('dataset/PickScore_v1')
            if os.path.isdir(local_pickscore):
                from t2v_metrics.models.clipscore_models import pickscore_model as _pick
                from transformers import AutoProcessor, AutoModel
                def _patched_load_model(self):
                    # Load both processor and model from the local PickScore_v1 folder
                    self.processor = AutoProcessor.from_pretrained(local_pickscore)
                    self.model = AutoModel.from_pretrained(local_pickscore).eval().to(self.device)
                _pick.PickScoreModel.load_model = _patched_load_model
                print(f'[INFO] Using local PickScore checkpoint at {local_pickscore}')
        except Exception:
            print('[WARN] Failed to patch PickScore to local path; will rely on HF cache in dataset.')
        score_model = t2v_metrics.get_score_model(
            model=model_name,  # 'pickscore-v1'
            device=device,
            cache_dir=cache_dir,
        )
        scores_tensor = score_model.batch_forward(batch_dataset(items), batch_size=32)
    else:
        raise ValueError(f'Unknown metric: {metric}')

    # scores_tensor shape: [N, 1, 1]; reduce to list[float]
    scores = scores_tensor.detach().float().cpu().view(len(items), -1).numpy().squeeze(axis=1)
    return [float(x) for x in scores]


def save_results(out_dir: str, items: List[ImageEvalItem]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Compute per-metric means
    def mean(values: List[Optional[float]]) -> Optional[float]:
        xs = [x for x in values if isinstance(x, (int, float)) and not (x is None or (isinstance(x, float) and (x != x)))]
        if not xs:
            return None
        return float(sum(xs) / len(xs))

    metrics = ['HPSv2Score', 'VQAScore', 'CLIPScore', 'ITMScore', 'PickScore', 'ImageReward']
    metric_means: Dict[str, Optional[float]] = {m: mean([getattr(it, m) for it in items]) for m in metrics}

    # CSV
    csv_path = os.path.join(out_dir, 'janus_eval_scores.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        header = ['prompt_idx', 'prompt', 'image_path'] + metrics
        f.write(','.join(header) + '\n')
        for it in items:
            row = [
                str(it.prompt_idx),
                json.dumps(it.prompt),  # quote commas safely
                json.dumps(os.path.relpath(it.image_path, out_dir)),
            ] + [
                '' if getattr(it, m) is None else f'{getattr(it, m):.6f}' for m in metrics
            ]
            f.write(','.join(row) + '\n')
        # averages row
        avg_row = ['AVERAGE', '', ''] + [
            '' if metric_means[m] is None else f'{metric_means[m]:.6f}' for m in metrics
        ]
        f.write(','.join(avg_row) + '\n')

    # JSON
    json_path = os.path.join(out_dir, 'janus_eval_scores.json')
    out = {
        'per_image': [asdict(it) for it in items],
        'averages': metric_means,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f'[OK] Wrote results: {csv_path}\n[OK] Wrote results: {json_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--cache_dir', default=ensure_path('dataset/hf_cache'), help='HF/open_clip cache dir (kept under dataset by default)')
    p.add_argument('--prompt_json', default=ensure_path('prompt.json'))
    # Model selections (VQA has two options as noted by user)
    p.add_argument('--vqa_model', default='clip-flant5-xxl', choices=['clip-flant5-xxl', 'clip-flant5-xl', 'blip2-flan-t5-xxl'], help='VQAScore model')
    p.add_argument('--clip_model', default='openai:ViT-L-14-336', help='CLIPScore model from open_clip; e.g., openai:ViT-L-14-336')
    p.add_argument('--itm_model', default='blip2-itm', choices=['blip2-itm', 'blip2-itm-vitL', 'blip2-itm-coco'])
    p.add_argument('--hps_model', default='hpsv2', choices=['hpsv2'])
    p.add_argument('--pickscore_model', default='pickscore-v1', choices=['pickscore-v1'])
    p.add_argument('--imagereward_ckpt', default=ensure_path('dataset/ImageReward/ImageReward.pt'), help='Path to ImageReward.pt or use "ImageReward-v1.0"')
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # Informative prints
    print(f'[Info] Device: {args.device}')
    print(f'[Info] Cache dir: {args.cache_dir}')

    # Route common caches to dataset cache_dir to keep downloads inside dataset/
    for k in ['HF_HOME', 'TRANSFORMERS_CACHE', 'HUGGINGFACE_HUB_CACHE', 'XDG_CACHE_HOME', 'TORCH_HOME', 'OPENCLIP_CACHE_DIR']:
        os.environ.setdefault(k, args.cache_dir)

    # Load prompts
    prompts = load_prompts(args.prompt_json)
    print(f'[Info] Loaded {len(prompts)} prompts from {args.prompt_json}')

    # Parent folders to evaluate (absolute paths)
    parent_dirs = [
        ensure_path('Janus_combined_dola_early_exit'),
        ensure_path('Janus_combined_dola_no_early_exit'),
        ensure_path('Janus_llama_dola_early_exit'),
        ensure_path('Janus_llama_dola_no_early_exit'),
        ensure_path('Janus_VIT_dola_early_exit'),
        ensure_path('Janus_VIT_dola_no_early_exit'),
    ]

    # Score all sets
    for parent in parent_dirs:
        if not os.path.isdir(parent):
            print(f'[WARN] Skip missing dir: {parent}')
            continue
        print(f'\n=== Evaluating set: {parent} ===')
        pairs = find_images_for_set(parent)
        if not pairs:
            print(f'[WARN] No images found under {parent}, skipping')
            continue

        # Build eval items; align prompt index to prompt list
        items: List[ImageEvalItem] = []
        for idx, img_path in pairs:
            if idx < 0 or idx >= len(prompts):
                print(f'[WARN] prompt idx {idx} out of range for {img_path}')
                continue
            items.append(ImageEvalItem(prompt_idx=idx, prompt=prompts[idx], image_path=ensure_path(img_path)))

        # Compute metrics one-by-one to limit peak memory
        # HPSv2Score
        try:
            print('[Step] HPSv2Score...')
            scores = compute_metric_scores(items, metric='hpsv2', model_name=args.hps_model, device=args.device, cache_dir=args.cache_dir)
            for it, s in zip(items, scores):
                it.HPSv2Score = s
        except Exception:
            print('[ERR] HPSv2Score failed:')
            traceback.print_exc()

        # VQAScore
        try:
            print(f'[Step] VQAScore ({args.vqa_model})...')
            scores = compute_metric_scores(items, metric='vqa', model_name=args.vqa_model, device=args.device, cache_dir=args.cache_dir)
            for it, s in zip(items, scores):
                it.VQAScore = s
        except Exception:
            print('[ERR] VQAScore failed:')
            traceback.print_exc()

        # CLIPScore
        try:
            print(f'[Step] CLIPScore ({args.clip_model})...')
            scores = compute_metric_scores(items, metric='clip', model_name=args.clip_model, device=args.device, cache_dir=args.cache_dir)
            for it, s in zip(items, scores):
                it.CLIPScore = s
        except Exception:
            print('[ERR] CLIPScore failed:')
            traceback.print_exc()

        # ITMScore
        try:
            print(f'[Step] ITMScore ({args.itm_model})...')
            scores = compute_metric_scores(items, metric='itm', model_name=args.itm_model, device=args.device, cache_dir=args.cache_dir)
            for it, s in zip(items, scores):
                it.ITMScore = s
        except Exception:
            print('[ERR] ITMScore failed:')
            traceback.print_exc()

        # PickScore
        try:
            print(f'[Step] PickScore ({args.pickscore_model})...')
            scores = compute_metric_scores(items, metric='pickscore', model_name=args.pickscore_model, device=args.device, cache_dir=args.cache_dir)
            for it, s in zip(items, scores):
                it.PickScore = s
        except Exception:
            print('[ERR] PickScore failed:')
            traceback.print_exc()

        # ImageReward
        try:
            ckpt = args.imagereward_ckpt if os.path.isfile(args.imagereward_ckpt) else 'ImageReward-v1.0'
            print(f'[Step] ImageReward ({os.path.basename(ckpt) if os.path.isfile(ckpt) else ckpt})...')
            scores = compute_metric_scores(items, metric='imagereward', model_name=ckpt, device=args.device, cache_dir=args.cache_dir)
            for it, s in zip(items, scores):
                it.ImageReward = s
        except Exception:
            print('[ERR] ImageReward failed:')
            traceback.print_exc()

        # Save results next to the set root
        save_results(parent, items)

    print('\n[Done] All sets processed.')


if __name__ == '__main__':
    main()

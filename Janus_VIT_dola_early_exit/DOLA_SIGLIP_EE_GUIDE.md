# Janus SigLIP DoLa (Early‑Exit) — Changes & How To Run

- Scope: Add DoLa (Decoding by Contrasting Layers) as post‑processing for SigLIP ViT-L/16 per modify1.md, without changing ViT classes or trained weights. Also add a batch prompt runner for `prompt.json`.
- Repo: `/z_data/migration/syxin/janus/Janus_VIT_dola_early_exit`

## What I Changed
- Added pure DoLa runtime utilities (no model edits):
  - `janus/utils/dola_runtime.py`
    - `build_probs_from_rep(z, text_bank, tau)` → q(c) via softmax of dot products.
    - `jsd(p, q)` → Jensen–Shannon divergence.
    - `select_early_exit(qN, {qj})` → per-sample argmax JSD layer index.
    - `contrast_and_mask(qN, qM, alpha)` → APC mask + log‑ratio softmax to get p̂.
    - `dola_from_distributions(qN, q_list, alpha)` → convenience wrapper returning `(p̂, m_idx)`.
  - `janus/utils/rerank_with_dola.py`
    - `rerank_with_dola_siglip(vit, images, text_bank, alpha=0.1, tau=1.0, candidate_layers=(18,20,22))`.
    - Uses `VisionTransformer.get_intermediate_layers(..., norm=True, return_prefix_tokens=True)` and `forward_head(..., pre_logits=True)` to pool each candidate layer and the final layer, build q’s, select Early‑Exit via JSD, then fuse per DoLa.
    - Does not modify `janus/models/siglip_vit.py`.
- Added batch prompt generation script:
  - `scripts/generate_batch_from_prompt_json.py`
    - Loads Janus model once and generates images for each prompt from a JSON file containing `{ "prompts": [ ... ] }`.
    - Saves outputs under a target directory, one subfolder per prompt.

No existing files were modified; only new files were added, keeping the main model intact as required.

## DoLa Usage (SigLIP ViT‑L/16)
- Inputs:
  - `vit`: an instantiated SigLIP `VisionTransformer` (e.g., from `CLIPVisionTower`).
  - `images`: float tensor `[B,3,H,W]` normalized as the tower expects.
  - `text_bank`: tensor `[C,D]` of candidate text embeddings produced by the paired text encoder (same embedding space).
- Defaults align with modify1.md recommendations:
  - Bucket [17,24] with even layers `(18,20,22)`; `alpha=0.1`; `tau=1.0`.

Example snippet:

```python
import torch
from janus.models.siglip_vit import create_siglip_vit
from janus.utils.rerank_with_dola import rerank_with_dola_siglip

# Vision model (SigLIP ViT-L/16, 24 layers)
vit = create_siglip_vit(model_name="siglip_large_patch16_384", image_size=384)
vit.eval().cuda()

images = torch.randn(2, 3, 384, 384, device="cuda")  # preprocessed as needed
text_bank = torch.randn(10, vit.embed_dim, device="cuda")  # supply real text embds

p_hat, m_idx = rerank_with_dola_siglip(
    vit, images, text_bank, alpha=0.1, tau=1.0, candidate_layers=(18, 20, 22)
)
# p_hat: [B,C] final DoLa probabilities; m_idx: [B] chosen layer index (0..len-1)
```

Notes:
- `text_bank` must be computed by the SigLIP text encoder (same space/dim) for meaningful results.
- You can change `candidate_layers` to any 1‑based subset, e.g., per a validated bucket.

## Batch Run Prompts (Text→Image Generation)
- JSON source: `/z_data/migration/syxin/janus/prompt.json` with key `prompts`.
- Output: images will be saved under the specified directory: one folder per prompt.

Command line:

```bash
cd /z_data/migration/syxin/janus/Janus_VIT_dola_early_exit

# Example: generate 1 image per prompt with local Janus-Pro-7B
python scripts/generate_batch_from_prompt_json.py \
  --json /z_data/migration/syxin/janus/prompt.json \
  --out ./generated_from_prompts \
  --model /z_data/migration/syxin/janus/Janus-Pro-7B \
  --parallel-size 16 --per-prompt 1 --cfg-weight 5 --temperature 1.0
```

- Flags:
  - `--parallel-size`: total parallel chains (CFG pairs). Keep 16 if memory allows.
  - `--per-prompt`: number of images to save per prompt (<= `parallel-size`).
  - `--model`: HF hub id or local checkpoint path.
  - `--dtype`: `bfloat16` (default), `float16`, or `float32`.
  - `--device`: `cuda` or `cpu`.

Outputs will be under `generated_from_prompts/prompt_XXXX/img_YY.jpg`.

## Verify
- Sanity import test for DoLa:

```bash
python - << 'PY'
from janus.utils.dola_runtime import build_probs_from_rep, jsd
print('DoLa utils import OK')
PY
```

- Dry run for the batch script (just checks JSON and imports):

```bash
python - << 'PY'
import json
from pathlib import Path
from scripts.generate_batch_from_prompt_json.py import build_prompt
print(len(json.load(open('/z_data/migration/syxin/janus/prompt.json'))['prompts']))
print('batch script import OK')
PY
```

## Notes and Assumptions
- We did not edit ViT or model state/weights; DoLa is implemented as external utilities.
- For DoLa classification/reranking, ensure `text_bank` comes from the paired SigLIP text encoder and matches the vision head embedding dimension.
- Batch generation uses the existing Janus generation path; network access may be needed to fetch the HF model if not cached.

If you want me to wire `use_dola_siglip=True` as a toggle in a specific inference entry point you use in this repo, tell me which file/function you call for scoring, and I will add the minimal switch-in without touching model weights.

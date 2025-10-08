**Janus SigLIP DoLa-Static + Batch Prompts**
- Location: /z_data/migration/syxin/janus/Janus_VIT_dola_no_early_exit

- Summary: Implemented DoLa-static utilities for SigLIP ViT as a pure add-on (no weight/arch changes) per modify2.md, and added a batch text-to-image runner to process prompts from a JSON file.

What I Changed
- Added DoLa runtime utilities (no changes to existing classes):
  - janus/utils/dola_runtime.py
    - build_probs_from_rep(z, text_bank, tau): softmax over cosine scores <z, t_c>/tau.
    - contrast_and_mask(qN, qJ, alpha): APC head filtering (alpha) + log-ratio fusion restricted to head set; returns fused distribution.
    - rerank_with_dola_static_siglip(vit, image_tensor, text_bank, fixed_layer, alpha, tau): wires SigLIP ViT final layer and a fixed intermediate layer j* through forward_head to compute DoLa-static fused probs over text candidates.

- Added batch prompts runner for text-to-image generation:
  - batch_generate_from_prompts.py (executable)
    - Loads Janus (default /z_data/migration/syxin/janus/Janus-Pro-7B) once, reads a JSON with key "prompts", and generates images per prompt.
    - Saves outputs under an output directory, grouping per prompt.

Notes On DoLa-Static (SigLIP)
- Follows modify2.md exactly: only adds logic, uses VisionTransformer.get_intermediate_layers(..., norm=True) and forward_head(pre_logits=True) for z_j* and z_N; APC head with alpha=0.1 by default; final distribution p_hat uses softmax over the log-ratio on the head set.
- This repo does not include a SigLIP text encoder. Supply your own text embedding bank (C x D) when calling rerank_with_dola_static_siglip (e.g., from a SigLIP text tower you already use). The function leaves your existing towers/weights untouched.

How To Use DoLa-Static For SigLIP (Code Snippet)
```
import torch
from janus.models.siglip_vit import create_siglip_vit
from janus.utils.dola_runtime import rerank_with_dola_static_siglip

# 1) Build/load SigLIP ViT (vision tower). Use your ckpt if available.
vit = create_siglip_vit(
    model_name="siglip_large_patch16_384", image_size=384, ckpt_path=""
).cuda().eval()

# 2) Prepare a batch of images [B, 3, H, W] and a text embedding bank [C, D].
images = torch.randn(2, 3, 384, 384).cuda()  # replace with real images
text_bank = torch.randn(10, vit.embed_dim).cuda()  # replace with your SigLIP text encodings

# 3) DoLa-static rerank: choose j* (e.g., 18 for 24-layer model), alpha and tau.
p_hat, qN, qJ = rerank_with_dola_static_siglip(
    vit, images, text_bank, fixed_layer=18, alpha=0.1, tau=1.0
)

# p_hat is your final class distribution over the C candidates.
```

Batch Run Prompts (Text-to-Image)
- Input: /z_data/migration/syxin/janus/prompt.json (expects a top-level key "prompts": list of strings)
- Output: images are saved under an output directory with a subdirectory per prompt.

Run
- Single command (defaults shown):
```
cd /z_data/migration/syxin/janus/Janus_VIT_dola_no_early_exit
python3 batch_generate_from_prompts.py \
  --prompts_json /z_data/migration/syxin/janus/prompt.json \
  --model /z_data/migration/syxin/janus/Janus-Pro-7B \
  --out_dir generated_samples_batch \
  --parallel_size 4 \
  --temperature 1.0 \
  --cfg_weight 5.0 \
  --image_token_num 576 \
  --img_size 384 \
  --patch_size 16
```

Outputs
- Images are saved to generated_samples_batch/<index>_<sanitized-prompt>/img_<timestamp>_<i>.jpg.

Assumptions & Tips
- For DoLa-static with SigLIP, fix j* via small validation once (e.g., 18 or 20 on 24-layer model), then reuse online. If you have no validation set, 18/20 are reasonable starting points.
- tau: keep consistent with your baseline normalization. The helpers default to 1.0 but you can pass any temperature.
- alpha: 0.1 by default; adjust in [0.05, 0.2] if needed.

Rollback/Consistency Check
- If alpha -> 0 and qJ == qN (or you skip contrast), p_hat == qN, matching the baseline distribution.

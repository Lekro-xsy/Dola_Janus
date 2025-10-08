# Janus LLaMA DoLa (Early Exit) Integration

This implements modify3.md (方案③: LLaMA + DoLa + Early Exit) in this repo with minimal, additive changes. No model weights or architecture are changed.

What I changed
- Added `dola_runtime.py` (new file)
  - `compute_logprobs_from_hidden(h, ln_out, phi)`: LN_out -> head -> log_softmax
  - `jsd_from_logprobs(P, Q)`: Jensen–Shannon divergence
  - `select_layer_by_jsd(qN, {qj})`: choose candidate layer with max JSD
  - `contrast_with_apc_from_logprobs(qN, qM, alpha)`: APC + log-ratio + normalize to `p_hat`
  - `sanitize_candidate_layers(...)`: keep valid layer indices w.r.t. HF hidden_states layout
- Updated `generation_inference.py`
  - Kept original `generate(...)` unchanged
  - Added `generate_with_dola(...)` that:
    - runs the base LLaMA with `output_hidden_states=True, use_cache=True`
    - computes `q_N` (final) and `{q_j}` (candidate layers) using the same `LN_out` and the same output head (`gen_head` here)
    - applies CFG in the same place as baseline (on distributions) before DoLa
    - selects `M` by max JSD, then DoLa APC+contrast to get `p_hat`
    - samples next token from `p_hat` (temperature preserved)
    - returns images saved to `generated_samples/`
- Added `run_batch_prompts.py` (new file)
  - Batch runner that reads `/z_data/migration/syxin/janus/prompt.json` and generates images per prompt using the DoLa path
  - Saves results to an output directory with informative filenames

Notes on correctness w.r.t. modify3.md
- Uses the model final RMSNorm (`mmgpt.language_model.model.norm`) as `LN_out` for all layers
- Uses the exact same head `phi` as baseline (`mmgpt.gen_head`) for both final and intermediate layers
- Keeps KV-cache and block structure intact; only reads last-position hidden states
- JSD and APC follow the paper’s equations; computations use stable log-softmax forms
- DoLa only replaces the distribution source; temperature and CFG semantics remain unchanged

How to run (batch prompts)
1) Install deps (if not already):
   - `pip install -e .`
2) Run batch generation with DoLa:
   - Local model path (默认已指向本地权重，无需传参也可）：
     - `python run_batch_prompts.py --prompt_json /z_data/migration/syxin/janus/prompt.json --outdir out_dola --parallel_size 4 --alpha 0.1 --candidate_layers 22,24,26,28`

Arguments
- `--model`: HF repo id or local path
- `--prompt_json`: path to JSON with `{\"prompts\": [...]}`
- `--outdir`: output directory for images
- `--parallel_size`: images per prompt (each step uses CFG pairs internally)
- `--cfg_weight`: classifier-free guidance weight (default 5.0)
- `--temperature`: sampling temperature on `p_hat` (default 1.0)
- `--alpha`: APC threshold (default 0.1)
- `--candidate_layers`: comma-separated hidden layer indices (HF hidden_states indexing; use later layers e.g. `22,24,26,28` for 30L; for 32L models, `24,26,28,30`)
- `--max_prompts`: limit number of prompts processed (debugging)

Outputs
- Images are saved as `outdir/{idx}_{prompt_stub}_p{k}.jpg`
- Each prompt yields `parallel_size` images

Tips
- If your model has 32 layers (e.g., some 7B variants), suggested candidates: `24,26,28,30`
- If you want to disable DoLa and compare baseline, continue to use your original code path (we did not remove or change it)
- VRAM usage grows with `parallel_size`; reduce it if OOM

Programmatic use
- See `generation_inference.py:generate_with_dola` to call DoLa generation directly from Python

Verification
- With `--alpha 0.0` and forcing `M` to the final layer (not exposed via CLI, but you can modify the code accordingly), the outputs should match baseline sampling step by step

File references
- dola_runtime.py
- generation_inference.py
- run_batch_prompts.py

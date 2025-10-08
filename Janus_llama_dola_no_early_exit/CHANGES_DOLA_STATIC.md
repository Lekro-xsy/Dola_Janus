# Janus LLaMA DoLa-static (No Early Exit)

本次改动严格按 /z_data/migration/syxin/janus/modify4.md 的方案④实现：
- 仅新增最小逻辑，不改底层模型架构或权重；
- 以固定层 j* 与最终层做对数比（DoLa）并配合 APC 头部筛选，替换采样分布；
- 保持解码后续流程一致（温度/CFG/top-k 等不变，按基线顺序进行）。

## 变更内容
- 新增: `janus/utils/dola_runtime.py`
  - `compute_probs_from_hidden(h, ln_out, phi) -> q`：对单层最后位置隐状态施加相同 LN_out 与 head 得到分布（支持 log_probs）。
  - `contrast_with_apc(qN, qJ, alpha, ...) -> p_hat`：实现 APC 头部过滤 + 对数比分布融合；支持在 DoLa 之后再施加 temperature。
  - `generate_with_dola_static_imgtokens(...)`：最小替换版图生 token 生成循环，按固定 j* 执行 DoLa-static，并与 CFG 同步在两层上混合后再对比。
- 新增: `run_batch_prompts.py`
  - 批量读取 `/z_data/migration/syxin/janus/prompt.json` 的 `prompts`，使用 DoLa‑static 生成图像，逐条保存及拼图保存。
- 更新: `generation_inference.py`
  - 新增 `generate_dola_static(...)` 演示接口（不修改原有 `generate`）。
- 更新: `interactivechat.py`
  - 新增 `generate_dola_static(...)`，并将交互默认改为 DoLa-static（固定 j*）。

涉及文件（相对路径）：
- Janus_llama_dola_no_early_exit/janus/utils/dola_runtime.py
- Janus_llama_dola_no_early_exit/run_batch_prompts.py
- Janus_llama_dola_no_early_exit/scripts/generate_batch_from_prompt_json.py
- Janus_llama_dola_no_early_exit/generation_inference.py
- Janus_llama_dola_no_early_exit/interactivechat.py

## 使用说明
- 环境：使用本地权重 `/z_data/migration/syxin/janus/Janus-Pro-7B`（已在脚本中设为默认）。
- 推荐默认超参：`j_star=24`，`alpha=0.1`，`temperature=1.0`，`cfg=5.0`。

### 一键运行（請整行複製，不要在參數中間回車）
- GPU03（最穩定做法：顯式指定 PYTHONPATH 並使用絕對路徑腳本）：
  `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/z_data/migration/syxin/janus/Janus_llama_dola_no_early_exit python /z_data/migration/syxin/janus/Janus_llama_dola_no_early_exit/scripts/generate_batch_from_prompt_json.py --prompt_file /z_data/migration/syxin/janus/prompt.json --model /z_data/migration/syxin/janus/Janus-Pro-7B --outdir /z_data/migration/syxin/janus/Janus_llama_dola_no_early_exit/generated_samples_batch --parallel 4 --tokens 576 --img_size 384 --patch 16 --temperature 1.0 --cfg 5.0 --j_star 24 --alpha 0.1 --seed 42`
- 如需更換 GPU：把 `CUDA_VISIBLE_DEVICES=3` 改為你的 GPU 編號（如 0/1/2）。
- 若希望在倉庫目錄下以模組方式調用（無需手動設置 PYTHONPATH）：
  - `cd /z_data/migration/syxin/janus/Janus_llama_dola_no_early_exit`
  - `CUDA_VISIBLE_DEVICES=3 python -m scripts.generate_batch_from_prompt_json --prompt_file /z_data/migration/syxin/janus/prompt.json --model /z_data/migration/syxin/janus/Janus-Pro-7B --outdir generated_samples_batch --parallel 4 --tokens 576 --img_size 384 --patch 16 --temperature 1.0 --cfg 5.0 --j_star 24 --alpha 0.1 --seed 42`
- 路徑自檢（可選）：
  - `ls -l /z_data/migration/syxin/janus/prompt.json`
  - `ls -ld /z_data/migration/syxin/janus/Janus-Pro-7B`
- 顯存不足：將 `--parallel` 降到 1 或 2；必要時也可減小 `--tokens`。

1) 单条 DoLa‑static 演示（生成若干并行图像）
- 进入目录：`cd Janus_llama_dola_no_early_exit`
- 运行：`python generation_inference.py`
  - 该脚本现在默认走 DoLa‑static 版本（在 `__main__` 中调用 `generate_dola_static`）。
  - 生成图片输出到 `generated_samples/`，文件名包含 `img_dola_static_*.jpg`。

2) 交互式 DoLa‑static 生成（已默认）
- 进入目录：`cd Janus_llama_dola_no_early_exit`
- 运行：`python interactivechat.py`
  - 交互默认走 DoLa‑static（固定层 j*）。

  3) 批量运行 `/z_data/migration/syxin/janus/prompt.json` 的 prompts
- 进入目录：`cd Janus_llama_dola_no_early_exit`
- 运行（默认路径已指向该文件）：
  - `python run_batch_prompts.py \
      --prompt_file /z_data/migration/syxin/janus/prompt.json \
      --model /z_data/migration/syxin/janus/Janus-Pro-7B \
      --outdir generated_samples_batch \
      --parallel 4 \
      --tokens 576 \
      --img_size 384 \
      --patch 16 \
      --temperature 1.0 \
      --cfg 5.0 \
      --j_star 24 \
       --alpha 0.1`
   - 或等价脚本（参数对齐，包含固定 seed 与 batch 大小）：
     最穩定（絕對路徑 + 顯式 PYTHONPATH）：
     `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/z_data/migration/syxin/janus/Janus_llama_dola_no_early_exit python /z_data/migration/syxin/janus/Janus_llama_dola_no_early_exit/scripts/generate_batch_from_prompt_json.py --prompt_file /z_data/migration/syxin/janus/prompt.json --model /z_data/migration/syxin/janus/Janus-Pro-7B --outdir /z_data/migration/syxin/janus/Janus_llama_dola_no_early_exit/generated_samples_batch --parallel 4 --tokens 576 --img_size 384 --patch 16 --temperature 1.0 --cfg 5.0 --j_star 24 --alpha 0.1 --seed 42`

> 注意
> - 請“整行複製”，不要把路徑或參數分成多行；如果一定要換行，務必在每一行末尾加 `\`，且不要在參數的引號內換行。
> - 如使用 `python /某目錄/` 這種形式會報 `__main__` 找不到，必須指向具體的 `.../xxx.py` 文件或使用 `python -m package.module` 形式。
- 输出：
  - 每个 prompt 一个子目录（含多张图和 `prompt.txt`），以及汇总网格图 `*_grid.jpg`。

## 说明与约束对应
- LN_out 与读头共享：对固定层 j* 使用 `mmgpt.language_model.model.norm` 与 `mmgpt.gen_head`，与最终层保持一致。
- 稳定性：所有对数计算使用 `log_softmax`，并在 APC 中使用概率阈值（内部做最小截断）。
- CFG 一致性：对最终层与 j* 层分别进行 cond/uncond 混合，再执行对数比，保证流程公平可比。
- 后续不变：DoLa 得到的 `F` 仅作为新 logits，再按温度等策略采样。
- 动态早退未启用：`j_star` 为常数（默认 24），可在脚本参数调整或线下网格搜索确定。

## 可选参数与调优
- `--j_star`：建议先在 {22, 24, 26} 中试验；如有小验证集可做网格搜索。
- `--alpha`：APC 阈值，默认 0.1，可在 [0.05, 0.2] 之间微调。
- `--temperature`、`--cfg` 等：保持与基线一致的默认即可作公平对比。

## 回退与验证
- 将 `--alpha` 设为极小（近 0）且令 j* 接近最终层时，DoLa 趋近于基线分布，输出应与基线一致。
- 如需切换到非 DoLa 版本，继续使用原始 `generate`/`interactivechat` 的 `generate` 即可。

# Janus Combined DoLa-Static (No Early Exit)

本次按 modify6（双塔均不带 Early Exit 的 DoLa-static）思路，在不改任一骨干/权重的前提下，新增运行时工具与批量生图脚本，支持用 DoLa 对比重写生成头的分布，并批量运行 `prompt.json` 的提示词。

## 改动内容
- 新增工具：`janus/utils/dola_runtime.py`
  - `contrast_with_apc_from_logits(final_logits, mid_logits, alpha)`：对数域对比 + APC 头部筛选，返回可直接作为 logits 使用的对比分数 `F`（屏蔽头部外词为大负值）。
  - `dola_static_from_hidden(hidden_states, head_module, j_index, alpha)`：对固定中间层 `j*` 与最终层进行 DoLa-static，对最后位置输出构造对比分布（适用于 `gen_head` 或 `lm_head`）。
  - `pick_static_layer_index(num_hidden_states, has_embedding, ratio)`：按比例选择固定层索引（默认靠后 0.8 位置）。
- 新增批量生图脚本：`batch_generate_from_prompts.py`
  - 载入 Janus 模型与处理器，按 `generation_inference.py` 的 CFG 两路（cond/uncond）策略生成图像。
  - 在每步生成时，开启 `output_hidden_states=True`，使用 `gen_head` 对最终层与固定中间层读出 logits，按 DoLa-static + APC 获得对比分布 `F`，再做 CFG 融合与采样。
  - 从 `/z_data/migration/syxin/janus/prompt.json` 中读取 prompts，逐条生成并保存到输出目录下的子文件夹。

> 说明：未修改任何已有类与权重；所有 DoLa 逻辑为“运行时后处理”。

## 运行方法
1) 准备依赖（一次）：
   - 可选：`pip install -r requirements.txt`
   - 模型路径可用本地目录（如 `/z_data/syxin/janus/Janus-Pro-7B`）或 HuggingFace 名称（如 `deepseek-ai/Janus-1.3B`）。

2) 批量生成命令示例：
```
cd /z_data/migration/syxin/janus/Janus_combined_dola_no_early_exit
python batch_generate_from_prompts.py \
  --prompt_json /z_data/migration/syxin/janus/prompt.json \
  --output_dir ./generated_samples_dola_static \
  --model_path /z_data/syxin/janus/Janus-Pro-7B \
  --parallel_size 8 \
  --cfg_weight 5.0 \
  --alpha 0.1 \
  --layer_ratio 0.8
```
- `--parallel_size`：每个 prompt 生成的图片数量（CFG 两路内部自动处理）。
- `--cfg_weight`：classifier-free guidance 强度（与基线一致语义）。
- `--alpha`：APC 阈值，默认 0.1。
- `--layer_ratio`：固定对比层位置（0~1 之间，默认 0.8，越靠后越接近最终层）。
- 若想对比基线，可加 `--no_dola` 关闭 DoLa。

3) 输出结果：
- 生成图片保存在 `--output_dir` 下，以 `序号_截断后的提示词/` 为子目录；每个目录内有若干 `img_XX.jpg`。

## 实现要点（对齐 DoLa-static）
- 语言塔（此处用于图像 token 生成头 `gen_head`）：
  - 取 `hidden_states[-1][:, -1, :]`（最终层、最后位置）与 `hidden_states[j*][:, -1, :]`（固定中间层）。
  - 同一 `gen_head` 投影为 `final_logits` 与 `mid_logits`。
  - `F = log_softmax(final) - log_softmax(mid)`，在 `q_N` 头部集合上保留，其余置大负值，作为新的 logits 进入 CFG 与采样。
- 视觉端（静态 DoLa）共用同一组工具函数，后续若需在 SigLIP 检索/分类上启用，可直接用 `contrast_with_apc_from_logits` 组合相似度打分的 logits（本次未触动原视觉编码路径）。

## 参数建议
- `alpha=0.1`（APC）起步，0.05~0.2 可微调。
- 固定层：`layer_ratio=0.8`；若需手设，可把比率换算到靠后中间层附近。
- 其余采样相关（温度、CFG、长度）均与基线一致。

## 回退与对比
- 运行时加 `--no_dola` 即回到基线。
- 或将 `alpha` 设为极小并把固定层选到最终层附近，理论上也接近基线行为。

---

如需把 DoLa-static 应用于图像-文本问答（`inference.py` 路径）或加入跨模态融合（例如对数域可加的视觉先验偏置），可在当前 `dola_runtime.py` 基础上极少改动即插入：只需在每步生成时将语言端 logits 替换为 DoLa 对比分布，再（可选）叠加来自视觉端的对数偏置。若需要我继续接入该部分，请告知具体期望。


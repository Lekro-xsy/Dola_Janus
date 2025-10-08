# Janus DoLa (方案⑤) 集成说明

本次在不改动任何骨干结构/权重的前提下，新增了 DoLa（层对比 + Early Exit + APC）能力，并优先在“语言端（生成图像 token）”生效；视觉端工具函数也已补齐，后续可在图文任务中启用跨模态融合（方案 B）。

变更与新增
- 新增工具
  - `janus/utils/dola_runtime.py`：DoLa 核心，包括 `jsd`、`select_early_exit`、`contrast_and_mask`、`apply_dola_on_hidden_states`、`pick_candidate_layers` 等。
  - `janus/utils/mm_fusion.py`：跨模态融合工具（方案 B），支持候选短语到词表起始 token 的映射与 `r_V(x)` 计算。
- 生成侧改动（文本→图像）
  - 保留原始 `generation_inference.py` 的接口，内部将每步 logits 改写为 DoLa 分布：
    - 从 LLaMA 的各层 hidden_states 计算“最终层 vs 候选中间层”的分布；
    - 以 JSD 最大层作为 Early Exit，做对数域对比 + APC；
    - 对 CFG（classifier-free guidance）保持一致：对 cond/uncond 分支分别取各层 logits 后再按 `logit_u + w*(logit_c-logit_u)` 合成，再做 DoLa。
- 批量脚本
  - 新增 `batch_generate_prompts.py`：读取 JSON prompts，批量按 DoLa 策略出图，默认并行多图。

与方案⑤的一致性
- 语言端（LLaMA）：严格执行 q_N 与 q_j 的 JSD 选层、APC 掩码与 `log q_N - log q_M` 的对比分布，最终在掩码上 `softmax` 归一化；温度/CFG 等保持与基线一致。
- 视觉端（SigLIP）：提供 `build_probs_from_rep` 与候选层选取 `pick_candidate_layers` 等工具；若后续提供候选 `text_bank` 与映射 `T(c)`，可直接通过 `mm_fusion.visual_prior_log_bias` 生成 `r_V(x)` 并在语言端按 `F^joint = F^{(L)} + λ r_V` 注入。当前批量出图流程不需要视觉先验，默认 λ=0。

如何运行（批量 prompts 出图）
1) 准备环境（首次）
- 建议使用已有 requirements：`pip install -r requirements.txt`
- 确保可访问模型权重：`/z_data/migration/syxin/janus/Janus-Pro-7B`

2) 批量生成
- 命令示例：
  - 最简（按给定 JSON 批量出图，默认每个 prompt 生成 4 张）：
    ```bash
    python batch_generate_prompts.py \
      --prompt-file /z_data/migration/syxin/janus/prompt.json \
      --out-dir outputs_dola \
      --parallel-size 4
    ```

  - 可调超参：
    - `--temperature 1.0` 采样温度
    - `--cfg-weight 5.0` CFG 权重
    - `--image-token-num 576` 每张图片 token 数
    - `--apc-alpha 0.1` APC 阈值（语言端）
    - `--layer-k 5` 候选中间层数（语言端 Early Exit）

3) 输出
- 生成图片保存在 `--out-dir`，命名包含 prompt 片段与索引，例如 `00012_A_cat_with_three_eyes_0.jpg`。

可选：图文任务（图→文）启用 DoLa 与跨模态融合
- 文本生成（VQA/描述）可参照 `janus/utils/dola_runtime.py` 的 `apply_dola_on_hidden_states`：
  - 调用 `language_model.model(..., output_hidden_states=True)` 拿到所有层的 hidden states；
  - 使用 `head=language_model.lm_head`、`ln_out=language_model.model.norm`；
  - 选层/对比/掩码同上，得到 `hat p_L` 替换原分布；
  - 如有视觉候选 `c` 与映射 `T(c)`，用 `mm_fusion.visual_prior_log_bias` 得到 `r_V(x)`，以 `F^joint = F^{(L)} + λ r_V` 融合。

参数建议（默认）
- APC：`alpha_L = 0.1`
- 候选层数：`layer_k = 3~5`（均匀取、偏偶数层）
- 融合强度：`λ = 0.7`（若启用视觉先验）

回退与验证
- 将 `--apc-alpha 0` 且将候选层固定为最终层即可回到基线分布。
- 将 `--layer-k 1` 或 `λ=0` 禁用跨模态影响。
- 若需要对比，保持“重复惩罚→DoLa→（可选）融合”的固定次序。

文件改动一览
- 新增：`janus/utils/dola_runtime.py`
- 新增：`janus/utils/mm_fusion.py`
- 修改：`generation_inference.py`（在不改接口前提下引入 DoLa）
- 新增：`batch_generate_prompts.py`（批量出图脚本）

如需我把 DoLa 融合接到图→文 `inference.py` 并支持候选词表映射，请提供/确认候选短语集合与映射策略（或允许我内置一版默认映射规则）。

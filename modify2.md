这是**方案②：SigLIP ViT-L/16 使用 DoLa（不带 Early Exit / 固定对比层，DoLa-static）**的落地说明。依然坚持**只新增逻辑、零改架构/权重**，并且**全部沿用 DoLa 原论文公式**（只是把“词表”解释为“类别/文本候选集”）。

---

# 方案②：SigLIP ViT-L/16＋DoLa-static（不带 Early Exit）

## 0. 目标与约束
- **目标**：在 SigLIP ViT-L/16 的推理阶段采用 **DoLa 的层对比与对数比融合**，但**不做动态选层**；即在**固定中间层 \(j^\*\)** 与最终层 \(N\) 之间做对比，得到最终类别/检索分布，用于**零样本分类/检索重排序**。
- **不改架构/权重**：不改 ViT 主干与 head，不改文本塔；只在**后处理**阶段新增“读中间层→读头→对比→重打分”的逻辑。
- **仓库对齐（你当前项目）**  
  - ViT：`janus/models/siglip_vit.py` 已提供 `get_intermediate_layers(..., norm=True)` 与 `forward_head(...)`；  
  - VLM 装配：`janus/models/modeling_vlm.py`（SigLIP 作为视觉塔）；  
  - 我们只在外部新增调用与工具函数，不改已有类。

---

## 1. DoLa 的**标准公式**（静态对比版本）

> 记候选/类别集合为 \(\mathcal{C}\)（零样本分类可由一组文本提示产生）；“LM 词表分布”映射为“类别/文本候选分布”。

**(1) 成熟分布（最终层）**  
令 \(z_N\) 为 ViT **最终层**经 head 路径得到的图像表示；\(\{t_c\}_{c\in\mathcal{C}}\) 为文本端候选表示（SigLIP 文本编码器输出）；\(\tau\) 为温度：
\[
s_N(c)=\frac{\langle z_N,\, t_c\rangle}{\tau},\qquad 
q_N(c)=\frac{\exp\big(s_N(c)\big)}{\sum_{u\in\mathcal{C}}\exp\big(s_N(u)\big)}.
\]

**(2) 固定早退层分布（中间层 \(j^\*\)）**  
从 ViT 的固定中间层 \(j^\*\)（\(1\le j^\*<N\)）取得图像表示 \(z_{j^\*}\)（与 \(z_N\) 相同的归一化与 head 路径），计算：
\[
s_{j^\*}(c)=\frac{\langle z_{j^\*},\, t_c\rangle}{\tau},\qquad 
q_{j^\*}(c)=\mathrm{softmax}\big(s_{j^\*}(c)\big).
\]

**(3) 自适应头部约束（APC）**  
用成熟分布 \(q_N\) 的**头部筛选**：
\[
\mathcal{V}_{\text{head}}=\Big\{c\in\mathcal{C}:\; q_N(c)\ge \alpha\cdot \max_{u\in\mathcal{C}} q_N(u)\Big\},
\]
常用 \(\alpha=0.1\)。

**(4) 对比融合（DoLa 核心）**  
仅在 \(\mathcal{V}_{\text{head}}\) 上用**对数比**构造对比分数：
\[
F(c)=\log q_N(c)-\log q_{j^\*}(c)=\log\frac{q_N(c)}{q_{j^\*}(c)},\qquad c\in\mathcal{V}_{\text{head}},
\]
将 \(\mathcal{V}_{\text{head}}\) 外的类视为 \(-\infty\)（屏蔽），并对 \(F\) 做 `softmax` 得最终分布：
\[
\hat p(c)=\frac{\exp\big(F(c)\big)}{\sum_{u\in\mathcal{V}_{\text{head}}}\exp\big(F(u)\big)}.
\]
推理时用 \(\hat p\) 替换原本由 \(q_N\) 直接给出的分布/分数。

> 与方案①相比，这里只把“动态选层 \(M\)”**替换为固定层 \(j^\*\)**；其余公式完全一致。

---

## 2. 固定层 \(j^\*\) 的选择（离线确定，一次设定）
- **全局固定**（推荐）：在你的目标任务/域上，用一个**小验证集**对若干候选层做**网格试验**（如 \(\{8,10,12,14,16,18,20,22\}\)），选择验证指标最优的 \(j^\*\)；线上推理时固定使用该层。  
- **分桶先验**（加速挑选）：24 层可分 **3 个 bucket**：\([1,8],[9,16],[17,24]\)。先测试每个 bucket 的中后段（例如 6/7/8、14/15/16、22/23/24 中挑 2–3 个层），再细化。  
- **经验起点**：若无验证集，可先从**靠后的中间层**尝试（如 18 或 20），通常较稳；之后再精调。

> 注意：DoLa-static 的最优层对数据分布**更敏感**，因此推荐使用一次性的小验证来定层；但一旦定层，线上推理**无需再做层选择**，开销更低。

---

## 3. 你需要“新增”的东西（不改已有类/参数）

### 3.1 通用 DoLa 工具（与方案①共用，少用两个函数即可）
**位置建议**：`janus/utils/dola_runtime.py`  
**本方案仅需**：
1) `build_probs_from_rep(z, text_bank, tau)`：输入图像表示 \(z\) 与文本库 \(\{t_c\}\)，返回分布 \(q(\cdot)\)。  
2) `contrast_and_mask(qN, qJ, alpha)`：实现 APC（\(\alpha\)）＋对数比 \(F=\log\frac{q_N}{q_J}\)＋`softmax` 得 \(\hat p\)。

> 与方案①相比，本方案不需要 `jsd(...)` 与 `select_early_exit(...)`。

### 3.2 SigLIP 推理侧的接线（不改 ViT 类）
- **取最终层表示**：按你现有 pipeline，`forward_head(..., pre_logits=False)` 得 \(z_N\)（在当前实现中，`head` 为恒等或线性，沿用基线路径）。  
- **取固定中间层表示**：用 `get_intermediate_layers(x, indices=[j*], norm=True)` 得到 \(z_{j^\*}\)（保持与最终层相同的 `norm/head` 方式）。  
- **文本候选嵌入**：沿用你现有 SigLIP 文本塔处理，得到 \(\{t_c\}\)。  
- **DoLa-static 重打分**：  
  - 计算 \(q_N(c)\) 与 \(q_{j^\*}(c)\)；  
  - 用 APC 得 \(\mathcal{V}_{\text{head}}\)（\(\alpha=0.1\)）；  
  - 用 \(F(c)=\log q_N(c)-\log q_{j^\*}(c)\) 得 \(\hat p(c)\)，作为最终分布/分数。  
- **落点建议**：新增一个函数（例如）  
  - `rerank_with_dola_static_siglip(image_tensor, text_bank, fixed_layer=j*, alpha=0.1, tau=...)`  
  将其在你当前的 VLM 推理/评测入口（`janus/models/modeling_vlm.py` 的推理分支）通过一个开关 `use_dola_siglip_static=True` 接入。  
- **保证**：不动 `VisionTransformer` 与现有权重，仅调用其公开方法读取中间层并重打分。

---

## 4. 推理时序（一步一步）
1) **标准前向**得到 \(z_N\) → 计算成熟分布 \(q_N\)。  
2) **读取固定层 \(j^\*\)** 得 \(z_{j^\*}\) → 计算 \(q_{j^\*}\)。  
3) **APC 约束**：\(\mathcal{V}_{\text{head}}=\{c: q_N(c)\ge \alpha\cdot\max q_N\}\)，\(\alpha=0.1\)。  
4) **对比融合**：\(\hat p(c)\propto \exp\!\big(\log q_N(c)-\log q_{j^\*}(c)\big)\)（仅在 \(\mathcal{V}_{\text{head}}\) 上）。  
5) **输出**：用 \(\hat p\) 做最终分类决策或检索排序。

---

## 5. 超参与开销
- **\(\alpha\)**：0.1（可在 0.05–0.2 间微调）。  
- **\(\tau\)**：与基线保持一致（沿用你当前 SigLIP 温度或相似度归一化配置）。  
- **固定层 \(j^\*\)**：由验证挑选；若无验证，先试 18/20。  
- **性能开销**：相对基线只多一次“中间层读头＋softmax＋对比”，比方案①（需要多层计算＋选层）更轻。

---

## 6. 诊断与回退
- **一致性回退**：当 \(\alpha\to 0\) 且强制 \(q_{j^\*}=q_N\)（或直接跳过对比）时，\(\hat p\equiv q_N\)——应与现有基线完全一致。  
- **敏感性检查**：在验证集上画“固定层 \(j\) vs 指标”曲线，确认最优 \(j^\*\) 的稳定性；如在不同数据子集漂移明显，可考虑切换到方案①（动态选层）。  
- **可视化**：比较 \(q_N\) 与 \(q_{j^\*}\) 的前 K 类概率差异，确认对比确实提升了判别性（错误类别在 \(q_{j^\*}\) 上常更“自信”）。

---

## 7. 你要改哪些地方（**只新增，不覆盖**）
1) **新增** `janus/utils/dola_runtime.py`（或共用方案①的同名文件）：实现  
   - `build_probs_from_rep` 与 `contrast_and_mask` 两个函数（本方案只需这两个）。  
2) **新增** `rerank_with_dola_static_siglip(...)`（文件位置随你，建议 `janus/utils/`）：  
   - 从 ViT 取 \(z_N\) 与 \(z_{j^\*}\)，计算 \(q_N\)、\(q_{j^\*}\) 并输出 \(\hat p\)。  
3) **在推理入口接一个布尔开关**（如 `use_dola_siglip_static=True`）来启用本方案；不开关则仍走你原有分布 \(q_N\)。
/z_data/migration/syxin/janus/Janus_VIT_dola_no_early_exit在这里改
---

### CODING PROTOCOL ###
开发守则：
- 严格用最少的代码完成当前任务
- 不进行大规模改动
- 不做无关编辑，专注于你正在开发的任务
- 代码必须精确、模块化、可测试
- 不破坏现有功能
- 如果需要我做任何配置（例如Supabase/AWS），请明确告诉我
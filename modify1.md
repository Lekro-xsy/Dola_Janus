# 方案①：SigLIP ViT-L/16（24 层）＋ DoLa（带 Early Exit）

## 0. 目标与约束
- **目标**：在 **SigLIP ViT-L/16** 的推理阶段引入 **DoLa（Decoding by Contrasting Layers）** 的层对比分布与动态早退层选择（Early Exit），用于**零样本分类/检索重排序**等评分环节。
- **不改架构/权重**：不改 ViT 主干、头部、投影或任何已训练参数；仅新增“取中间层→读出→对比→重打分”的**后处理逻辑**。
- **仓库对齐**（来自你上传的代码）  
  - ViT 实现：`janus/models/siglip_vit.py`（已提供 `get_intermediate_layers(..., norm=True)` 与 `forward_head(...)`）。  
  - VLM 总装：`janus/models/modeling_vlm.py`（集成视觉与语言；SigLIP 作为视觉塔）。  
  - 下面若提到“新增模块”，建议放在 `janus/utils/` 下，保持主干清爽。

---

## 1. DoLa 的**标准公式**（原式，变量名与符号对齐）

> ViT 的“词表”用**类别/文本候选集** \(\mathcal{C}\) 替代；“token 分布”即“类别分布”。

**(1) 成熟分布**（最终层）  
令 \(z_N\) 为 ViT **最终层**经 head 投影后的图像表示，\(\{t_c\}_{c\in\mathcal{C}}\) 为文本端候选表示（SigLIP 文本编码器输出，或你现有提示模板的文本嵌入），\(\tau\) 为温度。  
\[
s_N(c)=\frac{\langle z_N,\, t_c\rangle}{\tau},\qquad 
q_N(c)=\frac{\exp\big(s_N(c)\big)}{\sum_{u\in\mathcal{C}}\exp\big(s_N(u)\big)}.
\]

**(2) 早退分布**（中间层 \(j\)）  
取 ViT 第 \(j\) 层的中间表示 \(z_j\)（经 `norm=True` 与与最终层一致的 head 路径），同一打分方式：  
\[
s_j(c)=\frac{\langle z_j,\, t_c\rangle}{\tau},\qquad 
q_j(c)=\mathrm{softmax}\big(s_j(c)\big).
\]

**(3) 动态早退层选择（Early Exit）**  
在候选集合 \(\mathcal{J}\subset\{1,\dots,N-1\}\) 上，选出与成熟分布差异最大的层
\[
M=\arg\max_{j\in\mathcal{J}}\mathrm{JSD}\!\left(q_N\,\Vert\,q_j\right),
\]
其中 Jensen–Shannon 散度
\[
\mathrm{JSD}(P\Vert Q)=\tfrac12\,\mathrm{KL}\!\left(P\Vert \tfrac{P+Q}{2}\right)+\tfrac12\,\mathrm{KL}\!\left(Q\Vert \tfrac{P+Q}{2}\right).
\]

**(4) 对比融合（DoLa 核心）**  
先用成熟分布的**自适应头部约束（APC）**筛集合：
\[
\mathcal{V}_{\text{head}}=\Big\{c\in\mathcal{C}:\; q_N(c)\ge \alpha\cdot \max_{u\in\mathcal{C}} q_N(u)\Big\},
\]
常用 \(\alpha=0.1\)。然后对 \(c\notin\mathcal{V}_{\text{head}}\) 置 \(-\infty\)（等价于屏蔽），其余类别按**对数比**：
\[
F(c)=\log q_N(c)-\log q_M(c)=\log\frac{q_N(c)}{q_M(c)}.
\]
对 \(F\) 做 softmax 得最终分布 \(\hat p\)：
\[
\hat p(c)=\frac{\exp\big(F(c)\big)}{\sum_{u\in\mathcal{V}_{\text{head}}}\exp\big(F(u)\big)}.
\]
推理时用 \(\hat p\) 替换原本由 \(q_N\) 直接给出的打分/排序概率。

> 以上就是 DoLa 原公式在 ViT 分类/检索语义下的**一一映射**：  
> 「词表」→「候选类/文本提示」；「LM head \(\phi\)」→「SigLIP 相似度打分+softmax」。

---

## 2. 层候选与“分桶”策略（论文建议的鲁棒实践）
- **总层数** \(N=24\)。把层划分为 **3 个 bucket**：\([1,8],[9,16],[17,24]\)（含边界）。  
- 用一个**小验证集**先确定**最佳 bucket**；推理时仅在该 bucket 内抽若干候选层（建议**偶数层**或等间隔层，如 \(\{18,20,22\}\)）。  
- 这样既省算力，也提升 Early Exit 的稳定性。

---

## 3. 你仓库里的**最小新增点**（只加，不改）

### 3.1 新增一个通用的 DoLa 工具模块
- **位置**：`janus/utils/dola_runtime.py`（名称可自定）  
- **职责**（纯函数化，便于单测）：  
  1) `build_probs_from_rep(z, text_bank, tau)`：给定图像表示 \(z\) 与文本库 \(\{t_c\}\)，产出 \(q(\cdot)\)。  
  2) `jsd(qA, qB)`：计算 \(\mathrm{JSD}\)。  
  3) `select_early_exit(qN, {qj})`：在 \(\mathcal{J}\) 上取 \(M\)。  
  4) `contrast_and_mask(qN, qM, alpha)`：按 APC 与 \(\log\frac{q_N}{q_M}\) 得 \(\hat p\)。

> 这些函数**不依赖**具体模型类，只吃张量或 numpy，确保与主干解耦。

### 3.2 在 SigLIP ViT 推理路径上接线（不改 ViT 类）
- **取中间层表示**：复用 `janus/models/siglip_vit.py` 里的  
  `get_intermediate_layers(x, indices, norm=True)` 拿到 \(\{z_j\}_{j\in\mathcal{J}}\)。  
- **取最终层表示**：走你当前的 `forward_head(..., pre_logits=False)` 路径得到 \(z_N\)（与基线一致）。  
- **文本端候选**：保持你现有的 SigLIP 文本编码流程，得到 \(\{t_c\}\)。  
- **DoLa 重打分**：调用 3.1 的工具函数，计算 \(\hat p(c)\) 作为**最终类别分布/排序分数**。  
- **落点**：把这段逻辑封装为**新增函数**（建议）
  - `janus/utils/rerank_with_dola.py` 中的 `rerank_with_dola_siglip(image_tensor, text_bank, alpha=0.1, candidate_layers=..., tau=...)`  
  - 或在 `janus/models/modeling_vlm.py` 的推理/评价分支里新增一个**可选开关**（如 `use_dola=True`）以调用上面工具函数。  
- **保证**：不动 `siglip_vit.VisionTransformer` 的 `forward`/`state_dict`；仅在**外部**读取中间层并重打分。

---

## 4. 推理时序（一步到位的执行流程）
给定一张图像与文本候选 \(\mathcal{C}\)：

1) **标准前向**得到最终层表示 \(z_N\) 与成熟分布 \(q_N\)。  
2) **抽取候选层** \(\mathcal{J}\)（来自所选 bucket 的若干层），对每个 \(j\)：  
   - 取 \(z_j\) → 计算 \(q_j\)。  
3) **Early Exit 选层**：  
   \[
   M=\arg\max_{j\in\mathcal{J}}\mathrm{JSD}\!\left(q_N\,\Vert\,q_j\right).
   \]
4) **APC 约束**：  
   \[
   \mathcal{V}_{\text{head}}=\{c\!:\!q_N(c)\ge \alpha\cdot \max q_N\},\quad \alpha=0.1.
   \]
5) **对比融合**并得到最终分布：  
   \[
   \hat p(c)\propto \exp\!\Big(\log q_N(c)-\log q_M(c)\Big),\; c\in\mathcal{V}_{\text{head}}.
   \]
6) **输出**：用 \(\hat p\) 做最终的**类别决策/检索重排序**。

---

## 5. 可选与默认值（与论文习惯一致）
- **\(\alpha\)**（APC 阈值）：默认 **0.1**；任务更细粒度可降到 0.05。  
- **分桶数**：**3**（24 层）；也可试 2 或 4，以验证集选择。  
- **候选层个数**：每步用 **3–5 个**层足够（偶数层优先）。  
- **温度 \(\tau\)**：沿用你现有 SigLIP 配置。  
- **代价**：相对基线仅多几次“读头＋softmax”，时延通常是可忽略的（\(\approx 1.01\)–\(1.08\times\)）。

---

## 7. 你需要改哪些地方（**只新增，不覆盖**）
1) **新增** `janus/utils/dola_runtime.py`：实现 `build_probs_from_rep` / `jsd` / `select_early_exit` / `contrast_and_mask`。  
2) **新增** `janus/utils/rerank_with_dola.py`（或同文件内函数）：把 SigLIP 专用的取层、取头与 DoLa 串起来。  
3) **在推理入口接一个开关**（如 `use_dola_siglip=True`）：  
   - 入口一般在 `janus/models/modeling_vlm.py` 的评测/推理分支里（你用哪条路径，就在那条路径里**调用新增函数**完成重打分）。  
   - **不修改** `janus/models/siglip_vit.py` 的类定义与已有 forward，仅调用其已暴露的 `get_intermediate_layers` 与 `forward_head`。

---

> 以上即**第一种情况：SigLIP ViT-L/16 使用 DoLa 且带 Early Exit**的落地方案
# 在/z_data/migration/syxin/janus/Janus_VIT_dola_early_exit修改。  

好的——这是**方案②：SigLIP ViT-L/16 使用 DoLa（不带 Early Exit / 固定对比层，DoLa-static）**的落地说明。依然坚持**只新增逻辑、零改架构/权重**，并且**全部沿用 DoLa 原论文公式**（只是把“词表”解释为“类别/文本候选集”）。

### CODING PROTOCOL ###
开发守则：
- 严格用最少的代码完成当前任务
- 不进行大规模改动
- 不做无关编辑，专注于你正在开发的任务
- 代码必须精确、模块化、可测试
- 不破坏现有功能
- 如果需要我做任何配置（例如Supabase/AWS），请明确告诉我
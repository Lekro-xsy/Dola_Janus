这是**方案⑤：SigLIP ViT-L/16 与 LLaMA（30 层）同时使用 DoLa，且都开启 Early Exit（动态选层）**的落地说明。坚持**只新增逻辑，不改任何架构/权重**，并给出**端到端的对接与融合公式**。  
（你已经有前四个方案的单塔实现；本方案在此基础上把两端串起来。）

---

# 方案⑤：ViT+LLaMA 双塔 DoLa（均带 Early Exit）

## 0) 目标与约束
- **目标**：视觉塔（SigLIP ViT-L/16）与语言塔（LLaMA-30L）都采用 **DoLa：层对比 + 动态 Early Exit + APC 头部筛选**，各自得到更判别/稳健的分布；再在**不改架构**的前提下进行**轻量融合**，指导最终文本生成。
- **不改结构/权重**：不动 ViT/LLaMA 的 block、norm、投影/`lm_head`；仅新增若干**后处理函数**与**生成时的分布融合**（logit bias/词表约束/前缀注入），属于“接线级”改动。

---

## 1) 两端各自的 DoLa 标准公式（回顾与统一记号）

### 1.1 视觉端（SigLIP ViT，候选集合 \(\mathcal{C}\) = 文本提示/类别）
- **成熟分布（最终层）**  
  \[
  s_N(c)=\tfrac{\langle z_N,t_c\rangle}{\tau},\qquad 
  q_N^{(V)}(c)=\mathrm{softmax}\big(s_N(c)\big).
  \]
- **早退分布（中间层 \(j\)）**  
  \[
  s_j(c)=\tfrac{\langle z_j,t_c\rangle}{\tau},\qquad 
  q_j^{(V)}(c)=\mathrm{softmax}\big(s_j(c)\big).
  \]
- **动态选层**  
  \[
  M_V=\arg\max_{j\in\mathcal{J}_V}\mathrm{JSD}\!\left(q_N^{(V)}\Vert q_j^{(V)}\right).
  \]
- **APC（头部集）**  
  \[
  \mathcal{V}^{(V)}=\big\{c:\,q_N^{(V)}(c)\ge \alpha_V\cdot\max_u q_N^{(V)}(u)\big\}.
  \]
- **对比融合（视觉端最终分布）**  
  \[
  F^{(V)}(c)=\log q_N^{(V)}(c)-\log q_{M_V}^{(V)}(c),\quad 
  \hat p_V(c)=\mathrm{softmax}\big(F^{(V)}(c)\big)\ \text{on}\ \mathcal{V}^{(V)}.
  \]

### 1.2 语言端（LLaMA，自回归词表 \(\mathcal{X}\)）
- **成熟分布（最终层）**  
  \[
  q_N^{(L)}(x)=\mathrm{softmax}\!\big(\phi(\mathrm{LN}_{\text{out}}(h_t^{(N)}))\big)_x .
  \]
- **早退分布（中间层 \(j\)）**  
  \[
  q_j^{(L)}(x)=\mathrm{softmax}\!\big(\phi(\mathrm{LN}_{\text{out}}(h_t^{(j)}))\big)_x .
  \]
- **动态选层**  
  \[
  M_L=\arg\max_{j\in\mathcal{J}_L}\mathrm{JSD}\!\left(q_N^{(L)}\Vert q_j^{(L)}\right).
  \]
- **APC（头部集）**  
  \[
  \mathcal{V}^{(L)}=\big\{x:\,q_N^{(L)}(x)\ge \alpha_L\cdot\max_w q_N^{(L)}(w)\big\}.
  \]
- **对比融合（语言端每步最终分布）**  
  \[
  F^{(L)}(x)=\log q_N^{(L)}(x)-\log q_{M_L}^{(L)}(x),\quad 
  \hat p_L(x)=\mathrm{softmax}\big(F^{(L)}(x)\big)\ \text{on}\ \mathcal{V}^{(L)}.
  \]

> 上述公式全部是 DoLa 原始写法的**逐模态直译**：视觉端把“词表”换成“文本提示/类别集合”。

---

## 2) 端到端对接与**不改架构的融合**（三种可选，推荐 B）

记视觉端的最终分布为 \(\hat p_V(c)\)，语言端（某步）的 DoLa 分布为 \(\hat p_L(x)\)。令 \(\mathcal{T}(c)\subseteq \mathcal{X}\) 表示“与候选 \(c\) 直接相关的词表子集”（例如候选短语的起始 token、或候选词的若干同义/子词单元）。我们提供三种**零改模型**的融合方式：

### A) **头部交集（Hard gating，最保守）**
把语言端 APC 的头部词再与视觉证据做交集（或加权交并）：
\[
\mathcal{S}_V=\bigcup_{c\in\text{Top-}K(\hat p_V)}\mathcal{T}(c),\qquad 
\mathcal{V}^{(\text{joint})}=\mathcal{V}^{(L)}\cap \mathcal{S}_V .
\]
然后仅在 \(\mathcal{V}^{(\text{joint})}\) 上归一化 \(\hat p_L\)。  
优点：实现简单；缺点：过于保守，可能误杀可行词。

### B) **对数域可加融合（推荐，最平衡）**
在**不改模型结构**的前提下，对语言端 **DoLa 对比分数** \(F^{(L)}(x)\) 施加**视觉先验偏置**：
\[
r_V(x)=\log\!\Big(\sum_{c:\,x\in \mathcal{T}(c)} \hat p_V(c)\Big)\quad(\text{若无命中则取 }-\infty\ \text{或 }0),
\]
\[
F^{(\text{joint})}(x)=F^{(L)}(x)+\lambda\cdot r_V(x),\qquad 
\hat p_{\text{joint}}(x)=\mathrm{softmax}\big(F^{(\text{joint})}(x)\big)\ \text{on}\ \mathcal{V}^{(L)}.
\]
其中 \(\lambda\ge 0\) 是融合强度（默认 0.5–1.0）。  
直观：仍然以**语言端 DoLa**为主，只是给与视觉一致的 token 以**对数加性偏置**。

### C) **前缀注入（Prompt-only，无需打分融合）**
把视觉端 Top-K 候选（按 \(\hat p_V\) 排序）格式化为**结构化前缀**（如「Detected: dog, frisbee, park」或「<tags>…</tags>」），拼到文本输入最前部；语言端仍按其 **DoLa** 解码。这种方式**不动词表/打分**，但会改变上下文内容。  
优点：最稳妥；缺点：对 prompt 依赖较强，融合力度不可直接控。

> 实战建议：**先用 B（对数域可加融合）**；如需更强约束再叠加 A 的硬 gating；想要极小改动时用 C。

---

## 3) 流水线与时序（一步到位）

**Step-V（视觉）**  
1. 用 SigLIP 正常前向得 \(z_N\) 与 \(q_N^{(V)}\)。  
2. 取候选层集合 \(\mathcal{J}_V\)（分桶 + 偶数层 3–5 个），计算 \(\{q_j^{(V)}\}\)。  
3. **Early Exit**：\(M_V=\arg\max_{j}\mathrm{JSD}(q_N^{(V)}\Vert q_j^{(V)})\)。  
4. **APC**：\(\mathcal{V}^{(V)}=\{c:\,q_N^{(V)}(c)\ge\alpha_V\cdot\max q_N^{(V)}\}\)。  
5. **对比融合**：\(\hat p_V(c)\propto \exp\big(\log q_N^{(V)}(c)-\log q_{M_V}^{(V)}(c)\big)\)（仅在 \(\mathcal{V}^{(V)}\) 上）。

**Step-Map（词表映射）**  
构造 \(\mathcal{T}(c)\)（候选到词表的映射）。常见做法：  
- 直接用候选短语的 **BPE 起始 token** 集合作为 \(\mathcal{T}(c)\)；  
- 或维护一个**同义词/别名词典**，并取其起始子词。  
此步骤**不改模型**，只是离线/加载的词典。

**Step-L（语言，每步解码）**  
1. 打开 `output_hidden_states=True, use_cache=True`，取最后位置 \(\{h_t^{(j)}\}\)。  
2. 构造 \(q_N^{(L)}\)、\(\{q_j^{(L)}\}\)；  
3. **Early Exit**：\(M_L=\arg\max_j\mathrm{JSD}(q_N^{(L)}\Vert q_j^{(L)})\)；  
4. **APC**：\(\mathcal{V}^{(L)}=\{x:\,q_N^{(L)}(x)\ge\alpha_L\cdot\max q_N^{(L)}(w)\}\)；  
5. **DoLa 对比分布**：\(\hat p_L(x)\propto \exp(\log q_N^{(L)}(x)-\log q_{M_L}^{(L)}(x))\)；  
6. **跨模态融合（选 B）**：按上式计算 \(r_V(x)\)，得到  
   \[
   \hat p_{\text{joint}}(x)=\mathrm{softmax}\big(F^{(L)}(x)+\lambda r_V(x)\big)\ \text{on}\ \mathcal{V}^{(L)}.
   \]  
7. 在 \(\hat p_{\text{joint}}\) 上继续你原有的采样（温度、top-k/p、beam 等）。

---

## 4) “改哪里”（只新增，不覆盖）

### 4.1 公用工具（建议新文件 `janus/utils/dola_runtime.py`）
- `build_probs_from_rep(z, text_bank, tau)`：SigLIP 端 \(q(\cdot)\)。  
- `compute_probs_from_hidden(h, ln_out, phi)`：LLaMA 端 \(q(\cdot)\)。  
- `jsd(P,Q)`、`select_early_exit(qN,{qj})`：JSD 与动态选层。  
- `contrast_and_mask(qN,qM,alpha)`：APC + 对数比 + 归一化（两端复用）。

### 4.2 视觉端接线（不改 ViT 类）
- 在现有视觉打分处新增函数（例如 `rerank_with_dola_siglip(...)`）：  
  读取中间层表示（`get_intermediate_layers(..., norm=True)`）、构造 \(\hat p_V\)。

### 4.3 语言端接线（不改 LLaMA 类）
- 新增 `generate_with_dola(...)`（或开关 `use_dola_lm=True`）：  
  在每步 logits→采样之间调用 DoLa（\(\hat p_L\)），替换分布。

### 4.4 融合层（新文件 `janus/utils/mm_fusion.py`）
- 维护 \(\mathcal{T}(c)\) 词典与 `r_V(x)` 计算；  
- 实现方案 B 的  
  \[
  F^{(\text{joint})}(x)=F^{(L)}(x)+\lambda r_V(x)
  \]
  并在语言端分布替换前调用；  
- （可选）方案 A 的 hard gating 与方案 C 的前缀格式化函数。

> 整个过程**只是在推理时多做几步计算/融合**，不改变任一 backbone 或 head。

---

## 5) 超参与推荐默认
- **APC 阈值**：\(\alpha_V=\alpha_L=0.1\)。  
- **候选层数**：每端各 3–5 个候选层（分桶优先，偶数层）。  
- **融合强度**：\(\lambda=0.7\) 起步（0–1 探索）。  
- **Top-K（视觉→词表映射）**：\(K=5\) 起步。  
- **温度/重复惩罚**：与基线一致；若你在基线对 logits 做重复惩罚，保持惩罚→DoLa→融合的固定顺序，方便 A/B。

---

## 6) 诊断、回退与一致性
- **任一端回退**：令该端 \(\alpha\to 0\) 且强制 \(M=N\)（或跳过对比）即可回到基线分布。  
- **融合回退**：设 \(\lambda=0\) 即只用语言端 DoLa；或不用 \(\mathcal{T}\) 直接退到方案③。  
- **可视化**：记录两端 \(\mathrm{JSD}\)-vs-层号热图、以及融合前后 Top-K 变化；对错误样本打印 \(\hat p_V\) 与 \(\hat p_L\) 的差异，观察视觉先验是否纠正了语言误导。  
- **开销**：相对方案③/①仅多一次 `r_V(x)` 与 `softmax` 计算；总体时延开销仍在可控范围（通常个位数百分比）。

---

### 小结（你要做的）
1) 给两端分别接上 **DoLa+Early Exit**（已在方案①与③给清楚）。  
2) 新增一个**跨模态融合层**：优先用 **对数域可加融合（B）**，不改模型，仅加一个可控的 logit 偏置。  
3) 维护一个简易的候选→词表映射词典 \(\mathcal{T}(c)\)（可自动从候选短语做 BPE 起始 token 提取）。  
4) 在语言端生成循环里，用 \(\hat p_{\text{joint}}\) 替换原分布。

# /z_data/migration/syxin/janus/Janus_combined_dola_early_exit在这里修改

### CODING PROTOCOL ###
开发守则：
- 严格用最少的代码完成当前任务
- 不进行大规模改动
- 不做无关编辑，专注于你正在开发的任务
- 代码必须精确、模块化、可测试
- 不破坏现有功能
- 如果需要我做任何配置（例如Supabase/AWS），请明确告诉我
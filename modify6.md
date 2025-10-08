# 方案⑥：ViT + LLaMA 双塔 DoLa-static（均不带 Early Exit）

> **目标**：SigLIP ViT-L/16 与 LLaMA（30 层）两端均采用 **DoLa 的层对比（固定中间层）+ APC 头部筛选**，分别得到更判别的分布，再做**轻量级跨模态融合**指导最终生成/决策。  
> **约束**：不改任一 backbone/block、Norm、`lm_head`/投影头或任何训练权重；只**新增后处理**与**推理时融合**步骤（零侵入）。

---

## 1) 标准公式（双塔、静态对比）

### 1.1 视觉端（SigLIP ViT；候选集合 \(\mathcal{C}\) = 文本提示/类别）

**成熟分布（最终层）**
\[
s_N(c)=\frac{\langle z_N,t_c\rangle}{\tau},\qquad
q_N^{(V)}(c)=\frac{\exp(s_N(c))}{\sum_{u\in\mathcal{C}}\exp(s_N(u))}.
\]

**固定中间层分布（选定 \(j_V^\*\)）**
\[
s_{j_V^\*}(c)=\frac{\langle z_{j_V^\*},t_c\rangle}{\tau},\qquad
q_{j_V^\*}^{(V)}(c)=\mathrm{softmax}\!\big(s_{j_V^\*}(c)\big).
\]
> \(z_{j_V^\*}\) 由 `get_intermediate_layers(..., norm=True)` + 与最终层一致的 `forward_head(...)` 得到；确保与最终层**同路径**读出（不加新头）。

**APC（头部筛选）**
\[
\mathcal{V}^{(V)}=\Big\{c:\;q_N^{(V)}(c)\ge \alpha_V\cdot\max_{u} q_N^{(V)}(u)\Big\}.
\]

**对比融合（视觉端最终分布）**
\[
F^{(V)}(c)=\log q_N^{(V)}(c)-\log q_{j_V^\*}^{(V)}(c),\qquad
\hat p_V(c)=\mathrm{softmax}\big(F^{(V)}(c)\big)\ \text{on}\ \mathcal{V}^{(V)}.
\]

---

### 1.2 语言端（LLaMA；词表 \(\mathcal{X}\)）

**成熟分布（最终层）**
\[
\tilde h_t^{(N)}=\mathrm{LN}_{\text{out}}(h_t^{(N)}),\quad
\ell_N=\phi(\tilde h_t^{(N)}),\quad
q_N^{(L)}(x)=\mathrm{softmax}(\ell_N)_x.
\]

**固定中间层分布（选定 \(j_L^\*\)）**
\[
\tilde h_t^{(j_L^\*)}=\mathrm{LN}_{\text{out}}(h_t^{(j_L^\*)}),\quad
\ell_{j_L^\*}=\phi(\tilde h_t^{(j_L^\*)}),\quad
q_{j_L^\*}^{(L)}(x)=\mathrm{softmax}(\ell_{j_L^\*})_x.
\]
> 中间层与最终层**共用**同一 \(\mathrm{LN}_{\text{out}}\) 与 \(\phi\)；不新增参数。

**APC（头部筛选）**
\[
\mathcal{V}^{(L)}=\Big\{x:\;q_N^{(L)}(x)\ge \alpha_L\cdot\max_{w} q_N^{(L)}(w)\Big\}.
\]

**对比融合（语言端每步最终分布）**
\[
F^{(L)}(x)=\log q_N^{(L)}(x)-\log q_{j_L^\*}^{(L)}(x),\qquad
\hat p_L(x)=\mathrm{softmax}\big(F^{(L)}(x)\big)\ \text{on}\ \mathcal{V}^{(L)}.
\]

---

## 2) 跨模态融合（不改架构，三选一；**推荐 B**）

令视觉端最终分布为 \(\hat p_V(c)\)，语言端某步分布为 \(\hat p_L(x)\)。设 \(\mathcal{T}(c)\subseteq \mathcal{X}\) 把候选 \(c\) 映射到词表（如候选短语的 **BPE 起始子词**集合）。



### B) 对数域可加融合（**推荐**）
\[
r_V(x)=\log\!\Big(\sum_{c:\,x\in\mathcal{T}(c)} \hat p_V(c)\Big)\ \ (\text{未命中可取 }-\infty\text{ 或 }0),
\]
\[
F^{(\text{joint})}(x)=F^{(L)}(x)+\lambda\cdot r_V(x),\qquad
\hat p_{\text{joint}}(x)=\mathrm{softmax}\big(F^{(\text{joint})}(x)\big)\ \text{on}\ \mathcal{V}^{(L)}.
\]
> 不改模型，仅给语言端 DoLa 对比分数**加一个视觉先验偏置** \(\lambda\cdot r_V(x)\)（\(\lambda\in[0,1]\)）。


---

## 3) 端到端流程（一步到位）

**Step-V（视觉）**  
1) 前向得到 \(z_N\) 与 \(q_N^{(V)}\)；  
2) 读取**固定层** \(j_V^\*\) 得 \(z_{j_V^\*}\) → \(q_{j_V^\*}^{(V)}\)；  
3) APC：\(\mathcal{V}^{(V)}=\{c:\,q_N^{(V)}(c)\ge\alpha_V\cdot\max q_N^{(V)}\}\)；  
4) 视觉 DoLa 对比：\(\hat p_V(c)\propto \exp\!\big(\log q_N^{(V)}(c)-\log q_{j_V^\*}^{(V)}(c)\big)\)（仅在 \(\mathcal{V}^{(V)}\)）。

**Step-Map（候选→词表映射）**  
构造/加载 \(\mathcal{T}(c)\)：用候选短语的 **BPE 起始 token** 或同义词起始 token 集合。

**Step-L（语言；每步解码）**  
1) 打开 `output_hidden_states=True, use_cache=True`；  
2) 构造 \(q_N^{(L)}\) 与 \(q_{j_L^\*}^{(L)}\)；  
3) APC：\(\mathcal{V}^{(L)}=\{x:\,q_N^{(L)}(x)\ge\alpha_L\cdot\max q_N^{(L)}(x)\}\)；  
4) 语言 DoLa 对比：\(\hat p_L(x)\propto \exp\!\big(\log q_N^{(L)}(x)-\log q_{j_L^\*}^{(L)}(x)\big)\)（仅在 \(\mathcal{V}^{(L)}\)）。  
5) **跨模态融合（选 B）**：  
   \[
   \hat p_{\text{joint}}(x)=\mathrm{softmax}\big(F^{(L)}(x)+\lambda r_V(x)\big)\ \text{on}\ \mathcal{V}^{(L)}.
   \]
6) 在 \(\hat p_{\text{joint}}\) 上执行既有采样（温度、top-k/p、beam…）。

---

## 4) 你需要“改哪里”（**只新增，不覆盖**）

### 4.1 公用工具（建议：`janus/utils/dola_runtime.py`）
- `build_probs_from_rep(z, text_bank, tau)`：SigLIP 端 \(q(\cdot)\)。  
- `compute_probs_from_hidden(h, ln_out, phi)`：LLaMA 端 \(q(\cdot)\)。  
- `contrast_and_mask(qN, qJ, alpha)`：APC + \(\log\frac{q_N}{q_J}\) + 归一化（两端复用）。  
> 本方案为 **static**，不需要 JSD/选层函数。

### 4.2 视觉端接线（不改 ViT 类）
- 新增（例如）`rerank_with_dola_static_siglip(image, text_bank, j_V*, alpha_V=0.1, tau=...)`：  
  用 `get_intermediate_layers(..., norm=True)` + `forward_head(...)` 取 \(z_{j_V^\*}, z_N\)，计算 \(\hat p_V\)。

### 4.3 语言端接线（不改 LLaMA 类）
- 新增 `generate_with_dola_static(...)` 或开关 `use_dola_lm_static=True`：  
  在“每步 logits→采样”之间把 \(q_N^{(L)}\) 用 `contrast_and_mask(q_N^{(L)}, q_{j_L^\*}^{(L)}, \alpha_L)` 替换为 \(\hat p_L\)。

### 4.4 融合层（建议：`janus/utils/mm_fusion.py`）
- 维护 \(\mathcal{T}(c)\) 与 `r_V(x)` 计算；  
- 实现方案 B 的
  \[
  F^{(\text{joint})}(x)=F^{(L)}(x)+\lambda r_V(x)
  \]
  并在语言端分布替换前最后一步调用；  

### 4.5 配置开关
- `use_dola_siglip_static`、`use_dola_lm_static`、`lambda_mm`、`alpha_V`、`alpha_L`、`fixed_layer_vit=j_V*`、`fixed_layer_llm=j_L*`、`mm_topk_for_T=K` 等。

---

## 5) 固定层如何选（一次线下确定）

- **ViT（24 层）**：分 **3 桶** \([1,8],[9,16],[17,24]\)。先在后两个桶试 \(\{16,18,20,22\}\)，常见起点 \(j_V^\*=20\) 或 \(22\)。  
- **LLaMA（30 层）**：分 **3 桶** \([1,10],[11,20],[21,29]\)。常见起点 \(j_L^\*=\{22,24,26\}\)。  
- **策略**：用小验证集**网格搜索**确定 \(j_V^\*\)、\(j_L^\*\)；若不同子域差异大，可按域保存两套固定层（推理时按域选择）。

---

## 6) 超参数与默认值

- **APC 阈值**：\(\alpha_V=\alpha_L=0.1\)（可在 0.05–0.2 微调）。  
- **融合强度**：\(\lambda=0.7\) 起步（0–1 探索）。  
- **映射 Top-K**：从 \(\hat p_V\) 取 \(K=5\) 做 \(\mathcal{T}(c)\) 的候选集合。  
- **SigLIP 温度**：\(\tau\) 沿用你当前配置，不另行修改。  
- **采样超参**：温度、top-k/p、重复惩罚、beam 与基线一致（只是把分布来源换为 \(\hat p_{\text{joint}}\)）。

---

## 7) 数值与工程注意

- **读头一致性**：中间层读出必须走与最终层**相同**的归一化与投影（ViT：`norm=True` + `forward_head`；LLaMA：`LN_out` + `lm_head`）。  
- **数值稳定**：对比用 `log_softmax` 形式计算 \(\log q\)；对零概率加 \(\varepsilon\)（如 \(10^{-12}\)）。  
- **KV-cache 不受影响**：语言端仅多一次“把中间层最后位置过 `LN_out+phi+softmax`”，不影响注意力与缓存。  
- **复杂度**：比基线增加很小的读头/`softmax` 代价；比“带 Early Exit”的方案更轻（无需多层评估与选层）。

---

## 8) 诊断与回退

- **端到端回退**：令 \(\alpha_V,\alpha_L\to 0\) 且把 \(q_{j^\*}\) 替换为 \(q_N\)，立刻退回原分布；或直接关 `use_dola_*` 开关。  
- **融合回退**：设 \(\lambda=0\) 即仅用语言端 DoLa；或仅保留视觉 DoLa 重排序。  
- **灵敏度检查**：在验证集画“层号→指标”曲线，确认 \(j_V^\*, j_L^\*\) 稳定；如漂移大，考虑切换到**方案⑤**（带 Early Exit）。

---

### 小结（你要做的）

1) 给 **SigLIP** 与 **LLaMA** 各接上 **DoLa-static**（固定层对比 + APC），不改任何权重。  
2) 新增一个**融合层**（推荐对数域可加融合 B），把视觉先验以 \(\lambda r_V(x)\) 形式加到语言端 DoLa 对比分数上。  
3) 线下用小验证集各自确定 \(j_V^\*, j_L^\*\)；线上固定使用。  
4) 通过若干布尔/数值开关即可在运行时灵活开启/关闭与调参。

# /z_data/migration/syxin/janus/Janus_combined_dola_no_early_exit在这里修改

### CODING PROTOCOL ###
开发守则：
- 严格用最少的代码完成当前任务
- 不进行大规模改动
- 不做无关编辑，专注于你正在开发的任务
- 代码必须精确、模块化、可测试
- 不破坏现有功能
- 如果需要我做任何配置（例如Supabase/AWS），请明确告诉我
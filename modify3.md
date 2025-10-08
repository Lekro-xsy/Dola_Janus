下面是**方案③：LLaMA-系（30 层）使用 DoLa，带 Early Exit（动态选层）**的落地说明。全程**只新增逻辑，不改模型架构/权重**；**所有公式严格沿用 DoLa 原论文写法**（仅把符号对齐到自回归 LM 语境）。
# 方案③：LLaMA (30L) + DoLa（Early Exit）

## 0. 目标与约束
- **目标**：在自回归解码时，用 DoLa 的“层对比 + 动态早退”重写每步的**下一词分布**，以更稳的事实性/判别性。
- **不改架构/权重**：不动 transformer block、RMSNorm、`lm_head`/词嵌入；**只在生成循环外插一个“读中间层→读头→对比→替换分布”的分支**。
- **接线位置**：在你调用 `generate`/自定义解码循环的那一层（你仓库里通常是“语言模型推理入口/包装器”），**新增**一个 `generate_with_dola(...)`（或等价开关）。原 `generate` 不动。

---

## 1. DoLa 的标准公式（LM 语境）

**符号与设定**  
- 层数：\(N\)（这里 \(N=30\)）。  
- 第 \(t\) 步最后一个位置（新 token 要生成的位置）的隐状态：\(h^{(j)}_t\) 表示第 \(j\) 层输出。  
- 若模型有**最终输出层归一化**（如 LLaMA 的 `final_rmsnorm`），记为 \(\mathrm{LN}_\text{out}(\cdot)\)。  
- 词表投影头（与嵌入权重**权重共享**或独立均可）：\(\phi(\cdot)\)。

**(1) 成熟分布（最终层）**  
\[
\tilde h^{(N)}_t=\mathrm{LN}_\text{out}\!\left(h^{(N)}_t\right),\qquad
\ell_N=\phi\!\left(\tilde h^{(N)}_t\right)\in\mathbb{R}^{|\mathcal{X}|},\qquad
q_N(x)=\mathrm{softmax}(\ell_N)_x.
\]

**(2) 早退分布（中间层 \(j\)）**  
\[
\tilde h^{(j)}_t=\mathrm{LN}_\text{out}\!\left(h^{(j)}_t\right),\qquad
\ell_j=\phi\!\left(\tilde h^{(j)}_t\right),\qquad
q_j(x)=\mathrm{softmax}(\ell_j)_x,\quad j\in\mathcal{J}\subset\{1,\dots,N-1\}.
\]
> 关键点：**中间层也走同一 \(\mathrm{LN}_\text{out}\) 与同一 \(\phi\)**，以保持读出一致性（不新增可训练参数）。

**(3) 动态早退层选择（Early Exit）**  
对候选集合 \(\mathcal{J}\)，按 **Jensen–Shannon 散度**选择与成熟分布差异最大的层：
\[
M=\arg\max_{j\in\mathcal{J}}\mathrm{JSD}\!\left(q_N\Vert q_j\right),
\quad\text{其中}\quad
\mathrm{JSD}(P\Vert Q)=\tfrac12\,\mathrm{KL}\!\left(P\Vert \tfrac{P+Q}{2}\right)+\tfrac12\,\mathrm{KL}\!\left(Q\Vert \tfrac{P+Q}{2}\right).
\]

**(4) 自适应头部约束（APC）**  
\[
\mathcal{V}_{\text{head}}=\Big\{x\in\mathcal{X}:\; q_N(x)\ge \alpha\cdot \max_{w\in\mathcal{X}}q_N(w)\Big\},
\]
常用 \(\alpha=0.1\)（可调）。

**(5) 对比融合（DoLa 核心）**  
仅在 \(\mathcal{V}_{\text{head}}\) 上做**对数比**：
\[
F(x)=\log q_N(x)-\log q_M(x)=\log\frac{q_N(x)}{q_M(x)},\quad x\in\mathcal{V}_{\text{head}},
\]
对 \(F\) 做 softmax 得最终分布（屏蔽集合外词）：
\[
\hat p(x)=\frac{\exp(F(x))}{\sum_{u\in\mathcal{V}_{\text{head}}}\exp(F(u))}.
\]
随后按你原有策略（greedy / top-k / nucleus / 温度 / beam）在 \(\hat p\) 上采样与打分即可。

> 如需“重复惩罚”\(\theta>1\)，放在 \(\hat p\) 之前对 \(\ell_N\) 或 \(q_N\) 做你原先的惩罚，再进入上述流程（保持与基线一致的插入点）。

---

## 2. 生成循环中的**新增步骤**（每个解码步）

> 仅描述“改哪里/加哪里”，不贴代码。

1) **打开中间层输出**：调用模型前向时设置  
   `output_hidden_states=True, use_cache=True`（保持 KV-cache 不变）。

2) **取最后位置隐状态**：从返回的 `hidden_states`（长度 \(N\) 或 \(N{+}1\)，视实现而定）抓取各层在**当前位置 \(t\)** 的向量 \(h^{(j)}_t\in\mathbb{R}^d\)（只要**最后一个 token**，无需整句）。

3) **候选层集合 \(\mathcal{J}\)**（分桶 + 取样）：
   - 将 30 层划为 **3 个 bucket**：\([1,10],[11,20],[21,29]\)（最终层 \(N\!=\!30\) 专用于 \(q_N\)）。  
   - 先用一个小验证集选择**哪个 bucket**更优；在线推理时只在该 bucket 内取**3–5 个候选层**（如偶数层 \(\{22,24,26,28\}\)）。

4) **构造 \(q_N\)**：对 \(\tilde h^{(N)}_t\) 过 \(\phi\) 与 softmax 得 \(q_N\)。

5) **构造 \(\{q_j\}_{j\in\mathcal{J}}\)**：对每个候选层 \(j\)，先过相同 \(\mathrm{LN}_\text{out}\) 再过同一 \(\phi\) 与 softmax。

6) **选层 \(M\)**：按 \(\mathrm{JSD}(q_N\Vert q_j)\) 最大化取 \(M\)。

7) **APC + 对比融合**：  
   - \(\mathcal{V}_{\text{head}}=\{x:q_N(x)\ge\alpha\cdot\max q_N\}\)；  
   - \(F(x)=\log q_N(x)-\log q_M(x)\)（仅 \(\mathcal{V}_{\text{head}}\) 上）；  
   - 归一化得 \(\hat p\)，作为该步的**最终下一词分布**。

8) **采样/打分**：在 \(\hat p\) 上执行你既有的采样策略、温度缩放、top-k/p、beam 扩展等（次序与基线一致，唯一变化是“分布来源”从 \(q_N\) 换成 \(\hat p\)）。

> **KV-cache 与计算量**：仅增加若干次“把中间层**最后位置**过 \(\mathrm{LN}_\text{out}\) 和 \(\phi\) + softmax”的开销；不改自注意力、KV-cache 与 block 结构。

---

## 3. 你需要“新增”的模块/开关（不覆盖原实现）

- **新增工具模块**（建议）：`dola_runtime.py`  
  - `compute_probs_from_hidden(h, ln_out, phi) → q`：对单层最后位置隐状态做 \(\mathrm{LN}_\text{out}\rightarrow\phi\rightarrow\mathrm{softmax}\)。  
  - `jsd(P,Q)`：JSD。  
  - `select_layer(qN, {qj}) → M`：最大 JSD 选层。  
  - `contrast_with_apc(qN, qM, alpha) → p_hat`：APC + 对数比 + 归一化。
- **新增生成入口**：`generate_with_dola(...)`（或在你现有生成入口加开关 `use_dola_lm=True`）。  
  - 仅在**每步拿到 logits 之后、采样之前**，调用上面工具函数把 \(q_N\to\hat p\)。

> 这样做能保证**原模型类与原 `generate`** 完整保留；新入口可与基线 A/B 对比。

---

## 4. 张量与数值细节（避免坑）

- **输出层归一化**：若模型在最终层后有 `final_rmsnorm`（或其他 `LN_out`），务必也对中间层 \(h^{(j)}_t\) 先做同样的归一化，再送入 \(\phi\)。  
- **权重共享**：若 \(\phi(h)=E^\top h + b\) 且 \(E\) 与词嵌入共享权重，**仍然用同一份 \(\phi\)**，不复制。  
- **数值稳定**：计算 \(F=\log q_N - \log q_M\) 用 `log_softmax` 形式更稳；JSD/KL 时对零概率加 \(\varepsilon\)（如 \(1\mathrm{e}{-}12\)）。  
- **批/束**：beam search 时对每个 beam 独立取 \(h^{(j)}_t\) 与分布；张量维度通常是 `[num_beams, vocab]`。  
- **效率**：可把 JSD 的计算限制在 \(\mathcal{V}_{\text{head}}\cup\) 若干高频词上作**近似**，但**公式本身保持不变**（只是在实现上裁剪支持集）。

---

## 5. 超参与建议默认
- **\(\alpha\)**（APC 阈值）：0.1（任务更难可降至 0.05）。  
- **候选层数**：每步 3–5 个足够。  
- **分桶**：30 层 → 3 桶（前/中/后），先用验证集挑一个桶。  
- **采样温度**：与基线一致（DoLa 替换分布，不改变你的温度语义）。  
- **重复惩罚**：保持与你基线**相同位置**插入（推荐对 \(\ell_N\) 或 \(q_N\) 先做，再进入 DoLa）。

---

## 6. 验证与回退

- **一致性回退**：当 \(\alpha\to 0\) 且强制 \(M=N\)（或 \(q_M=q_N\)）时，\(\hat p\equiv q_N\)，应与原生成逐 token 对齐。  
- **JSD 轨迹**：记录每步 \(\mathrm{JSD}(q_N\Vert q_j)\) vs. 层号热图；“信息密集”或“需要推理”的步通常在后段层 JSD 更大（可作为质检）。  
- **时延**：与基线比较实时延；合理设置候选层后，通常只增加约 1–8% 的生成时延。

---

## 7. 你要“改哪里”（文字版接线图，**只新增**）

1) **语言模型推理入口**（你当前包装 `generate` 的文件/类）：  
   - 新增 `generate_with_dola(...)`（或在现有 `generate` 外再包一层），在内部**每步**插入“取中间层→DoLa 对比→替换分布”的流程；别动原 `generate`。  
2) **模型前向调用处**：把 `output_hidden_states=True, use_cache=True` 传给底层 LM；从 `hidden_states` 里提取各层**最后位置**的 \(h^{(j)}_t\)。  
3) **新增模块**：放置 `dola_runtime.py`（JSD、APC、对数比、分桶/选层等工具函数），供 LLaMA 与（未来）ViT 方案复用。  
4) **配置开关**：在推理配置里加 `use_dola_lm`、`alpha`、`candidate_layers`/`bucket_id` 等项，便于 A/B。
在这里改/z_data/migration/syxin/janus/Janus_llama_dola_early_exit
---

到这里，**方案③（LLaMA + DoLa + Early Exit）**就完整了。  

# 在/z_data/migration/syxin/janus/Janus_llama_dola_early_exit修改

### CODING PROTOCOL ###
开发守则：
- 严格用最少的代码完成当前任务
- 不进行大规模改动
- 不做无关编辑，专注于你正在开发的任务
- 代码必须精确、模块化、可测试
- 不破坏现有功能
- 如果需要我做任何配置（例如Supabase/AWS），请明确告诉我
我给出**方案④：LLaMA（30 层）使用 DoLa（不带 Early Exit / 固定对比层）**。

# 方案④：LLaMA (30L) + DoLa-static（固定对比层）

> **约定**：全程**只新增逻辑，不改任何架构或权重**；所有推导**严格使用 DoLa 论文的标准公式**（把“层对比 + 对数比融合 + 头部约束”直接迁移到自回归 LM 的下一词分布上）。

---

## 0) 目标与约束
- **目标**：在生成解码时，用最终层分布与**固定中间层**分布的**对数比**来得到最终的下一词分布，提高判别性与稳健性。
- **不改架构/权重**：不动任何 Transformer Block、Norm、`lm_head`/embedding 共享；**仅在生成循环外加一段“从中间层读出→对比→替换分布”的后处理**。
- **接线位置**：在你包装的 `generate`/自定义解码循环层面，新增一个 `generate_with_dola_static(...)`（或一个布尔开关）。原生 `generate` 不动。

---

## 1) 公式（严格沿用 DoLa，LM 语境）

设总层数 \(N=30\)。第 \(t\) 步要生成的新 token 的位置记为“最后位置”。用 \(h^{(j)}_t\) 表示第 \(j\) 层在该位置的隐状态；\(\mathrm{LN}_{\text{out}}\) 为最终输出层归一化（如 LLaMA 的 `final_rmsnorm`）；\(\phi(\cdot)\) 为词表仿射头（与你模型的 `lm_head` 对齐）。

**(1) 成熟分布（最终层）**

\[
\tilde h^{(N)}_t=\mathrm{LN}_{\text{out}}\!\left(h^{(N)}_t\right),\quad
\ell_N=\phi\!\left(\tilde h^{(N)}_t\right)\in\mathbb{R}^{|\mathcal{X}|},\quad
q_N(x)=\mathrm{softmax}(\ell_N)_x.
\]

**(2) 固定早退层分布（中间层 \(j^\*\)）**  
> 这是与“带 Early Exit”唯一不同之处：我们**不用动态选层**，而是固定一个 \(j^\*\in\{1,\dots,N-1\}\)。

\[
\tilde h^{(j^\*)}_t=\mathrm{LN}_{\text{out}}\!\left(h^{(j^\*)}_t\right),\quad
\ell_{j^\*}=\phi\!\left(\tilde h^{(j^\*)}_t\right),\quad
q_{j^\*}(x)=\mathrm{softmax}(\ell_{j^\*})_x.
\]

> 关键：**中间层也必须过同一 \(\mathrm{LN}_{\text{out}}\) 与同一 \(\phi\)**，保证读出一致性；不新增参数。

**(3) 自适应头部约束（APC）**

\[
\mathcal{V}_{\text{head}}=\Big\{x\in\mathcal{X}:\; q_N(x)\ge \alpha\cdot \max_{w\in\mathcal{X}}q_N(w)\Big\},\quad \alpha\approx 0.1.
\]

**(4) 对比融合（DoLa 核心）**  
在 \(\mathcal{V}_{\text{head}}\) 上用**对数比**构造对比分数：

\[
F(x)=\log q_N(x)-\log q_{j^\*}(x)=\log\frac{q_N(x)}{q_{j^\*}(x)},\qquad x\in\mathcal{V}_{\text{head}}.
\]

将 \(\mathcal{V}_{\text{head}}\) 以外的词视为 \(-\infty\)（屏蔽），然后 softmax 得到最终分布：

\[
\hat p(x)=\frac{\exp(F(x))}{\sum_{u\in\mathcal{V}_{\text{head}}}\exp(F(u))}.
\]

最终在 \(\hat p\) 上执行你现有的解码策略（greedy / top-k / nucleus / 温度 / beam）。

> 说明：这与“带 Early Exit”的公式完全一致，只是把 \(M\)（动态选择的层）**替换为常数 \(j^\*\)**。

---

## 2) 生成循环中**要新增/改动的步骤**（逐步说明）

> 只写“加哪里/怎么接”，不写代码。

1. **启用中间层输出**  
   在每步解码前向时设置 `output_hidden_states=True, use_cache=True`，以获得各层在**最后位置**的隐状态 \(h^{(j)}_t\)。

2. **抓取目标位置的隐状态**  
   从 `hidden_states` 中提取所有层在最后位置的向量（只取最后一个 token 的向量，无需整句张量）。

3. **构造成熟分布 \(q_N\)**  
   对 \(h^{(N)}_t\) 施加同一 \(\mathrm{LN}_{\text{out}}\) 和同一 `lm_head`（\(\phi\)），得到 \(\ell_N\) 与 \(q_N\)。

4. **构造固定层分布 \(q_{j^\*}\)**  
   对 \(h^{(j^\*)}_t\) 用**相同**的 \(\mathrm{LN}_{\text{out}}\) 与 `lm_head` 得到 \(\ell_{j^\*}\) 与 \(q_{j^\*}\)。

5. **APC 头部筛选**  
   \(\mathcal{V}_{\text{head}}=\{x:\,q_N(x)\ge\alpha\cdot \max q_N\}\)，\(\alpha=0.1\) 起步。

6. **对比融合并替换分布**  
   计算 \(F=\log q_N-\log q_{j^\*}\)（仅在 \(\mathcal{V}_{\text{head}}\)），softmax 得 \(\hat p\)。  
   把原本用于采样/打分的分布从 \(q_N\)**替换为** \(\hat p\)。

7. **后续步骤不变**  
   温度缩放、top-k / nucleus、重复惩罚、beam 扩展等与基线保持相同的位置与次序（建议：如果你基线把重复惩罚作用在 logits 或 \(q_N\)，在进入 DoLa 前保持一致以便对比公平）。

---

## 3) 该“改哪里”（文件/模块层级上的最小改动）

- **新增一个工具模块**（与前面方案可共用）：例如 `janus/utils/dola_runtime.py`  
  - `compute_probs_from_hidden(h, ln_out, phi) → q`：对单层最后位置隐状态做 \(\mathrm{LN}_{\text{out}}\rightarrow\phi\rightarrow\mathrm{softmax}\)。  
  - `contrast_with_apc(qN, qJ, alpha) → p_hat`：实现 APC + 对数比 + 归一化。
- **新增一个生成入口/开关**  
  - 新函数：`generate_with_dola_static(...)`；或在你现有生成封装里加布尔开关 `use_dola_lm_static=True`。  
  - 只在“每步拿到 \(q_N\) 后、采样之前”调用 `contrast_with_apc(qN, q_{j^\*}, alpha)` 替换分布即可。  
- **绝不修改**底层 LLaMA 模型类与原 `generate`；这样便于与基线 A/B。

---

## 4) 如何选固定层 \(j^\*\)（一次线下确定）

- **验证集网格搜索（推荐）**：在任务域的**小验证集**上枚举若干层（例如 \(\{8,10,12,14,16,18,20,22,24,26,28\}\)），用与线上一致的解码配置评测，选最优 \(j^\*\)。  
- **分桶优先（提速）**：把 30 层分 **3 桶**（前/中/后：\([1,10],[11,20],[21,29]\)），先在每桶末段试 2–3 个层，再细化到单层。  
- **经验起点**：若无验证集，先从**靠后的中间层**试（如 22/24/26），通常更稳定；再微调。

> 注意：DoLa-static 的最优层对数据分布相对敏感；如果发现不同子域的最优层漂移明显，可以考虑切回**方案③（动态 Early Exit）**。

---

## 5) 数值与工程注意
- **归一化与读头共享**：中间层读出必须使用与最终层**完全相同**的 \(\mathrm{LN}_{\text{out}}\) 与 \(\phi\)；不要复制参数。  
- **稳定性**：计算对数比优先用 `log_softmax` 形式；对极小概率加 \(\varepsilon(10^{-12})\) 以免 \(\log 0\)。  
- **束与批**：beam search 时对每个 beam 独立构造 \(q_N\) 与 \(q_{j^\*}\)；维度形如 `[num_beams, vocab]`。  
- **复杂度**：比基线仅多一次“把中间层最后位置过 \(\mathrm{LN}_{\text{out}}\) 和 \(\phi\)”与一次 `softmax`；开销低于动态方案。

---

## 6) 超参数与默认建议
- **\(\alpha\)**（APC 阈值）：0.1（可在 0.05–0.2 间微调）。  
- **\(j^\*\)**：由验证集确定；若缺省先试 22 或 24。  
- **采样温度 / top-k / nucleus / 长度惩罚 / 重复惩罚**：全部沿用你基线配置；唯一变化是把采样分布从 \(q_N\) 换成 \(\hat p\)。

---

## 7) 诊断与回退
- **一致性回退**：当 \(\alpha\to 0\) 且强制 \(q_{j^\*}=q_N\)（或直接绕过对比）时，\(\hat p\equiv q_N\)，应与基线逐 token 对齐。  
- **敏感性检查**：在验证集画“层号 → 指标”的曲线，确认 \(j^\*\) 的稳健性；如漂移大，建议切到方案③。  
- **误差分析**：对错误样本打印 \(q_N\) 与 \(q_{j^\*}\) 的 Top-K，常见现象是 \(q_{j^\*}\) 对“表面相近但错误”的词更自信，DoLa 对数比能抑制这类干扰。

---

### 小结（只新增，不覆盖）
1) 加 `dola_runtime.py`（或共用前面已加的）：`compute_probs_from_hidden`、`contrast_with_apc`。  
2) 新增 `generate_with_dola_static(...)` 或开关 `use_dola_lm_static`，在每步把 \(q_N\) 替换为 \(\hat p\)。  
3) 线下用小验证集选好 \(j^\*\)，线上固定使用；其余推理流程保持不变。 
# 在这里修改/z_data/migration/syxin/janus/Janus_llama_dola_no_early_exit

### CODING PROTOCOL ###
开发守则：
- 严格用最少的代码完成当前任务
- 不进行大规模改动
- 不做无关编辑，专注于你正在开发的任务
- 代码必须精确、模块化、可测试
- 不破坏现有功能
- 如果需要我做任何配置（例如Supabase/AWS），请明确告诉我
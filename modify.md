# 评测改造说明（只用本地权重，不触网）

本页记录为按 dataset/task.md 要求，仅使用 `/z_data/migration/syxin/janus/dataset` 下已有权重完成评测所做的修改与使用方法（含命令）。

## 1. 做了哪些修改（代码路径 + 说明）

- t2v_metrics 对 timm 的兼容修复（避免远程下载、修正 API 变动）
  - 文件：`dataset/t2v_metrics/t2v_metrics/models/vqascore_models/lavis/common/dist_utils.py`
  - 改动：将 `import timm.models as timm_hub` 改为 `from timm.models import hub as timm_hub`
  - 目的：新老 timm 版本的 `download_cached_file/get_cache_dir` 存在于 `timm.models.hub`，避免 `AttributeError` 并且让权重缓存落在我们指定目录。

- HPSv2 只走本地包与本地权重
  - 文件：`eval_janus_metrics.py`
  - 改动：`try_import_hpsv2()` 强制把 `dataset/HPSv2_github` 加入 `sys.path` 并设置 `HPS_ROOT=dataset/HPSv2`，屏蔽掉可能存在的 site-packages 版本，确保走本地代码与本地权重。
  - 文件：`dataset/HPSv2_github/hpsv2/img_score.py`
  - 改动：`score()` 在 `cp is None` 时优先用本地 `$HPS_ROOT/HPS_v2*_compressed.pt`，只有本地缺失时才会尝试下载（当前我们有本地权重，不会触发下载）。

- ITMScore（blip2-itm）只走本地 Transformers 权重
  - 文件：`eval_janus_metrics.py`
  - 改动：在 `metric == 'itm'` 分支对 `BLIP2ITMScoreModel` 做 monkey‑patch：
    - 使用 `transformers.Blip2ForImageTextRetrieval` + `AutoProcessor`，从本地 `dataset/blip2-itm-vit-g` 加载；
    - `forward()` 使用 `logits_per_image` 的对角元素经 `sigmoid` 作为匹配分数；
    - 避开 LAVIS 里的 EVA ViT 权重远程 URL 与任何外部下载。

- CLIPScore（openai:ViT-L-14-336）优先本地 HF 权重
  - 文件：`eval_janus_metrics.py`
  - 逻辑：若存在 `dataset/clip-vit-large-patch14-336/`，则把 `CLIPScoreModel` 换成 `transformers.CLIPModel/CLIPProcessor`，从本地加载；否则才回退到 open_clip。

- PickScore（pickscore-v1）只走本地权重
  - 文件：`eval_janus_metrics.py`
  - 改动：将 `PickScoreModel.load_model()` 改为同时从 `dataset/PickScore_v1` 读取 `AutoProcessor` 与 `AutoModel`，完全本地加载。

- ImageReward 本地化兜底
  - 文件：`eval_janus_metrics.py`
  - 逻辑：优先用本地 `dataset/ImageReward/ImageReward.pt`；若缺 `clip` 包则优先使用本地 `dataset/CLIP` 代码。

- 统一缓存位置（不写到别处）
  - 文件：`eval_janus_metrics.py`
  - 逻辑：将 `HF_HOME/TRANSFORMERS_CACHE/HUGGINGFACE_HUB_CACHE/XDG_CACHE_HOME/TORCH_HOME/OPENCLIP_CACHE_DIR` 全部默认指向 `dataset/hf_cache`。

## 2. 依赖/环境变更

- Conda（在环境 `t2v` 中）：
  - 安装/确认：`ffmpeg=6.1.2`（conda‑forge）、`libiconv>=1.17`
- Pip：
  - `timm==0.9.16`（包含 `timm.layers`，与 LAVIS/TIMM API 兼容；open‑clip/t2v‑metrics 会报软冲突告警，但已通过本地化补丁避开其受约束路径，实测可用）
  - `transformers==4.49.0`（当前环境已存在并可用，含 `Blip2ForImageTextRetrieval`）

说明：ffmpeg 必须在 conda 的 `t2v` 环境内可执行，否则 t2v_metrics 在 import 阶段就会报错（exit status 127）。

## 3. 本地权重使用表

- VQA（推荐）：`dataset/clip-flant5-xxl`（脚本已重定向 tokenizer/model 到该目录）
- CLIPScore（ViT-L/14-336）：`dataset/clip-vit-large-patch14-336`
- ITMScore（BLIP2‑ITM）：`dataset/blip2-itm-vit-g`
- PickScore：`dataset/PickScore_v1`
- HPSv2：`dataset/HPSv2`（HPS_v2.pt / HPS_v2_compressed.pt / HPS_v2.1_compressed.pt）
- ImageReward：`dataset/ImageReward/ImageReward.pt`
- 统一缓存：`dataset/hf_cache`

## 4. 运行命令（评测全部 6 个集合）

```bash
# 进入环境（若已在 t2v，可跳过）
source /z_data/miniconda3/etc/profile.d/conda.sh
conda activate t2v

# 仅首次需要（若已安装可跳过）
conda install -y -c conda-forge ffmpeg=6.1.2 libiconv>=1.17

# 评测（只用本地权重与本地缓存）
python eval_janus_metrics.py \
  --device cuda \
  --cache_dir dataset/hf_cache \
  --vqa_model clip-flant5-xxl \
  --clip_model openai:ViT-L-14-336 \
  --itm_model blip2-itm \
  --hps_model hpsv2 \
  --pickscore_model pickscore-v1 \
  --imagereward_ckpt dataset/ImageReward/ImageReward.pt
```

备注：
- `clip-flant5-xxl` 体量很大，建议 40GB 显存；如需改为 `blip2-flan-t5-xxl` 版本的 VQAScore，请告知，我可按本地 `dataset/blip2-flan-t5-xxl` 做同样的路径重定向以确保不下载。
- 首次运行会读取/解压大权重（本地 IO），ITM/HPSv2 初始化较慢，属正常。

## 5. 输出结果

- 每个集合目录会生成两份结果：
  - `janus_eval_scores.csv`：每张图 6 个分数 + 末行平均分（每个指标一个平均分）
  - `janus_eval_scores.json`：按图片记录所有分数和各指标平均值
- 生成位置（与 task.md 要求一致）：
  - `/z_data/migration/syxin/janus/Janus_combined_dola_early_exit/`
  - `/z_data/migration/syxin/janus/Janus_combined_dola_no_early_exit/`
  - `/z_data/migration/syxin/janus/Janus_llama_dola_early_exit/`
  - `/z_data/migration/syxin/janus/Janus_llama_dola_no_early_exit/`
  - `/z_data/migration/syxin/janus/Janus_VIT_dola_early_exit/`
  - `/z_data/migration/syxin/janus/Janus_VIT_dola_no_早_exit/`

## 6. 已验证要点（抽样）

- ffmpeg 在 `t2v` 环境内可执行；t2v_metrics 能正常 import。
- CLIPScore（ViT-L/14-336）使用本地 HF 权重评分正常。
- ITMScore（BLIP2‑ITM）已切到本地 Transformers 权重，评分正常。
- PickScore 使用本地 `PickScore_v1`（模型与处理器）评分正常。
- ImageReward 使用本地 `ImageReward.pt` 评分正常。
- HPSv2 使用本地包与本地权重评分正常。

## 7. 可能的后续

- 若要把 VQAScore 从 `clip-flant5-xxl` 切到 `blip2-flan-t5-xxl`，需要给 LAVIS 路径做本地权重映射（与上面对 ITM 的处理类似）。如需我来补丁，请告知。
- 如需固定特定 `timm/transformers` 版本或去掉 pip 软冲突告警，我也可以加到环境清单里。


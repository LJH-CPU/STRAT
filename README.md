# STRAT

**Spatio-Temporal Relational-Attention Transformer** — a leakage-free framework
for uncertainty-aware tourist trajectory forecasting in scenic areas.

STRAT discovers scenic regions and routes with granular-ball spectral
clustering, enforces a strict no-leakage evaluation protocol, and jointly
predicts the **next region** and a **probabilistic arrival time** that is
propagated along the trajectory and calibrated at both segment and path level.

> **Data are not included in this repository.** Raw/processed trajectories are
> private and excluded via `.gitignore`. See [`docs/data.md`](docs/data.md) for
> the data availability statement. Only the source code and the curated results
> under `final_results/` are shipped.

---

## Pipeline

```
Raw GPS trajectories (data-project/cleaned_labeled_data/<scene>_cleaned.csv, not shipped)
  │
  ├─ [Clustering]  cluster/Granular_Spherical_Clustering.py + scenery_route_clustering.py
  │    · stay points → lon/lat → metres → MinMax → granular-ball split (balanced
  │      bisection) → spectral clustering → region_id
  │    · route discovery (hierarchical clustering) → route_id
  │
  ├─ [Strict no-leakage data flow]  prediction/ar_pipeline.py
  │    · 70/30 split by track → clustering/routes/statistics/normalisation are
  │      fit on the train split only
  │    · test points are mapped to the nearest train region with a KD-tree
  │
  └─ [Autoregressive forecasting]  prediction/transformer_model.py
       · causal Transformer + trajectory structure infusion (R-GCN)
       · next-region head (top-1/3) + Gaussian arrival-time head, propagated
         and calibrated at segment and path level
```

## Repository structure

| Path | Contents |
|---|---|
| `prediction/` | Main method: no-leakage pipeline, causal Transformer, baselines, evaluation |
| `cluster/` | Granular-ball / affinity / route-discovery clustering |
| `data-project/` | Leakage-free data flow, segment/region benchmarks, table/figure scripts |
| `poi/` | POI collection, projection and visualisation (AMap API) |
| `final_results/` | Curated experimental results (JSON/CSV/figures) |
| `legacy/` | Archived early exploration code (BiGRU / TCN / GRU experiments) |
| `config.py` | Single source of truth for paths and shared constants |
| `run.py` | End-to-end entry point |

## Installation

```bash
git clone https://github.com/LJH-CPU/STRAT.git
cd STRAT

python -m venv .venv && source .venv/bin/activate
# install the PyTorch build matching your platform first:
#   https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the paper. A GPU is recommended
for training but the code also runs on CPU.

## Quick start

```bash
# Full pipeline (8 scenic areas × 3 seeds: clustering → train → baselines → transfer → report)
python run.py --all

# Specific scenes
python run.py --scene 峨眉山 青城山          # Emei, Qingcheng

# Retrain only (reuse cached data, skip clustering/route discovery)
python run.py --stage model --scene 青城山

# Debug: 1 scene, 1 seed, 10 epochs, per-epoch loss
python run.py --scene 青城山 --debug

# Fast smoke test
python run.py --scene 黄龙溪 --quick

# Regenerate the report from existing result JSON
python run.py --report-only
```

`run.py` stages: `data` (build + cache) / `model` (train + evaluate) /
`baseline` (Markov / kNN / LSTM / XGBoost) / `transfer` (cross-scene) /
`report`. Select with `--stage`; control with `--backbone
transformer|gpt2|lstm`, `--seeds`, `--epochs`, `--use_struct`, `--use_prior`.

## Key modules (`prediction/`)

| Module | Role |
|---|---|
| `ar_pipeline.py` | Strict no-leakage data flow (split / clustering / mapping / routes) |
| `ar_model.py` | Training dataset + normalisation + dataloader |
| `cluster_search.py` | Grid search over δ + validation subset |
| `transformer_model.py` | Causal Transformer + R-GCN structure infusion + 3 heads |
| `transformer_baselines.py` | Statistical / linear / LSTM baselines |
| `transformer_eval.py` | Teacher-forced / rollout evaluation + DM test + Wilcoxon |
| `run_transformer_experiment.py` | Single-scene train/eval runner |
| `xgboost_baseline.py` | XGBoost segment-duration regression baseline |
| `transformer_transfer.py` | Cross-scene arrival-time transfer |
| `transformer_report.py` | Markdown report generation |

## Results

Curated results and the machine-readable summary live in
[`final_results/`](final_results/README.md) (`summary.json`, `eta_table.csv`,
figures).

Segment-level zero-shot MAE (min/500 m):

| Model | MAE |
|---|---|
| Causal Transformer | **6.66** |
| LightGBM | 6.88 |
| XGBoost | 6.90 |
| Ridge | 7.44 |
| Naismith | 7.53 |
| RandomForest / ExtraTrees | 8.02 |
| kNN | 8.24 |
| LSTM | 8.99 |

Region-level next-region accuracy: same-scene **0.861** → zero-shot **0.139**
(20/20 scenes, p = 1.9e-6).

See [`final_results/README.md`](final_results/README.md) for the full protocol
(20-scene leave-one-scene-out), calibration results and negative controls.

## Citation

If you use this code, please cite the STRAT paper. See
[`CITATION.cff`](CITATION.cff).

## License

Released under the [MIT License](LICENSE).

---

## 中文简介

**STRAT（时空关系注意力 Transformer）** 是一个用于景区轨迹不确定性感知预测的
无泄漏框架：粒球谱聚类发现区域/路线 → 严格无泄漏数据流 → 因果 Transformer 联合
预测下一区域与概率化到达时间（沿轨迹传播并在段级/路径级校准）。

- **数据不随仓库发布**：原始/处理后的轨迹涉隐私，已在 `.gitignore` 中排除；
  数据获取方式见 [`docs/data.md`](docs/data.md)，仓库仅含代码与
  `final_results/` 精选结果。
- 安装与运行见上方 *Installation* / *Quick start*。
- `legacy/` 为早期探索代码（旧 BiGRU / TCN / GRU 线）归档，论文方法以
  `prediction/` 下的 `transformer_*` 模块为准。
- 结果汇总见 [`final_results/README.md`](final_results/README.md)。

# 无泄漏全量重跑结果（final_results）

本目录汇总 **STRAT 去泄漏 + 统计口径修正** 后的全部实验数据。

- 协议：20 景区 **leave-one-scene-out**；track 级划分；scene 级统计（n=20，seed 仅随机重复）。
- 段级特征 8 维（`dist_m, up_m, down_m, mean_grade, max_grade, elev_mean, start_ele, tod_hour`），
  地形来自 SRTM DEM；已删除 `move_time_s`/`n_pts`。
- 单位：段级 = min/500m；区域时长 = min；acc/F1 为比例。
- 机器可读汇总见 `summary.json`。

---

## 文件清单

### 段级（20 景区 × 5 seed）
| 文件 | 内容 |
|---|---|
| `result_segment.json` | 统一 benchmark：模型阶梯、#3 特征组消融(12组)、#5 few-shot(R=5)、#6 matched A/B、#9 权重、#10 TOD |
| `result_segment_char.json` | 分位校准(Exp2)、few-shot 曲线、特征消融、场景身份探针(G1 修正版)，5 seed |
| `result_deep_lstm.json` | 段级 LSTM 零样本（#12 深度家族） |
| `result_deep_transformer.json` | 段级因果 Transformer 零样本 |
| `result_stat_slope.json` | #P0-2 scene 级统计：去坡度 vs full |
| `result_stat_elev.json` | #P0-2 scene 级统计：去高程 vs full |
| `result_aggregate.json` | #9 macro vs segment-weighted |
| `result_drift.json` | #4 SMD + MMD 与 few-shot gain 的相关（含偏相关） |

### 区域级
| 文件 | 内容 |
|---|---|
| `result_region_main.json` | #7 零样本（20×5）：top-1/3/5、macro-F1、majority、1/K |
| `result_region_same.json` | 同场景上界（20×5） |
| `result_region_collapse.json` | same vs zero 的 collapse 汇总（scene 级检验） |
| `result_region_kvocab.json` | #8 原型词表 K 敏感性（K/2, K, 2K；5 target × 3 seed） |
| `result_region_module.json` | #12 模块消融 + 序列长度（5 target × 5 seed） |

---

## 关键数字（详见 summary.json）

### 段级
- 零样本 MAE：XGB **6.90** / LGB **6.88** / Transformer **6.66** / LSTM **8.99** / Ridge 7.44 / RF·ET 8.02 / kNN 8.24 / Naismith 7.53
- 自训 XGB 5.41 → **保留率 78%**（原论文 87%）
- **去坡度：Δ=+0.65，18/20 景区，p=0.0014（scene 级 n=20）**；`slope_only 6.98 ≈ full 6.90`
- 去高程：Δ=−0.02，p=0.96（不显著）
- few-shot 增益：+0.09 / +0.16 / +0.30（5/10/25%）
- matched：A（等量额外源）**−0.00**；B（总量匹配）**+0.19**
- 校准（共享 XGB 零样本）：PICP90 **0.877**（名义 0.90）
- 场景身份探针：**0.528**（chance 0.05；旧 in-sample 0.749）
- drift → few-shot：SMD ρ=0.31 p=0.19；MMD ρ=0.29 p=0.22（**不显著**）

### 区域级
- 下一区域 acc：同场景 **0.861** → 零样本 **0.139**（20/20 场景，p=1.9e-6）
- macro-F1：同场景 **0.650** → 零样本 **0.065**
- 时长：同场景 17.9 → 零样本 30.1 min
- 词表 K：K/2 **0.221** / K **0.110** / 2K **0.047**（崩塌跨 K 稳健）
- 模块（零样本 acc）：full 0.131 / −RGCN 0.113 / −route 0.130 / −region 0.127 / transformer 0.138
- 序列长度：L1 0.000 / L2 0.089 / L4 0.100 / L8 0.140

---

## 结论变化速览
- ✅ 保留：迁移存在、**坡度核心**（更强）、高程场景相关、region 身份崩塌（更完整）、校准欠覆盖（变轻）
- ⚠️ 降级：few-shot 随 drift 缩放（→弱正趋势，不显著）
- ⚠️ 改变：深度 vs 树（LSTM 8.99 差，因果 Transformer 6.66 持平/略优树 6.90）
- 🔢 数字更新：保留率 87%→78%；探针 74.9%→52.8%；"100 对"→"20 景区（5 seed 重复）"

---

## P3 负控：Randomized-Identity（region 身份是否提供可迁移信息）

四路对照（零样本下一区域 acc）：

| 设定 | acc |
|---|---|
| Region + Identity（full） | 0.131 |
| Region-only（−region） | 0.127 |
| **Randomized Identity（固定随机置换，rs∈{0,1,2}）** | **0.136** |
| Plain Transformer | 0.138 |
| Fine-grained Physical（段级 XGB） | **6.90 min，保留 78%** |

→ 四者几乎相同（~0.13）：**scene-identity 信息不提供可迁移的 zero-shot 预测价值**。
结论措辞固定为：*scene-identity information does not provide transferable predictive value under zero-shot region-level evaluation*（不写"collapse 由 identity 导致"）。

文件：`result_region_randid.json`

## P4：architecture vs historical-context（段级 ETA，5 target × 5 seed，共用 target 集 i≥7）

| Model | L=1 | L=2 | L=4 | L=8 |
|---|---|---|---|---|
| Transformer | 7.99 | 7.64 | 6.98 | **6.23** |
| LSTM | 8.33 | 8.03 | 7.87 | 7.84 |
| XGB / LGB（单步表格） | 6.90 / 6.88 | — | — | — |

- **Transformer(L=1)=7.99 > XGB 6.90**：无历史时，纯架构并不占优。
- **Transformer(L=8)=6.23 < XGB**：加上历史上下文后才反超。
- LSTM 随 L 改善但趋于平台（7.84），始终不及 XGB。
- → **Transformer 的优势是 historical-context effect，且只有 Transformer 有效利用历史**；不是单纯 architecture effect。
- 公平性：所有 L 使用**同一批 target segments**（`n_test` 相同），只变可见历史长度。

文件：`result_deep_transformer_seq.json`、`result_deep_lstm_seq.json`

## P1 说明
region next-region 序列长度只报 `L∈{2,4,8}`；`L=1` 因 `Lr−1=0` 无 next-region 目标被排除（`result_region_module.json` 中的 `seqlen1` 请忽略）。段级 ETA 的 `L=1` 保留（合法 no-history 基线）。

## P2 产物
`eta_table.csv` + `figure4_fewshot_matched.png`，均由 `result_segment.json` 自动生成（`data-project/make_eta_table.py`）。


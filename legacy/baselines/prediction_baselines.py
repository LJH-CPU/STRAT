"""
预测基线对比：Region-Median / LightGBM / MLP / BiLSTM / Transformer

与 STRAT BiGRU 使用完全相同的训练/验证/测试集和评估指标，保证公平对比。
"""

import os
import sys
import argparse
import json
import time
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'prediction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ablation'))

from prediction.data_utils import (
    extract_first_occurrence_sequences,
    build_region_metadata,
    split_by_route,
    GPS_SEG_DIM,
)
from prediction.sequence_model import SequencePredictor, create_dataloader
from run_prediction_ablation import (
    train_epoch, evaluate_model, predict_gru_segments,
    compute_all_metrics, segments_to_cumulative, relative_l1_loss,
)
from prediction.baseline import build_knn_baseline, predict_knn


# ============================================================
# Region-Median Baseline
# ============================================================
def train_region_median(train_records):
    median_map = {}
    for rec in train_records:
        for i, dur in enumerate(rec['segment_durations']):
            rid = rec['seq'][i]
            median_map.setdefault(rid, []).append(dur)
    for rid in median_map:
        median_map[rid] = float(np.median(median_map[rid]))
    return median_map


def predict_region_median(median_map, test_records):
    results = []
    for rec in test_records:
        preds = [median_map.get(rec['seq'][i], 600) for i in range(len(rec['segment_durations']))]
        results.append({
            'track_id': rec['track_id'],
            'route_id': rec['route_id'],
            'pred_segments': preds,
            'true_segments': rec['segment_durations'],
            'n_segs': len(preds),
        })
    return results


# ============================================================
# Flat (非序列) Baseline 模型
# ============================================================
def flatten_records(records, region_meta, num_regions):
    X_list = []
    y_list = []
    for rec in records:
        n_segs = len(rec['segment_durations'])
        gps_list = rec.get('gps_seg_features', [])
        hour = rec.get('start_time_of_day', 12.0)
        for i in range(n_segs):
            rid = rec['seq'][i]
            meta = region_meta.get(rid, {})
            role_id = meta.get('role_id', 0)
            lat = meta.get('lat', 0.0)
            lon = meta.get('lon', 0.0)
            elev = meta.get('elev_mean', 0.0)
            gps = gps_list[i] if i < len(gps_list) else [0.0] * GPS_SEG_DIM
            features = [float(rid), lat, lon, elev]
            features.extend(gps)
            features.append(np.sin(2 * np.pi * hour / 24))
            features.append(np.cos(2 * np.pi * hour / 24))
            features.append(float(role_id))
            features.append(float(i) / 10.0)
            X_list.append(features)
            y_list.append(float(rec['segment_durations'][i]))
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def run_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    try:
        import lightgbm as lgb
    except ImportError:
        print("  LightGBM not installed, skipping")
        return None

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'regression', 'metric': 'mae',
        'boosting_type': 'gbdt', 'num_leaves': 31,
        'learning_rate': 0.05, 'feature_fraction': 0.9,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'verbose': -1, 'seed': 42,
    }

    t0 = time.time()
    model = lgb.train(params, train_data, num_boost_round=500,
                       valid_sets=[val_data],
                       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    preds = model.predict(X_test, num_iteration=model.best_iteration)
    elapsed = time.time() - t0
    return preds, elapsed


# ============================================================
# 深度学习基线：MLP
# ============================================================
class MLPPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, region_ids, geo_feats, gps_feats, time_feats,
                role_ids, lengths):
        B, L, _ = geo_feats.shape
        device = geo_feats.device
        feats_list = []
        seg_map = []
        for b in range(B):
            seq_len = lengths[b].item()
            for i in range(1, seq_len):
                f = torch.cat([
                    region_ids[b, i].float().unsqueeze(0),
                    geo_feats[b, i],
                    gps_feats[b, i],
                    time_feats[b, i],
                    role_ids[b, i].float().unsqueeze(0),
                    torch.tensor([float(i) / 10.0], device=device),
                ], dim=0)
                feats_list.append(f)
                seg_map.append((b, i))
        if not feats_list:
            return torch.zeros(B, L, 1, device=device)
        flat_t = torch.stack(feats_list, dim=0)
        preds_flat = self.net(flat_t)
        out = torch.zeros(B, L, 1, device=device)
        for k, (b, i) in enumerate(seg_map):
            out[b, i, 0] = preds_flat[k, 0]
        return out


# ============================================================
# 深度学习基线：BiLSTM
# ============================================================
class BiLSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2,
                             batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, region_ids, geo_feats, gps_feats, time_feats,
                role_ids, lengths):
        B, L, _ = geo_feats.shape
        device = geo_feats.device
        pos_t = torch.arange(L, device=device).float().unsqueeze(0).unsqueeze(-1).expand(B, -1, -1) / 10.0
        rid_f = region_ids.float().unsqueeze(-1)
        role_f = role_ids.float().unsqueeze(-1)
        x = torch.cat([rid_f, geo_feats, gps_feats, time_feats, role_f, pos_t], dim=-1)
        x = self.proj(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=L)
        return self.head(out)


# ============================================================
# 深度学习基线：Transformer
# ============================================================
class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=1, dim_feedforward=16,
            dropout=0.5, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, region_ids, geo_feats, gps_feats, time_feats,
                role_ids, lengths):
        B, L, _ = geo_feats.shape
        device = geo_feats.device
        pos_t = torch.arange(L, device=device).float().unsqueeze(0).unsqueeze(-1).expand(B, -1, -1) / 10.0
        rid_f = region_ids.float().unsqueeze(-1)
        role_f = role_ids.float().unsqueeze(-1)
        x = torch.cat([rid_f, geo_feats, gps_feats, time_feats, role_f, pos_t], dim=-1)
        x = self.proj(x)
        mask = torch.arange(L, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.head(x)


# ============================================================
# 训练 & 评估 (统一接口)
# ============================================================
def train_deep_baseline(model, train_loader, val_loader, val_dataset, device, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20)
    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        _, val_mae = evaluate_model(model, val_loader, val_dataset, device)
        scheduler.step(val_mae)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            break

    model.load_state_dict(best_state)
    return model


# ── 公共景区解析 ─────────────────────────────────────
CLUSTER_DIR = os.path.join(os.path.dirname(__file__), '..', 'cluster', 'output')
PAPER_SCENES = {"青城山", "峨眉山", "武侯祠博物馆"}


def _resolve_scenes(scene_name=None, scene_all=False, scene_paper=False, min_tracks=30):
    """解析要处理的景区列表（从 cluster/output 找聚类结果）。"""
    if scene_name:
        for ext in ['_strat.csv', '_clustered.csv']:
            p = os.path.join(CLUSTER_DIR, f"{scene_name}{ext}")
            if os.path.exists(p):
                return [(scene_name, p)]
        # fallback to cleaned
        cleaned = os.path.join(os.path.dirname(__file__), '..', 'data-project',
                               'cleaned_labeled_data', f"{scene_name}_cleaned.csv")
        if os.path.exists(cleaned):
            return [(scene_name, cleaned)]
        print(f"错误: 找不到 {scene_name} 的数据")
        return []
    csv_files = sorted(glob.glob(os.path.join(CLUSTER_DIR, "*_strat.csv")))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(CLUSTER_DIR, "*_clustered.csv")))
    if not csv_files:
        print("错误: cluster/output 无聚类结果，请先运行 generate_clustered_csvs.py")
        return []
    candidates = []
    for f in csv_files:
        name = os.path.basename(f).replace("_strat.csv", "").replace("_clustered.csv", "")
        if scene_paper and name not in PAPER_SCENES:
            continue
        df_tmp = pd.read_csv(f, usecols=["trackId"])
        n = df_tmp["trackId"].nunique()
        del df_tmp
        if n >= min_tracks:
            candidates.append((name, f))
    return candidates


def load_and_run_one(scene_name, csv_path, args):
    """对单个景区加载数据并跑所有预测基线。"""
    print(f"\n{'#'*60}")
    print(f"# [预测基线] {scene_name}")
    print(f"{'#'*60}")

    df = pd.read_csv(csv_path)
    n_tracks = df['trackId'].nunique()
    print(f"  Points: {len(df):,}, Tracks: {n_tracks}")

    records = extract_first_occurrence_sequences(df)
    if len(records) < 10:
        print(f"  [跳过] 有效轨迹不足 ({len(records)})")
        return None

    region_meta = build_region_metadata(df)
    num_regions = len(region_meta)
    num_roles = max(m['role_id'] for m in region_meta.values()) + 1

    train_records, val_records, test_records = split_by_route(records, seed=args.seed)
    print(f"  Train: {len(train_records)}, Val: {len(val_records)}, "
          f"Test: {len(test_records)}, Regions: {num_regions}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cpu')

    max_len = 8
    train_loader, train_dataset = create_dataloader(
        train_records, region_meta, num_regions, args.batch_size, shuffle=True, max_len=max_len)
    val_loader, val_dataset = create_dataloader(
        val_records, region_meta, num_regions, args.batch_size, shuffle=False, max_len=max_len)
    test_loader, test_dataset = create_dataloader(
        test_records, region_meta, num_regions, args.batch_size, shuffle=False, max_len=max_len)

    flat_input_dim = 1 + 3 + GPS_SEG_DIM + 2 + 1 + 1
    route_stats = build_knn_baseline(train_records)

    scene_results = {}

    # 1. Region-Median
    print(f"\n  [Region-Median]")
    t0 = time.time()
    median_map = train_region_median(train_records)
    med_results = predict_region_median(median_map, test_records)
    metrics = compute_all_metrics(med_results, test_records, route_stats)
    metrics['time_s'] = time.time() - t0
    scene_results['region_median'] = metrics
    print(f"    Seg MAE: {metrics['seg_mae']:.0f}s")

    # 2. LightGBM
    if 'lightgbm' in args.models:
        print(f"  [LightGBM]")
        try:
            X_train, y_train = flatten_records(train_records, region_meta, num_regions)
            X_val, y_val = flatten_records(val_records, region_meta, num_regions)
            X_test, y_test = flatten_records(test_records, region_meta, num_regions)
            result = run_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test)
            if result is not None:
                lgb_preds, lgb_time = result
                flat_idx = 0
                lgb_results = []
                for rec in test_records:
                    n_segs = len(rec['segment_durations'])
                    pred_segs = lgb_preds[flat_idx:flat_idx + n_segs].tolist()
                    flat_idx += n_segs
                    lgb_results.append({
                        'track_id': rec['track_id'],
                        'route_id': rec['route_id'],
                        'pred_segments': pred_segs,
                        'true_segments': rec['segment_durations'],
                        'n_segs': n_segs,
                    })
                metrics = compute_all_metrics(lgb_results, test_records, route_stats)
                metrics['time_s'] = lgb_time
                scene_results['lightgbm'] = metrics
                print(f"    Seg MAE: {metrics['seg_mae']:.0f}s")
        except Exception as e:
            print(f"    LightGBM failed: {e}")

    # 3. BiLSTM
    if 'bilstm' in args.models:
        print(f"  [BiLSTM]")
        bilstm = BiLSTMPredictor(flat_input_dim, args.hidden_dim).to(device)
        t0 = time.time()
        bilstm = train_deep_baseline(bilstm, train_loader, val_loader, val_dataset, device, args)
        lstm_results = predict_gru_segments(bilstm, test_loader, test_dataset, device)
        metrics = compute_all_metrics(lstm_results, test_records, route_stats)
        metrics['time_s'] = time.time() - t0
        scene_results['bilstm'] = metrics
        print(f"    Seg MAE: {metrics['seg_mae']:.0f}s")

    return scene_results


def main():
    parser = argparse.ArgumentParser(description='Prediction Baselines Comparison')
    parser.add_argument('--scene', type=str, default=None, help='景区名称')
    parser.add_argument('--scene_all', action='store_true', help='对所有景区运行')
    parser.add_argument('--scene_paper', action='store_true',
                        help='只跑论文3个景区')
    parser.add_argument('--min_tracks', type=int, default=30)
    parser.add_argument('--csv', type=str, default=None, help='(兼容) 直接指定 CSV')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='baselines/output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--models', type=str, nargs='*',
                        default=['median', 'lightgbm', 'mlp', 'bilstm', 'transformer'])
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.csv:
        scenes = [("景区", args.csv)]
    else:
        scenes = _resolve_scenes(args.scene, args.scene_all, args.scene_paper, args.min_tracks)

    if not scenes:
        print("没有符合条件的景区。")
        return

    print("=" * 70)
    print("预测基线对比实验")
    print("=" * 70)
    print(f"将处理 {len(scenes)} 个景区")

    all_results = {}
    for scene_name, csv_path in scenes:
        r = load_and_run_one(scene_name, csv_path, args)
        if r is not None:
            all_results[scene_name] = r

    out_path = os.path.join(args.output_dir, 'prediction_baselines.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n所有结果已保存到 {out_path}")


if __name__ == '__main__':
    main()
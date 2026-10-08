"""RealTraj 式自监督预训练试点：masked-region 预训练编码器 → 微调下一区域。

预训练任务：随机掩盖轨迹中间位置，因果背板从其过去上下文重建被掩区域。
微调：用预训练 projector+backbone 初始化 TrajectoryPredictor，正常监督训练。
对比：从零训练（llm_main_transformer 的每场景 STRAT）。
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import CLEANED_DIR
from ar_model import create_dataloader, POI_DIM, GPS_SEG_DIM
from transformer_model import StepProjector, TransformerBackbone, TrajectoryPredictor
from transformer_eval import teacher_forced_eval, rollout_eval
import run_transformer_experiment as rte


def load_bundle(name, seed):
    cp = os.path.join(os.path.dirname(__file__), 'output', 'cache', f'{name}_{seed}_auto.pkl')
    if os.path.exists(cp):
        with open(cp, 'rb') as f:
            return pickle.load(f)
    import argparse as _ap
    a = _ap.Namespace(seed=seed, k_override=None, n_poi_regions=None, n_workers=8, max_len=8)
    b = rte.build_bundle(name, str(CLEANED_DIR / f'{name}_cleaned.csv'), seed, a)
    with open(cp, 'wb') as f:
        pickle.dump(b, f)
    return b


class PretrainDS(Dataset):
    """masked-region 预训练样本。region_ids 被掩位置置为 MASK(=V)。"""

    def __init__(self, records, region_meta, norm, V, mask_ratio=0.2, max_len=8, seed=0):
        self.items = []
        rng = np.random.RandomState(seed)
        poi_names = ['餐饮', '休闲', '住宿', '风景名胜', '科教文化', '公共设施']
        for r in records:
            seq = r['seq'][:max_len]
            L = len(seq)
            if L < 3:
                continue
            ids = list(seq) + [0] * (max_len - L)
            maskpos = []
            for i in range(1, L - 1):
                if rng.rand() < mask_ratio:
                    maskpos.append(i)
            if not maskpos:
                continue
            for i in maskpos:
                ids[i] = V  # MASK
            role = np.zeros(max_len, dtype=np.int64)
            geo = np.zeros((max_len, 3), dtype=np.float32)
            poi = np.zeros((max_len, POI_DIM), dtype=np.float32)
            tm = np.zeros((max_len, 2), dtype=np.float32)
            gps = np.zeros((max_len, GPS_SEG_DIM), dtype=np.float32)
            hour = r['start_time_of_day']
            tm[:, 0] = np.sin(2 * np.pi * hour / 24.0)
            tm[:, 1] = np.cos(2 * np.pi * hour / 24.0)
            mask_set = set(int(x) for x in maskpos)
            for i, rid in enumerate(seq):
                if i in mask_set:
                    continue  # 掩码位置不得含被掩码区域自身特征（geo/poi/role）
                m = region_meta.get(rid, {})
                role[i] = m.get('role_id', 0)
                geo[i, 0] = (m.get('lat', 0) - norm['lat_mean']) / norm['lat_std']
                geo[i, 1] = (m.get('lon', 0) - norm['lon_mean']) / norm['lon_std']
                geo[i, 2] = (m.get('elev_mean', 0) - norm['elev_mean']) / norm['elev_std']
                pv = np.zeros(POI_DIM)
                for k, v in m.get('poi_profile', {}).items():
                    if k in poi_names:
                        pv[poi_names.index(k)] = v
                poi[i] = (pv - norm['poi_mean']) / (norm['poi_std'] + 1e-9)
            gf = r['gps_seg_features']
            for i in range(min(L - 1, max_len)):
                if i in mask_set:
                    continue  # 掩码位置不得含后继段信息（A5）
                arr = np.array(gf[i], dtype=np.float32)
                gps[i] = (arr - norm['gps_mean']) / (norm['gps_std'] + 1e-9)
            tgt = np.full(max_len, -1, dtype=np.int64)
            for i in maskpos:
                tgt[i] = seq[i]
            msk = np.zeros(max_len, dtype=bool)
            for i in maskpos:
                msk[i] = True
            self.items.append((np.array(ids, dtype=np.int64), role, geo, poi, tm, gps, tgt, msk))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ids, role, geo, poi, tm, gps, tgt, msk = self.items[idx]
        pos = np.arange(8, dtype=np.int64)
        return ids, role, geo, poi, tm, gps, tgt, msk, pos


def pretrain(bundle, device, epochs=30, bs=128, lr=1e-3):
    region_meta, norm = bundle['region_meta'], bundle['norm']
    V = bundle['V']
    ds = PretrainDS(bundle['tr_records'], region_meta, norm, V)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=lambda b: tuple(torch.tensor(np.stack(x)) for x in zip(*b)))
    model = nn.Module()
    model.projector = StepProjector(V + 1, 1, bundle['geo_dim'], role_embed_dim=8, d_model=64)
    model.backbone = TransformerBackbone(64)
    model.head = nn.Linear(64, V)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for ids, role, geo, poi, tm, gps, tgt, msk, pos in dl:
            ids = ids.to(device); role = role.to(device); geo = geo.to(device)
            poi = poi.to(device); tm = tm.to(device); gps = gps.to(device)
            tgt = tgt.to(device); msk = msk.to(device); pos = pos.to(device)
            # gps/poi/tm/geo 拼接按 StepProjector 输入格式（geo=3+POI? StepProjector 用 geo_dim 含 poi）
            # StepProjector 输入: region_ids, geo(3+POI), gps, time_feat(2), role_ids, positions
            gfull = torch.cat([geo, poi], -1)
            h0 = model.projector(ids, gfull, gps, tm, role, pos)
            z = model.backbone(h0)
            lg = model.head(z)
            loss = F.cross_entropy(lg[msk], tgt[msk])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if ep % 10 == 0:
            print(f'    pretrain ep{ep} loss={tot/max(len(dl),1):.3f}')
    return model


def fine_tune(bundle, pretrained, device, epochs=80, seed=42):
    region_meta, norm = bundle['region_meta'], bundle['norm']
    tr, va, te = bundle['tr_records'], bundle['va_records'], bundle['te_records']
    V, R, geo_dim = bundle['V'], bundle['R'], bundle['geo_dim']
    train_dl, _ = create_dataloader(tr, region_meta, norm, batch_size=16, shuffle=True, max_len=8)
    val_dl, _ = create_dataloader(va, region_meta, norm, batch_size=16, shuffle=False, max_len=8)
    test_dl, _ = create_dataloader(te, region_meta, norm, batch_size=16, shuffle=False, max_len=8)
    model = TrajectoryPredictor(num_regions=V, num_routes=R, geo_dim=geo_dim, pair_mean=bundle['pair_mean'],
                                global_mean=bundle['gmean'], seg_p95=norm['seg_p95'], backbone='transformer',
                                d_model=64, use_struct=True, use_prior=True, time_feats='full').to(device)
    with torch.no_grad():
        pd_ = pretrained.projector.state_dict()
        sd = model.projector.state_dict()
        # region_embed: 预训练 V+1 行 → 取前 V 行
        for k in sd:
            if k in pd_:
                if k == 'region_embed.weight':
                    sd[k] = pd_[k][:V]
                elif sd[k].shape == pd_[k].shape:
                    sd[k] = pd_[k]
        model.projector.load_state_dict(sd)
        model.backbone.load_state_dict(pretrained.backbone.state_dict())
    model = rte.train_model(model, train_dl, val_dl, device, epochs=epochs, seed=seed, loss='mae')
    tf = teacher_forced_eval(model, test_dl, device, norm['seg_p95'])
    roll = rollout_eval(model, te, region_meta, norm, device)
    return tf['acc'], tf['acc3'], roll['acc'][0] / max(roll['acc'][1], 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', default=['青城山', '熊猫基地', '峨眉山'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pt_epochs', type=int, default=30)
    p.add_argument('--ft_epochs', type=int, default=80)
    p.add_argument('--out', default='prediction/output/llm_pretrain.json')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_json = json.load(open(os.path.join(os.path.dirname(__file__), 'output', 'llm_main_transformer.json'), encoding='utf-8'))
    def base(name, seed):
        for k, v in main_json.items():
            if v.get('status') == 'ok' and k.startswith(f'{name}|') and f'seed{seed}|' in k:
                return v['model']
        return None

    results = {}
    for name in args.scenes:
        b = load_bundle(name, args.seed)
        print(f'[{name}|{args.seed}] pretraining...')
        pt = pretrain(b, device, args.pt_epochs)
        tf_a, tf3, roll_a = fine_tune(b, pt, device, args.ft_epochs, args.seed)
        m = base(name, args.seed)
        print(f"[{name}|{args.seed}] pretrain: tf={tf_a:.3f} roll={roll_a:.3f} | from-scratch: tf={m['tf_acc']:.3f} roll={m['roll_acc']:.3f}")
        results[name] = {'pt_tf': tf_a, 'pt_tf3': tf3, 'pt_roll': roll_a,
                         'scratch_tf': m['tf_acc'], 'scratch_roll': m['roll_acc']}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('已保存:', args.out)


if __name__ == '__main__':
    main()

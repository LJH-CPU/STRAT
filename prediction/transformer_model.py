"""
轨迹↔Transformer/LLM 接口预测模型（STRAT 预测模块 v2）。

架构：
  step 特征(region id / geo+POI / role / 时刻 / 段特征 / 路线特征)
    → StepProjector（对齐层，投影到背板维度）
    → RGCNAggregator（轨迹结构注入：时序邻接/角色同步/POI相似/时段相似）
    → Backbone（可插拔：冻结GPT-2前4层 / 从零Transformer / LSTM）
    → 三头：
        ① next-region 头（因果注意力决定"去哪"，softmax over 区域）
        ② 时长头（特征回归"何时到"：区域对先验 + 残差）
        ③ route 头（辅助侧任务）

推理：自回归 rollout（预测的下一区域回灌继续）。

依赖：复用 ar_model.ARDataset / build_normalizers / create_dataloader / rel_l1。
"""

import torch
import torch.nn as nn
import numpy as np
from ar_model import POI_DIM, GPS_SEG_DIM


# ── 结构注入：R-GCN ──────────────────────────────────────
def _relation_edges(region_ids, role_ids, geo_poi, time_bins, edge_mask=(True, True, True, True),
                    causal=True):
    """
    返回每个关系类型的边列表（index 对），按样本独立计算。
    关系：0=时序邻接 1=角色同步 2=POI top-K 相似 3=时段同步
    edge_mask: 逐类型开关（用于消融）。
    causal=True: 只发前向边 (j→k, j<k)，节点 k 只聚合过去节点，杜绝未来信息经图泄漏
    （注意力因果掩码无法抹去已注入输入的未来信息）。
    causal=False: 原双向全连接（仅用于泄漏量化对照）。
    """
    L = len(region_ids)
    rels = [[] for _ in range(4)]
    # 0) 时序邻接
    if edge_mask[0]:
        for i in range(L - 1):
            rels[0].append((i, i + 1))
            if not causal:
                rels[0].append((i + 1, i))
    # 1) 角色同步（同 role 的 step 全连）
    if edge_mask[1]:
        for i in range(L):
            for j in range(i + 1, L):
                if role_ids[i] == role_ids[j]:
                    rels[1].append((i, j))
                    if not causal:
                        rels[1].append((j, i))
    # 3) 时段同步（同 4h 时段 bin）
    if edge_mask[3]:
        for i in range(L):
            for j in range(i + 1, L):
                if time_bins[i] == time_bins[j]:
                    rels[3].append((i, j))
                    if not causal:
                        rels[3].append((j, i))
    # 2) POI top-K 相似（动态边，K=2，有向）
    if edge_mask[2]:
        K = min(2, L - 1)
        if K >= 1:
            for j in range(L):
                if causal:
                    # 只在过去节点 (i<j) 中取 top-K 相似
                    past = list(range(j))
                    if len(past) < 1:
                        continue
                    sims = []
                    for i in past:
                        a, b = geo_poi[i], geo_poi[j]
                        denom = (torch.norm(a) * torch.norm(b)).clamp(min=1e-8)
                        sims.append((float((a @ b) / denom), i))
                    sims.sort(key=lambda x: -x[0])
                    for _, i in sims[:K]:
                        rels[2].append((i, j))
                else:
                    sims = []
                    for i in range(L):
                        if i == j:
                            continue
                        a, b = geo_poi[i], geo_poi[j]
                        denom = (torch.norm(a) * torch.norm(b)).clamp(min=1e-8)
                        sims.append((float((a @ b) / denom), i))
                    sims.sort(key=lambda x: -x[0])
                    for _, i in sims[:K]:
                        rels[2].append((i, j))
    return rels


def _relation_edges_batch(role_ids, geo_poi, time_bins, edge_mask=(True, True, True, True), causal=True):
    """批量化关系边：节点索引 = b*L + i。返回 (rel_src, rel_dst) 两个 list[Tensor]。
    语义与 `_relation_edges` 逐样本一致，但整体向量化（避免逐样本 Python 循环/核启动开销）。"""
    B, L = role_ids.shape
    device = role_ids.device
    empty = torch.empty(0, dtype=torch.long, device=device)

    def _append_sync(keys):
        eq = (keys.unsqueeze(2) == keys.unsqueeze(1))                     # (B,L,L) key_i==key_j
        tri = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        idx = (eq & tri.view(1, L, L)).nonzero(as_tuple=False)           # (E,3): b,i,j
        s = idx[:, 0] * L + idx[:, 1]
        d = idx[:, 0] * L + idx[:, 2]
        if not causal:
            s, d = torch.cat([s, d]), torch.cat([d, s])
        return s, d

    srcs, dsts = [], []
    # 0) 时序邻接
    if edge_mask[0] and L >= 2:
        b = torch.arange(B, device=device).view(B, 1)
        i = torch.arange(L - 1, device=device).view(1, L - 1)
        s = (b * L + i).reshape(-1); d = (b * L + i + 1).reshape(-1)
        if not causal:
            s, d = torch.cat([s, d]), torch.cat([d, s])
        srcs.append(s); dsts.append(d)
    else:
        srcs.append(empty); dsts.append(empty)
    # 1) 角色同步
    if edge_mask[1] and L >= 2:
        s, d = _append_sync(role_ids); srcs.append(s); dsts.append(d)
    else:
        srcs.append(empty); dsts.append(empty)
    # 2) POI top-K 相似（动态边，K=2）—— 必须放在 index 2，与 W_r 对齐
    K = min(2, L - 1)
    if edge_mask[2] and K >= 1:
        xn = geo_poi / geo_poi.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sim = torch.bmm(xn, xn.transpose(1, 2))                          # (B,L,L) sim[b,i,j]
        if causal:
            future = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=0)  # i>=j
            sim = sim.masked_fill(future.view(1, L, L), float("-inf"))
        else:
            sim = sim.masked_fill(torch.eye(L, device=device, dtype=torch.bool).view(1, L, L), float("-inf"))
        vals, idx = sim.topk(K, dim=1)                                   # (B,K,L) 每个 j 取 top-K 的 i
        b = torch.arange(B, device=device).view(B, 1, 1)
        j = torch.arange(L, device=device).view(1, 1, L)
        valid = torch.isfinite(vals)
        srcs.append((b * L + idx)[valid]); dsts.append((b * L + j.expand_as(idx))[valid])
    else:
        srcs.append(empty); dsts.append(empty)
    # 3) 时段同步
    if edge_mask[3] and L >= 2:
        s, d = _append_sync(time_bins); srcs.append(s); dsts.append(d)
    else:
        srcs.append(empty); dsts.append(empty)
    return srcs, dsts


class RGCNCell(nn.Module):
    """单层 R-GCN：对每个关系用独立权重聚合（批量化）。"""

    def __init__(self, d, n_rel, dropout=0.1):
        super().__init__()
        self.W_self = nn.Linear(d, d)
        self.W_r = nn.ModuleList([nn.Linear(d, d) for _ in range(n_rel)])
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, h, rel_src, rel_dst):
        """h: (N,d)，N=batch*L；rel_src/rel_dst: list[Tensor]。"""
        out = self.W_self(h)
        for r in range(len(rel_src)):
            src, dst = rel_src[r], rel_dst[r]
            if src.numel() == 0:
                continue
            msg = self.W_r[r](h[src])
            agg = torch.zeros_like(h)
            deg = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
            agg.index_add_(0, dst, msg)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
            agg = agg / deg.clamp(min=1.0).unsqueeze(-1)
            out = out + self.drop(agg)
        return self.act(out)


class RGCNAggregator(nn.Module):
    """HPRG 结构注入：R-GCN 生成结构感知表示 + 门控融合。"""

    def __init__(self, d, n_rel=4, layers=2, dropout=0.1, edge_mask=(True, True, True, True),
                 causal=True):
        super().__init__()
        self.cells = nn.ModuleList([RGCNCell(d, n_rel, dropout) for _ in range(layers)])
        self.gate = nn.Parameter(torch.zeros(1))
        self.edge_mask = tuple(bool(x) for x in edge_mask)
        self.causal = causal

    def _time_bins(self, time_feat):
        hour = torch.atan2(time_feat[..., 1], time_feat[..., 0]).fmod(2 * np.pi)
        hour = (hour / (2 * np.pi) * 24) % 24
        return (hour // 4).long()

    def forward(self, h0, region_ids, role_ids, geo_poi, time_feat):
        """h0:(B,L,d)；geo_poi:(B,L,poi_dim)。返回结构注入后的 (B,L,d)。"""
        B, L, d = h0.shape
        tb = self._time_bins(time_feat)
        rel_src, rel_dst = _relation_edges_batch(role_ids[:, :L], geo_poi[:, :L], tb[:, :L],
                                                 self.edge_mask, self.causal)
        h = h0.reshape(B * L, d)
        for cell in self.cells:
            h = cell(h, rel_src, rel_dst)
        g = torch.sigmoid(self.gate)
        h = h.reshape(B, L, d)
        return g * h + (1 - g) * h0


class GATAggregator(nn.Module):
    """GAT：对时序邻接（i-1,i,i+1）用共享注意力聚合（对照 R-GCN 的关系特异权重）。"""

    def __init__(self, d, layers=2, dropout=0.1):
        super().__init__()
        self.W = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.att = nn.ModuleList([nn.Linear(2 * d, 1) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h0):
        B, L, d = h0.shape
        outs = []
        for b in range(B):
            h = h0[b]
            for W, att in zip(self.W, self.att):
                nxt = []
                for i in range(L):
                    idxs = [i]
                    if i > 0:
                        idxs.append(i - 1)
                    nbr = h[idxs]
                    self_vec = h[i:i + 1].expand(len(idxs), -1)
                    scores = att(torch.cat([self_vec, nbr], dim=-1))  # (k,1)
                    a = torch.softmax(scores, dim=0)
                    agg = (a * nbr).sum(0)
                    nxt.append(agg)
                h = self.drop(nn.functional.gelu(W(torch.stack(nxt)))) + h
            g = torch.sigmoid(self.gate)
            outs.append(g * h + (1 - g) * h0[b])
        return torch.stack(outs)


class MLPAggregator(nn.Module):
    """MLP：无图结构的逐步非线性（对照：结构/边是否起作用）。"""

    def __init__(self, d, layers=2, dropout=0.1):
        super().__init__()
        self.W = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, h0):
        h = h0
        for W in self.W:
            h = self.drop(nn.functional.gelu(W(h))) + h
        return h


# ── 对齐层：step 特征 → 背板维度 ─────────────────────────
class StepProjector(nn.Module):
    def __init__(self, num_regions, num_routes, geo_dim, role_embed_dim,
                 region_embed_dim=32, d_model=256, use_region_id=True,
                 randomize_region_id=False, rand_seed=0):
        super().__init__()
        self.use_region_id = use_region_id
        self.randomize_region_id = randomize_region_id
        self.region_embed = nn.Embedding(num_regions, region_embed_dim, padding_idx=-1)
        self.role_embed = nn.Embedding(16, role_embed_dim)
        self.pos_embed = nn.Embedding(64, 8)
        in_dim = (region_embed_dim + geo_dim + GPS_SEG_DIM + 2 + role_embed_dim + 8)
        self.proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        if randomize_region_id:
            g = torch.Generator().manual_seed(int(rand_seed))
            self.register_buffer('region_perm', torch.randperm(num_regions, generator=g))
        else:
            self.region_perm = None

    def forward(self, region_ids, geo, gps, time_feat, role_ids, positions):
        B, L, _ = geo.shape
        rid = region_ids.clamp(min=0)
        if not self.use_region_id:
            re = torch.zeros(B, L, self.region_embed.embedding_dim, device=geo.device)
        elif self.randomize_region_id and self.region_perm is not None:
            re = self.region_embed(self.region_perm[rid])   # 固定随机置换身份（负控）
        else:
            re = self.region_embed(rid)
        ro = self.role_embed(role_ids.clamp(min=0))
        pe = self.pos_embed(positions)
        x = torch.cat([re, geo, gps, time_feat, ro, pe], dim=-1)
        return self.proj(x)


# ── 背板（可插拔）────────────────────────────────────────
class GPT2FrozenBackbone(nn.Module):
    """冻结 GPT-2 前 4 层（GraFT/GPT4TS 配方）：attention/FFN 冻结，LN+wpe 可训。"""

    def __init__(self, d_model, ckpt_dir=None, n_layer=4):
        super().__init__()
        from transformers import GPT2Config, GPT2Model
        if ckpt_dir and ckpt_dir != 'scratch':
            cfg = GPT2Config.from_pretrained(ckpt_dir)
            model = GPT2Model(cfg)
            sd = torch.load(f'{ckpt_dir}/pytorch_model.bin', map_location='cpu')
            model.load_state_dict({k: v for k, v in sd.items() if k in model.state_dict()}, strict=False)
        else:
            model = GPT2Model(GPT2Config(n_layer=n_layer, n_embd=d_model, n_head=8, n_positions=64, vocab_size=128))
        model.h = nn.ModuleList(list(model.h)[:n_layer])
        self.model = model
        self.d_model = self.model.config.n_embd
        # 冻结 attention 与 FFN，保留 LN + wpe 可训
        for name, p in self.model.named_parameters():
            if any(s in name for s in ['attn', 'mlp', 'wte']):
                p.requires_grad = False
            else:
                p.requires_grad = True

    def forward(self, h, mask=None):
        return self.model(inputs_embeds=h, attention_mask=mask).last_hidden_state


class TransformerBackbone(nn.Module):
    """从零因果 Transformer（同尺寸对照，纯 torch）。"""

    def __init__(self, d_model=64, layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.register_buffer('_tril', torch.tril(torch.ones(64, 64)).bool())

    def forward(self, h, mask=None):
        L = h.shape[1]
        # torch: True=被屏蔽(不能关注未来)，故取反 tril
        causal = ~self._tril[:L, :L]
        return self.enc(h, mask=causal, is_causal=False)


class LSTMBackbone(nn.Module):
    """单向 LSTM（对照）。"""

    def __init__(self, d_model=256, layers=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.lstm = nn.LSTM(d_model, d_model, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)

    def forward(self, h, mask=None):
        return self.lstm(h)[0]


# ── 主模型 ───────────────────────────────────────────────
class TrajectoryPredictor(nn.Module):
    def __init__(self, num_regions, num_routes, geo_dim, pair_mean, global_mean, seg_p95,
                 backbone='transformer', gpt2_ckpt=None, d_model=64,
                 use_struct=True, use_prior=True, hidden=32, dropout=0.1,
                 time_feats='full', prob=False, aggregator='rgcn',
                 edge_mask=(True, True, True, True),
                 interval_bins=None, causal_edges=True, use_region_id=True,
                 randomize_region_id=False, rand_seed=0):
        super().__init__()
        self.V = num_regions
        self.use_struct = use_struct
        self.use_prior = use_prior
        self.time_feats = time_feats
        self.seg_p95 = seg_p95
        # 先验（秒 → seg_p95 归一化）
        pm = np.array(pair_mean, dtype=np.float32)  # (V,V) 秒
        pm[pm <= 0] = float(global_mean)
        self.register_buffer('prior_table', torch.tensor(pm / max(seg_p95, 1e-6), dtype=torch.float32))

        if backbone == 'gpt2':
            self.backbone = GPT2FrozenBackbone(d_model, ckpt_dir=gpt2_ckpt)
        elif backbone == 'transformer':
            self.backbone = TransformerBackbone(d_model)
        else:
            self.backbone = LSTMBackbone(d_model)
        bdim = self.backbone.d_model

        self.projector = StepProjector(num_regions, num_routes, geo_dim,
                                       role_embed_dim=8, d_model=bdim,
                                       use_region_id=use_region_id,
                                       randomize_region_id=randomize_region_id,
                                       rand_seed=rand_seed)
        # 时间间隔嵌入（DSMR/GeoChronos 思路）：步 i 与 i-1 的到达间隔 → 分桶嵌入
        self.interval_embed = None
        if interval_bins is not None and len(interval_bins) > 0:
            self.register_buffer('interval_bins',
                                 torch.tensor(np.asarray(interval_bins, dtype=np.float32)))
            self.interval_embed = nn.Embedding(len(interval_bins) + 1, bdim)
        self.aggregator = aggregator
        if use_struct:
            if aggregator == 'rgcn':
                self.rgcn = RGCNAggregator(bdim, edge_mask=edge_mask, causal=causal_edges)
            elif aggregator == 'gat':
                self.rgcn = GATAggregator(bdim)
            elif aggregator == 'mlp':
                self.rgcn = MLPAggregator(bdim)
            else:
                self.rgcn = None
        else:
            self.rgcn = None
        d = bdim

        self.region_head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, num_regions))
        route_in = d
        self.route_head = nn.Sequential(nn.Linear(route_in, hidden), nn.ReLU(), nn.Linear(hidden, num_routes))
        # 时长头输入：basic = [hidden, geo, gps, prior]；
        # full 额外加 [pos_embed(8), cum_dist(1), cum_time(1)]（route 不作为输入，避免未来泄漏）
        time_in = d + geo_dim + GPS_SEG_DIM + 1
        self.time_extra = 0
        if time_feats == 'full':
            self.time_extra = 8
        self.time_head = nn.Sequential(nn.Linear(time_in + self.time_extra, hidden), nn.ReLU(),
                                       nn.Linear(hidden, 1))
        # 概率化时长头（可选）：输出 (μ, log σ)，σ = softplus(logσ)+eps
        self.prob = prob
        self.time_head_prob = None
        if prob:
            self.time_head_prob = nn.Sequential(nn.Linear(time_in + self.time_extra, hidden), nn.ReLU(),
                                                nn.Linear(hidden, 2))
            self.register_buffer('sigma_cal', torch.tensor(1.0))

    def set_sigma_cal(self, factor):
        """σ 校准：验证集上调温度系数使目标 PICP 命中。"""
        self.sigma_cal = torch.tensor(float(factor))

    def _prior(self, region_from, region_next):
        """(B,) 区域对先验（归一化）。"""
        v = self.prior_table[region_from, region_next]  # (B,)
        return v

    def forward(self, region_ids, geo, gps, time_feat, role_ids, positions, route_id,
                cum_dist=None, cum_time=None, interval=None):
        B, L, _ = geo.shape
        h0 = self.projector(region_ids, geo, gps, time_feat, role_ids, positions)
        if self.interval_embed is not None:
            if interval is None and cum_time is not None:
                ct = cum_time
                diff = ct[:, 1:] - ct[:, :-1]           # 步间间隔（归一化）
                diff = torch.clamp(diff, min=0.0)
                iv = torch.cat([torch.zeros(B, 1, device=ct.device), diff], dim=1)
            elif interval is not None:
                iv = torch.clamp(interval.float(), min=0.0)
            else:
                iv = torch.zeros(B, L, device=h0.device)
            b = torch.bucketize(iv, self.interval_bins)   # 0..nbins；0(起点/padding) → 桶0
            h0 = h0 + self.interval_embed(b)
        if self.rgcn is not None:
            if self.aggregator == 'rgcn':
                h0 = self.rgcn(h0, region_ids, role_ids, geo[..., 3:], time_feat)
            else:
                h0 = self.rgcn(h0)
        mask = (region_ids != -1).float()
        hidden = self.backbone(h0, mask)

        rlogits = self.region_head(hidden)          # (B,L,V)
        # where→when：用预测的下一区域取先验
        next_pred = rlogits.argmax(-1).detach()     # (B,L) 第 i 步预测 i+1
        src = region_ids.clamp(min=0, max=self.V - 1)
        prior = self._prior(src, next_pred)          # (B,L) 归一化先验
        if not self.use_prior:
            prior = torch.zeros_like(prior)
        resid_in = [hidden, geo, gps, prior.unsqueeze(-1)]
        if self.time_feats == 'full':
            pe = self.projector.pos_embed(positions)
            resid_in += [pe]
        x = torch.cat(resid_in, dim=-1)
        if self.prob:
            out = self.time_head_prob(x)                       # (B,L,2)
            mu = out[..., 0]
            log_sigma = out[..., 1]
            sigma = (nn.functional.softplus(log_sigma) + 1e-3) * self.sigma_cal
            dur_pred = prior + mu                              # 点估计 = 先验 + μ（归一化）
        else:
            resid = self.time_head(x).squeeze(-1)              # (B,L)
            dur_pred = prior + resid
            sigma = torch.zeros_like(dur_pred)

        # route：最后一个有效位置
        lens = (region_ids != -1).sum(1).clamp(min=1)
        rp = rlogits[torch.arange(B, device=hidden.device), lens - 1]
        route_logits = self.route_head(hidden[torch.arange(B, device=hidden.device), lens - 1])
        return rlogits, dur_pred, route_logits, lens, sigma


def build_pair_mean(tr_records, V):
    """训练集区域对历史均时长（秒）。返回 (V,V) 与全局均值。"""
    sums = {}
    cnts = {}
    all_d = []
    for r in tr_records:
        seq = r['seq']
        for i, d in enumerate(r['segment_durations']):
            k = (int(seq[i]), int(seq[i + 1]))
            sums[k] = sums.get(k, 0.0) + d
            cnts[k] = cnts.get(k, 0) + 1
            all_d.append(d)
    pm = np.zeros((V, V), dtype=np.float32)
    for (a, b), s in sums.items():
        if a < V and b < V:
            pm[a, b] = s / cnts[(a, b)]
    gm = float(np.mean(all_d)) if all_d else 3600.0
    return pm, gm

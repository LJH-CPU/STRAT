# -*- coding: utf-8 -*-
"""
SRTM DEM 查询（skadi 1-arcsec `.hgt`）。

- 按 (lat, lon) 定位 1°×1° tile，下载并缓存为解压后的 `.hgt`；
- 双线性插值 `elev(lats, lons)`（nan-aware，处理 SRTM void）；
- 供 build_segment_features 沿段内经纬度折线查地形，替代本次轨迹的 GPX 海拔。

数据源：https://s3.amazonaws.com/elevation-tiles-prod/skadi/{NS}{lat:02d}/{NS}{lat:02d}{EW}{lon:03d}.hgt.gz
（SRTMGL1，3601×3601，int16 大端，行 0=北边界，列 0=西边界；-32768=void）

用法：
  from dem_lookup import DEM
  dem = DEM()
  z = dem.elev(lats, lons)          # numpy 数组，双线性
  python dem_lookup.py --prefetch   # 预取 20 景区 bbox 覆盖的所有 tile
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEM_CACHE = os.environ.get("DEM_CACHE", os.path.join(HERE, "dem_cache"))
BBOX_FILE = os.path.join(HERE, "scenes_20_bbox.json")
SKADI_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{d}/{name}.hgt.gz"
VOID = -32768


def tile_of(lat, lon):
    """返回 (name, lat0, lon0)。"""
    la = math.floor(lat)
    lo = math.floor(lon)
    ns = "N" if la >= 0 else "S"
    ew = "E" if lo >= 0 else "W"
    return f"{ns}{abs(la):02d}{ew}{abs(lo):03d}", la, lo


def tile_hgt_path(lat, lon):
    name, _, _ = tile_of(lat, lon)
    return os.path.join(DEM_CACHE, name + ".hgt")


def _download_gz(lat, lon, retries=4, timeout=120):
    """下载 .hgt.gz 并解压为 .hgt（带断点续传）。返回解压后路径。"""
    name, la, lo = tile_of(lat, lon)
    os.makedirs(DEM_CACHE, exist_ok=True)
    dst = os.path.join(DEM_CACHE, name + ".hgt")
    if os.path.exists(dst):
        return dst
    part = dst + ".gz.part"
    d = f"{'N' if la >= 0 else 'S'}{abs(la):02d}"
    url = SKADI_URL.format(d=d, name=name)
    for attempt in range(retries):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            req = urllib.request.Request(url)
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            # gunzip
            with gzip.open(part, "rb") as fin, open(dst, "wb") as fout:
                while True:
                    b = fin.read(1 << 22)
                    if not b:
                        break
                    fout.write(b)
            os.remove(part)
            return dst
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return dst


@lru_cache(maxsize=8)
def _load_grid(name):
    """加载缓存 tile 为 float64 网格（void→nan）；返回 None 表示缺失。"""
    path = os.path.join(DEM_CACHE, name + ".hgt")
    if not os.path.exists(path):
        return None
    raw = np.fromfile(path, dtype=">i2")
    n = raw.size
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError(f"bad hgt size {n} for {name}")
    g = raw.reshape(side, side).astype(np.float64)
    g[g <= VOID] = np.nan
    return g


class DEM:
    def __init__(self, auto_download=True, cache_dir=None):
        self.auto_download = auto_download
        if cache_dir:
            global DEM_CACHE
            DEM_CACHE = cache_dir

    def _grid(self, la, lo):
        name = tile_of(la, lo)[0]
        g = _load_grid(name)
        if g is None and self.auto_download:
            _download_gz(la, lo)
            _load_grid.cache_clear()
            g = _load_grid(name)
        return g

    def elev(self, lats, lons):
        """双线性插值高程（米）。nan-aware；越界/void 返回 nan。"""
        lats = np.asarray(lats, dtype=np.float64)
        lons = np.asarray(lons, dtype=np.float64)
        out = np.full(lats.shape, np.nan, dtype=np.float64)
        if lats.size == 0:
            return out
        la = np.floor(lats).astype(np.int64)
        lo = np.floor(lons).astype(np.int64)
        stack = np.stack([la, lo], axis=1)
        uniq, inv = np.unique(stack, axis=0, return_inverse=True)
        for k in range(uniq.shape[0]):
            tla, tlo = int(uniq[k, 0]), int(uniq[k, 1])
            mask = inv == k
            grid = self._grid(tla, tlo)
            if grid is None:
                continue
            side = grid.shape[0]
            rf = (tla + 1.0 - lats[mask]) * (side - 1)
            cf = (lons[mask] - tlo) * (side - 1)
            rf = np.clip(rf, 0.0, side - 1 - 1e-9)
            cf = np.clip(cf, 0.0, side - 1 - 1e-9)
            r0 = np.floor(rf).astype(np.int64)
            c0 = np.floor(cf).astype(np.int64)
            r1 = np.minimum(r0 + 1, side - 1)
            c1 = np.minimum(c0 + 1, side - 1)
            wr = rf - r0
            wc = cf - c0
            corners = (grid[r0, c0], grid[r0, c1], grid[r1, c0], grid[r1, c1])
            weights = ((1 - wr) * (1 - wc), (1 - wr) * wc, wr * (1 - wc), wr * wc)
            num = np.zeros(mask.sum(), dtype=np.float64)
            den = np.zeros(mask.sum(), dtype=np.float64)
            for g, w in zip(corners, weights):
                valid = ~np.isnan(g)
                num += np.where(valid, w * np.where(valid, g, 0.0), 0.0)
                den += np.where(valid, w, 0.0)
            res = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
            out[mask] = res
        return out

    def elev_point(self, lat, lon):
        return float(self.elev(np.array([lat]), np.array([lon]))[0])


def tiles_for_bbox(bbox):
    latmin, latmax, lonmin, lonmax = bbox
    out = []
    for la in range(math.floor(latmin), math.floor(latmax) + 1):
        for lo in range(math.floor(lonmin), math.floor(lonmax) + 1):
            out.append((la, lo))
    return out


def prefetch(bboxes, workers=8):
    jobs = []
    seen = set()
    for bbox in bboxes:
        for la, lo in tiles_for_bbox(bbox):
            name = tile_of(la, lo)[0]
            if name not in seen:
                seen.add(name)
                jobs.append((la, lo, name))
    print(f"tiles to ensure: {len(jobs)}", flush=True)
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download_gz, la, lo): name for la, lo, name in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            name = futs[fut]
            try:
                fut.result()
                ok += 1
                print(f"[{i}/{len(jobs)}] {name} ok", flush=True)
            except Exception as e:
                print(f"[{i}/{len(jobs)}] {name} FAIL {type(e).__name__}: {e}", flush=True)
    print(f"done {ok}/{len(jobs)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--bbox-file", default=BBOX_FILE)
    args = ap.parse_args()
    with open(args.bbox_file, encoding="utf-8") as f:
        bbox = json.load(f)
    scenes = args.scenes or list(bbox.keys())
    bboxes = [bbox[s] for s in scenes]
    if args.prefetch:
        prefetch(bboxes, workers=args.workers)
        return
    dem = DEM()
    for s in scenes:
        b = bbox[s]
        lat = (b[0] + b[1]) / 2
        lon = (b[2] + b[3]) / 2
        print(f"{s}: center ({lat:.3f},{lon:.3f}) elev={dem.elev_point(lat, lon):.1f} m")


if __name__ == "__main__":
    main()

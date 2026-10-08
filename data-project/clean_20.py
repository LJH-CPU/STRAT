# -*- coding: utf-8 -*-
"""
20 景区批量清洗：convert_2bulu 按景区目录跑，输出 cleaned CSV + 汇总统计。

用法：
  python data-project/clean_20.py                # 全部 20 景区（4 进程并行）
  python data-project/clean_20.py --workers 6
  python data-project/clean_20.py --scenes 峨眉山 泰山
  python data-project/clean_20.py --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenes_20 import scenes, NAMES_20, CLEANED_DIR, BBOX_FILE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVERT = os.path.join(ROOT, "prediction", "convert_2bulu.py")
LOG_DIR = os.path.join(ROOT, "data-project", "logs")


def clean_one(s, bbox):
    out_csv = os.path.join(CLEANED_DIR, f"{s['name']}_2bulu_cleaned.csv")
    cmd = [sys.executable, CONVERT,
           "--gpx-dir", s["gpx_dir"],
           "--meta", s["meta_csv"],
           "--out", out_csv,
           "--bbox", ",".join(str(x) for x in bbox),
           "--min-pts", "50", "--min-km", "5", "--max-days", "3"]
    t0 = time.time()
    logf = os.path.join(LOG_DIR, f"clean_{s['name']}.log")
    with open(logf, "w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
    ok = r.returncode == 0 and os.path.exists(out_csv)
    stats = {}
    if ok:
        try:
            df = pd.read_csv(out_csv, usecols=["trackId"])
            stats = {"tracks": int(df["trackId"].nunique()), "points": int(len(df))}
        except Exception as e:
            ok, stats = False, {"error": str(e)}
    return {"name": s["name"], "ok": ok, "rc": r.returncode,
            "secs": round(time.time() - t0, 1), "log": logf, **stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(BBOX_FILE, encoding="utf-8") as f:
        bboxes = json.load(f)

    targets = [s for s in scenes() if not args.scenes or s["name"] in args.scenes]
    missing = [s["name"] for s in targets if s["name"] not in bboxes]
    if missing:
        print("缺 bbox，先跑 derive_bbox_20.py：", missing)
        return 1

    if args.dry_run:
        for s in targets:
            print(s["name"], "gpx:", len(os.listdir(s["gpx_dir"])), "bbox:", bboxes[s["name"]])
        return 0

    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"清洗 {len(targets)} 景区, workers={args.workers}", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(clean_one, s, bboxes[s["name"]]): s["name"] for s in targets}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"[{r['name']}] {'OK' if r['ok'] else 'FAIL'} rc={r['rc']} "
                  f"tracks={r.get('tracks','?')} pts={r.get('points','?')} {r['secs']}s", flush=True)

    results.sort(key=lambda r: r["name"])
    ok = [r for r in results if r["ok"]]
    print(f"\n=== 汇总: {len(ok)}/{len(results)} 成功, "
          f"tracks={sum(r.get('tracks',0) for r in ok)}, points={sum(r.get('points',0) for r in ok)} ===")
    with open(os.path.join(LOG_DIR, "clean_20_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    return 0 if len(ok) == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())

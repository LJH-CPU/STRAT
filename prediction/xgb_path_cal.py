"""A1: XGBoost 分位 + 路径级公平校准对比（4 场景 × 3 种子）。

给 XGBoost 分位传播做与 STRAT 同协议的路径级校准（track 级 val 调因子），
使两者都达名义覆盖后再比路径 CRPS/宽度（锐度）。
"""
import os
import sys
import json
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from xgboost_baseline import build_scene_data, run_xgb_quantile_cal


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', default=['峨眉山', '青城山', '熊猫基地', '龙泉'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 100, 2024])
    p.add_argument('--n_workers', type=int, default=8)
    p.add_argument('--out', default='prediction/output/llm_xgb_path_cal.json')
    args = p.parse_args()

    results = {}
    for name in args.scenes:
        for seed in args.seeds:
            try:
                tr, te, V = build_scene_data(name, seed, args.n_workers)
                r = run_xgb_quantile_cal(tr, te, V, seed)
                if r is None:
                    print(f'[{name}|{seed}] 样本不足'); continue
                print(f"[{name}|{seed}] rawPICP={r['raw_path_picp90']:.3f} calPICP={r['cal_path_picp90']:.3f} "
                      f"calCRPS={r['cal_path_crps_min']:.1f}min calW90={r['cal_path_width90_min']:.1f}min s={r['path_sigma_cal']:.2f}")
                results[f'{name}|{seed}'] = r
            except Exception as e:
                import traceback; traceback.print_exc()
                results[f'{name}|{seed}'] = {'error': str(e)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('已保存:', args.out)


if __name__ == '__main__':
    main()

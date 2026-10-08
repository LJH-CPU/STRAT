"""
全局配置（单一来源）。

所有脚本统一从这里取路径与共享常量：
- 禁止在其他模块中写死文件路径、景区名
- 所有路径基于本文件位置（PROJECT_DIR）推导，不依赖运行目录

用法（其他模块）：
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import CLEANED_DIR, POI_JSON, ...
"""
import sys
from pathlib import Path

# 项目根目录（本文件所在位置）
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

# ── 目录 ─────────────────────────────────────────────────
DATA_DIR             = PROJECT_DIR / "data-project"
CLEANED_DIR          = DATA_DIR / "cleaned_labeled_data"
CLUSTER_OUTPUT_DIR   = PROJECT_DIR / "cluster" / "output"
PREDICTION_OUTPUT_DIR = PROJECT_DIR / "prediction" / "output"
POI_DATA_DIR         = PROJECT_DIR / "poi" / "data"
POI_RAW_DIR          = POI_DATA_DIR / "raw"
POI_PROJECTED_DIR    = POI_DATA_DIR / "projected"

# ── 文件 ─────────────────────────────────────────────────
POI_JSON            = str(POI_PROJECTED_DIR / "poi_path_projected.json")
POI_PROJECTED_CSV   = str(POI_PROJECTED_DIR / "poi_path_projected.csv")
BATCH_RESULTS_JSON  = PROJECT_DIR / "batch_results.json"

# ── 共享常量 ──────────────────────────────────────────────
# POI 一级类型码 → 中文名
POI_TYPE_L1 = {
    "05": "餐饮", "08": "休闲", "10": "住宿",
    "11": "风景名胜", "14": "科教文化", "20": "公共设施",
}

# 论文使用的核心景区（仅影响 run_batch_pipeline 的 --only_paper 选项）
PAPER_SCENERIES = {"青城山", "峨眉山", "武侯祠博物馆"}

# ── 聚类常量（单一来源） ─────────────────────────────────
SEED                  = 42          # 全局随机种子
GB_SAMPLE_SIZE        = 5000        # 粒球生成/评测统一采样规模上限
MIN_POINTS_FOR_SPLIT  = 20          # 粒球细分最小点数
FULL_DISTANCE_THRESHOLD = 5000      # split_ball_by_distance 全距离矩阵上限（>此值走采样）
MERGE_THRESHOLD       = 0.5         # 邻近球合并阈值（× 较小球半径）
DELTA_GRID_LOWER      = 0.1         # δ 网格下界
DELTA_GRID_UPPER      = 1.0         # δ 网格上界
DELTA_GRID_STEP       = 0.1         # δ 网格步长
SELECTION_SPLIT       = 0.7         # δ 选参验证集比例（0.7 训练选参）


def adaptive_k_bounds(n_poi_regions, n_balls):
    """按 POI 区域数自适应 k 搜索范围。

    k_lower = max(2, n_poi_regions - 5)
    k_upper = max(12, n_poi_regions + 3)，再夹到 [2, n_balls]。
    覆盖 1~100+ 的区域数跨度，避免固定 [6,10] 在小场景强行过分割、
    在大场景又压不满。
    """
    lower = max(2, n_poi_regions - 5)
    upper = max(12, n_poi_regions + 3)
    upper = min(max(lower + 1, upper), max(2, n_balls))
    lower = min(lower, upper - 1)
    lower = max(2, lower)
    return int(lower), int(upper)


def scenery_name_from_csv(csv_path):
    """从 CSV 文件名提取景区名：'峨眉山_cleaned.csv' / '峨眉山_clustered.csv' → '峨眉山'"""
    name = Path(str(csv_path)).stem
    for suffix in ("_cleaned", "_clustered"):
        name = name.replace(suffix, "")
    return name


def default_cleaned_csv():
    """返回 CLEANED_DIR 下按名称排序的第一个 *_cleaned.csv；无则 None"""
    files = sorted(CLEANED_DIR.glob("*_cleaned.csv"))
    return str(files[0]) if files else None


def default_clustered_csv():
    """返回 CLUSTER_OUTPUT_DIR 下按名称排序的第一个 *_clustered.csv；无则 None"""
    files = sorted(CLUSTER_OUTPUT_DIR.glob("*_clustered.csv"))
    return str(files[0]) if files else None

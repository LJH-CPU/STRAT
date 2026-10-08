"""
粒球生成算法 (Granular Ball Generation)


作用：
将原始数据点聚合成少量球体 (Granular Balls)，实现数据抽象和降维

核心思想：
- 从粗到细的迭代分割策略
- 基于密度的自适应分割：密度高的区域自动分割成小球
- 基于半径的归一化：确保球体尺度合理

算法流程：
1. 数据预处理：归一化 + PCA 降维到 2D
2. 密度分割：迭代分割球体，直到密度不再提升
3. 半径归一化：分割过大的球体，确保尺度一致性
4. 输出：一组粒球，每个球包含若干原始数据点

为什么需要粒球？
- 降低计算复杂度：N 个点 → M 个球 (M << N)
- 提高鲁棒性：球体对噪声和异常点不敏感
- 几何特性：可以计算中心、半径、重叠度等特征
- 为后续谱聚类提供基础

支持文件格式：
- .csv 文件（新增，支持GPS轨迹数据）

"""

from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import numpy.linalg as la
import pandas as pd
import os

class GranularBall:
    """
    粒球类 - 存储单个球体的基本信息
    
    作用：
    - 封装球体的几何特征
    - 自动计算中心和半径
    
    参数：
    - points: 球体包含的所有数据点 (numpy 数组，形状：n×d)
    - label: 球体的标签 (用于追踪)
    
    属性：
    - points: 球体包含的所有点
    - center: 球体中心 (质心，所有点的均值)
    - label: 球体标签
    - radius: 球体半径 (中心到最远点的距离)
    - point_count: 球体包含的点数
    """
    
    def __init__(self, points, label):
        self.points = points
        # 计算中心：所有点的均值向量
        self.center = self.points.mean(0) if len(points) > 0 else np.array([])
        self.label = label
        # 计算半径
        self.radius = self._calculate_radius()
        # 记录点数
        self.point_count = len(points)
    
    def _calculate_radius(self):
        """
        计算球体半径
        
        定义：半径 = 中心点到所有点的最大距离 (欧氏距离)
        
        特殊情况：
        - 空球 (0 个点): 半径 = 0
        - 单点球 (1 个点): 半径 = 0 (点本身就是中心)
        
        返回：
        - 半径值 (非负数)
        """
        if self.point_count == 0:
            return 0.0
        if self.point_count == 1:
            return 0.0
        # 计算所有点到中心的欧氏距离
        distances = la.norm(self.points - self.center, axis=1)
        # 取最大距离作为半径
        return np.max(distances)

def calculate_radius(points):
    """
    计算点集的半径 (工具函数)
    
    作用：
    - 计算一组点的半径 (不依赖 GranularBall 类)
    - 用于后续的球体分割和归一化判断
    
    参数：
    - points: 点集 (numpy 数组，形状：n×d)
    
    返回：
    - 半径值 (中心到最远点的距离)
    
    计算步骤：
    1. 计算点集的中心 (均值)
    2. 计算所有点到中心的距离
    3. 取最大距离作为半径
    """
    num_points = len(points)
    # 单点或空集，半径为 0
    if num_points <= 1:
        return 0.0
    # 计算中心
    center = points.mean(0)
    # 计算所有点到中心的距离
    distances = la.norm(points - center, axis=1)
    # 取最大距离
    radius = np.max(distances)
    return radius

def split_ball_by_distance(points, max_points_for_full_distance=5000, rng=None):
    """
    基于距离分割球体 - 核心分割算法（最远点对二分）

    分割策略：
    1. 找到距离最远的两个点 (最远点对)
    2. 以这两个点为"种子点"
    3. 将其他点分配到距离较近的种子点
    4. 形成两个子球体

    内存控制：
    - 完整距离矩阵为 O(n²) 内存，n=50000 时约 20GB 会 OOM；
      故 max_points_for_full_distance 默认 5000，超过则用有播种的
      采样子集估计最远点对，避免内存峰值。

    参数：
    - points: 待分割的点集 (numpy 数组，形状：n×d)
    - max_points_for_full_distance: 使用完整距离矩阵的最大点数
    - rng: numpy 随机源（可传 RandomState/Generator，缺省用 np.random）

    返回：
    - [ball1_points, ball2_points]: 两个子球体的点集
    """
    if rng is None:
        rng = np.random

    # 初始化子球体点集（边界检查）
    ball1_points = []
    ball2_points = []
    num_points, num_features = points.shape
    if num_points < 2:
         return [points, np.array([])]

    
    if num_points <= max_points_for_full_distance:
        
        # 小数据集，直接计算距离矩阵
        transposed_points = points.T
        gram_matrix = np.dot(transposed_points.T, transposed_points)
        diag_gram = np.diag(gram_matrix)
        h_matrix = np.tile(diag_gram, (num_points, 1))

        # 计算距离矩阵（平方）
        distance_matrix_sq = np.maximum(0, h_matrix + h_matrix.T - gram_matrix * 2)
        distance_matrix = np.sqrt(distance_matrix_sq)
        
        # 判断是否有有效距离对
        if np.max(distance_matrix) == 0:
             mid_idx = num_points // 2
             ball1_points = points[:mid_idx]
             ball2_points = points[mid_idx:]
             return [np.array(ball1_points), np.array(ball2_points)]
        
        # 找到距离最远的两个点 (最远点对)
        row_indices, col_indices = np.where(distance_matrix == np.max(distance_matrix))
        valid_pair_found = False
        for r_idx, c_idx in zip(row_indices, col_indices):
            if r_idx != c_idx:
               point1_idx = r_idx
               point2_idx = c_idx
               valid_pair_found = True
               break
        
        # 如果没有有效距离对，随机分配点
        if not valid_pair_found:
            mid_idx = num_points // 2
            ball1_points = points[:mid_idx]
            ball2_points = points[mid_idx:]
            return [np.array(ball1_points), np.array(ball2_points)]
    else:
        # 大数据集，使用有播种的采样策略避免内存溢出
        sample_size = min(max_points_for_full_distance, num_points)
        sample_indices = rng.choice(num_points, sample_size, replace=False)
        sample_points = points[sample_indices]
        
        # 计算采样点的距离矩阵
        transposed_sample = sample_points.T
        gram_matrix_sample = np.dot(transposed_sample.T, transposed_sample)
        diag_gram_sample = np.diag(gram_matrix_sample)
        h_matrix_sample = np.tile(diag_gram_sample, (sample_size, 1))
        distance_matrix_sq_sample = np.maximum(0, h_matrix_sample + h_matrix_sample.T - gram_matrix_sample * 2)
        distance_matrix_sample = np.sqrt(distance_matrix_sq_sample)
        
        # 找到采样点中距离最远的两个点 (最远点对)   
        max_dist_sample = np.max(distance_matrix_sample)
        candidate_indices = np.where(distance_matrix_sample == max_dist_sample)
        
        for r_idx, c_idx in zip(candidate_indices[0], candidate_indices[1]):
            if r_idx != c_idx:
                sample_point1_idx = r_idx
                sample_point2_idx = c_idx
                break
        
        # 找到实际点的索引
        actual_point1_idx = sample_indices[sample_point1_idx]
        actual_point2_idx = sample_indices[sample_point2_idx]
        
        point1_idx = actual_point1_idx
        point2_idx = actual_point2_idx
    

    # 分配其他点到距离较近的种子点    
    for j in range(num_points):
        dist_to_p1 = np.linalg.norm(points[j] - points[point1_idx])
        dist_to_p2 = np.linalg.norm(points[j] - points[point2_idx])
        if dist_to_p1 <= dist_to_p2:
            ball1_points.append(points[j, :])
        else:
            ball2_points.append(points[j, :])
    
    # 处理边界情况：如果一个球体为空，将种子点添加到另一个球体
    if not ball1_points:
        ball1_points.append(points[point2_idx])
        ball2_points = [p for i, p in enumerate(ball2_points) if not np.array_equal(p, points[point2_idx])]
    elif not ball2_points:
        ball2_points.append(points[point1_idx])
        ball1_points = [p for i, p in enumerate(ball1_points) if not np.array_equal(p, points[point1_idx])]
    
    return [np.array(ball1_points), np.array(ball2_points)]


def calculate_density(points):
    """
    计算点集的密度
    
    作用：
    - 评估球体的紧密程度
    - 用于判断是否需要分割 (密度分割策略)
    
    密度定义：
    密度 = 点数 / 总半径
    其中：总半径 = 所有点到中心的距离之和
    
    直观理解：
    - 高密度：点多、半径小 → 紧密聚集
    - 低密度：点少、半径大 → 松散分布
    
    为什么用密度判断分割？
    - 高密度区域应该分割成更小的球
    - 分割后如果密度提升，说明分割有效
    - 避免无意义的分割 (越分越散)
    
    参数：
    - points: 点集 (numpy 数组)
    
    返回：
    - 密度值 (越大表示越紧密)
    
    特殊情况：
    - 点数 <= 1: 返回点数 (单点或空集，密度定义不明确)
    - 半径为 0: 密度为无穷大 (所有点重合)
    """
    num_points = len(points)
    # 点数太少，密度定义不明确
    if num_points <= 1:
        return float(num_points)
    
    # 计算中心
    center = points.mean(0)
    # 计算所有点到中心的距离
    distances = la.norm(points - center, axis=1)
    # 距离之和 (总半径)
    sum_radius = np.sum(distances)
    # 平均半径 (备用)
    mean_radius = sum_radius / num_points if num_points > 0 else 0
    
    # 计算密度
    if sum_radius > 1e-9:
        # 正常情况：密度 = 点数 / 总半径
        density_volume = num_points / sum_radius
    else:
        # 半径为 0：无穷大密度 (所有点重合)
        density_volume = float('inf') if num_points > 0 else 0.0
    
    return density_volume


def split_based_on_density(ball_list, min_points_for_split=20, rng=None):
    """
    递归最远点二分分割 - 粒球细分（第一轮迭代）

    作用：
    - 迭代把每个点足够的球体一分为二，得到细粒度的粒球
    - 球粒度由 min_points_for_split 控制

    分割策略：
    1. 对点数 >= min_points_for_split 的球体做最远点对二分
    2. 仅当两个子球各自点数 >= min_points_for_split // 2 时接受分割
    3. 否则保持原球体不变

    注：
    - 早期版本以"子球加权密度 > 父球密度"作为接受条件，但数值验证表明
      该判据在最远点二分下几乎从不拒绝（拒绝率 ~0.005%），实际不发挥
      作用，故已移除；真正的粒度控制参数是 min_points_for_split。

    参数：
    - ball_list: 球体列表，每个元素是点的集合
    - min_points_for_split: 最小点数阈值，低于此值不分割（默认 20）
    - rng: 随机源，透传给 split_ball_by_distance

    返回：
    - new_ball_list: 分割后的新球体列表
    """
    if rng is None:
        rng = np.random

    # 初始化新列表
    new_ball_list = []
    min_points_after_split = max(2, min_points_for_split // 2)

    # 遍历所有球体
    for ball_points in ball_list:
        # 只有点数足够才考虑分割
        if len(ball_points) >= min_points_for_split:
            # 尝试分割
            points_child1, points_child2 = split_ball_by_distance(ball_points, rng=rng)

            # 检查分割是否有效 (两个子球都不为空)
            if len(points_child1) == 0 or len(points_child2) == 0:
                new_ball_list.append(ball_points)  # 分割失败，保持原样
                continue

            # 子球点数足够才接受分割
            sufficient_points = (len(points_child1) >= min_points_after_split and
                                 len(points_child2) >= min_points_after_split)

            if sufficient_points:
                new_ball_list.extend([points_child1, points_child2])
            else:
                # 否则保持原样
                new_ball_list.append(ball_points)
        else:
            # 点数不足，不分割
            new_ball_list.append(ball_points)

    return new_ball_list

def normalize_balls_by_radius(ball_list, detection_radius, rng=None):
    """
    基于半径的球体归一化 - 第二次迭代分割
    
    作用：
    - 确保球体半径在合理范围内
    - 避免出现过大的球体
    
    归一化策略：
    1. 计算一个参考半径 (detection_radius)
    2. 对于半径 > 2 * detection_radius 的球体，进行分割
    3. 递归检查，直到所有球体半径合理
    
    参数：
    - ball_list: 球体列表
    - detection_radius: 参考半径 (基于所有球体的中位和平均半径计算)
    - rng: 随机源，透传给 split_ball_by_distance
    
    返回：
    - temp_ball_list: 归一化后的球体列表
    
    阈值设置：
    - min_points_for_normalize = 2: 至少 2 个点才考虑分割
    - radius_threshold_factor = 2.0: 半径超过 2 倍参考半径才分割
    """
    if rng is None:
        rng = np.random

    # 初始化临时列表
    temp_ball_list = []
    min_points_for_normalize = 2
    radius_threshold_factor = 2.0  # 半径阈值因子
    
    # 遍历所有球体
    for ball_points in ball_list:
        # 点数不足，不分割
        if len(ball_points) < min_points_for_normalize:
            temp_ball_list.append(ball_points)
        else:
            # 计算当前球体半径
            current_radius = calculate_radius(ball_points)
            
            # 检查半径是否合理
            if current_radius <= radius_threshold_factor * detection_radius:
                # 半径合理，保留
                temp_ball_list.append(ball_points)
            else:
                # 半径过大，分割
                points_child1, points_child2 = split_ball_by_distance(ball_points, rng=rng)
                # 将子球加入列表 (递归检查会在下一轮迭代进行)
                if len(points_child1) > 0:
                    temp_ball_list.append(points_child1)
                if len(points_child2) > 0:
                    temp_ball_list.append(points_child2)
    
    return temp_ball_list

def generate_granular_balls(dataset_name, data_path_prefix="", sampling_size=None,
                            use_stratified_sampling=True, random_state=None):
    """
    生成粒球 - 主函数（优化内存版本）

    【注意】本入口为按数据集名加载并做预处理的旧版封装，论文方法以
    scenery_route_clustering.generate_granular_balls（直接接收特征）为准。
    本函数保留用于数据调试，特征统一为 经度/纬度/海拔，不再纳入速度列，
    不再做 PCA。

    完整流程：
    1. 加载数据 (.csv / .mat)
    2. 采样策略：避免处理全部数据点
    3. 数据预处理：MinMax 归一化（3D：经度/纬度/海拔）
    4. 递归最远点二分分割
    5. 计算参考半径：基于所有球体的中位和平均半径
    6. 半径归一化：迭代分割直到半径收敛
    7. 返回结果：真实标签、归一化特征、球体列表、点数

    参数：
    - dataset_name: 数据集名称 (不含路径和后缀)
    - data_path_prefix: 数据文件路径前缀
    - sampling_size: 采样大小（默认None，自动根据数据量调整）
    - use_stratified_sampling: 是否使用分层采样（默认True，保留类别信息）
    - random_state: 随机种子（默认None，不固定）

    返回：
    - ground_truth: 真实标签 (用于评估)
    - scaled_features: MinMax 归一化后的特征 (3D)
    - final_ball_list: 最终生成的球体列表
    - num_points: 原始数据点数
    """
    if random_state is not None:
        np.random.seed(random_state)
    rng = np.random

    csv_path = f"{data_path_prefix}{dataset_name}.csv"
    mat_path = f"{data_path_prefix}{dataset_name}.mat"
    
    features = None
    ground_truth = None
    full_features = None
    
    # 加载CSV文件
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            feature_columns = []
            if '经度' in df.columns and '纬度' in df.columns:
                feature_columns.extend(['经度', '纬度'])
            if '海拔' in df.columns:
                feature_columns.append('海拔')
            
            if not feature_columns:
                raise ValueError("No valid feature columns found in CSV file.")
            
            full_features = df[feature_columns].values
            num_points = len(full_features)
            
            if 'route_id' in df.columns:
                full_ground_truth = df['route_id'].values
            else:
                full_ground_truth = np.zeros(num_points, dtype=int)
            
            print(f"Loaded CSV file: {csv_path}")
            print(f"  Full dataset size: {num_points} points")
            
            unique_routes = np.unique(full_ground_truth)
            print(f"  Detected {len(unique_routes)} unique routes/classes")
            
            if sampling_size is None:
                sampling_size = min(10000, max(5000, int(np.sqrt(num_points))))
            
            if num_points > sampling_size:
                print(f"  Sampling {sampling_size} points for granular ball generation...")
                
                if use_stratified_sampling and len(unique_routes) > 1:
                    samples_per_route = max(50, sampling_size // len(unique_routes))
                    sample_indices = []
                    
                    for route_id in unique_routes:
                        route_mask = full_ground_truth == route_id
                        route_indices = np.where(route_mask)[0]
                        
                        if len(route_indices) <= samples_per_route:
                            sample_indices.extend(route_indices)
                        else:
                            selected = rng.choice(route_indices, samples_per_route, replace=False)
                            sample_indices.extend(selected)
                    
                    sample_indices = np.sort(np.array(sample_indices))
                    features = full_features[sample_indices]
                    ground_truth = full_ground_truth[sample_indices]
                    
                    print(f"  Stratified sampling: {len(sample_indices)} points from {len(unique_routes)} routes")
                    print(f"  Per-route sample size: ~{samples_per_route} points")
                else:
                    sample_indices = rng.choice(num_points, sampling_size, replace=False)
                    sample_indices = np.sort(sample_indices)
                    features = full_features[sample_indices]
                    ground_truth = full_ground_truth[sample_indices]
                    print(f"  Random sampling: {sampling_size} points")
            else:
                features = full_features
                ground_truth = full_ground_truth
            
            print(f"  Sampled features shape: {features.shape}")
            print(f"  Sampled routes: {np.unique(ground_truth)}")
            print(f"  Points per route: {dict(zip(*np.unique(ground_truth, return_counts=True)))}")
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise
    elif os.path.exists(mat_path):
        try:
            mat_data = loadmat(mat_path)
            full_features = mat_data['fea']
            full_ground_truth = mat_data['gt'].flatten()
            num_points = full_features.shape[0]
            
            if sampling_size is None:
                sampling_size = min(10000, max(5000, int(np.sqrt(num_points))))
            
            if num_points > sampling_size:
                print(f"Loaded MAT file: {mat_path}")
                print(f"  Full dataset size: {num_points} points")
                print(f"  Sampling {sampling_size} points for granular ball generation...")
                sample_indices = rng.choice(num_points, sampling_size, replace=False)
                sample_indices = np.sort(sample_indices)
                features = full_features[sample_indices]
                ground_truth = full_ground_truth[sample_indices]
            else:
                features = full_features
                ground_truth = full_ground_truth
            
            print(f"  Shape: {features.shape}")
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {mat_path}")
            raise
        except Exception as e:
            print(f"Error loading MAT file: {e}")
            raise
    else:
        print(f"Error: Neither CSV nor MAT file found.")
        print(f"  Tried: {csv_path} or {mat_path}")
        raise FileNotFoundError(f"Dataset file not found: {dataset_name}")
    
    # 步骤 2: 数据预处理
    # 2.1 MinMax 归一化（3D：经度/纬度/海拔）
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    # 初始化：所有点作为一个大球
    current_ball_list = [scaled_features]
    
    # 步骤 3: 递归最远点二分分割 (迭代)
    iteration_count = 0
    max_iterations = 50  # 最大迭代次数，防止无限循环
    
    # 迭代分割，直到球数不再增加或达到最大迭代次数
    while iteration_count < max_iterations:
        iteration_count += 1
        ball_count_before_split = len(current_ball_list)
        # 执行递归最远点二分
        current_ball_list = split_based_on_density(current_ball_list, rng=rng)
        ball_count_after_split = len(current_ball_list)
        
        # 如果没有球体被分割，说明已经收敛
        if ball_count_after_split == ball_count_before_split:
            break
    
    # 达到最大迭代次数，警告
    if iteration_count == max_iterations:
        print("Warning: Max iterations reached during ball splitting.")
    
    # 步骤 4: 计算参考半径 (detection_radius)
    # 只计算点数 >= 2 的球体半径
    radii = [calculate_radius(ball_points) for ball_points in current_ball_list if len(ball_points) >= 2]
    if not radii:
         print("Warning: No balls with sufficient points to calculate radii.")
         detection_radius = 0.0
    else:
        # 取中位数和平均值的较大者作为参考半径
        radius_median = np.median(radii)
        radius_mean = np.mean(radii)
        detection_radius = max(radius_median, radius_mean, 1e-6)
    
    # 步骤 5: 半径归一化 (迭代)
    iteration_count = 0
    while iteration_count < max_iterations:
         iteration_count += 1
         ball_count_before_norm = len(current_ball_list)
         # 执行半径归一化
         current_ball_list = normalize_balls_by_radius(current_ball_list, detection_radius, rng=rng)
         ball_count_after_norm = len(current_ball_list)
         
         # 如果没有球体被分割，说明已经收敛
         if ball_count_after_norm == ball_count_before_norm:
             break
    
    # 达到最大迭代次数，警告
    if iteration_count == max_iterations:
        print("Warning: Max iterations reached during radius normalization.")
    
    # 步骤 6: 过滤空球
    final_ball_list = [ball for ball in current_ball_list if len(ball) > 0]
    
    # 返回结果
    return ground_truth, scaled_features, final_ball_list, num_points

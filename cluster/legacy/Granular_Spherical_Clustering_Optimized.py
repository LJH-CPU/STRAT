"""
[DEPRECATED - 已归档] 监督 BKOA 版粒球谱聚类（不再用于论文方法/实验）。

原因：BKOA 以 ground-truth ARI 为适应度调参存在信息泄漏，且当前消融
证明不了其优于网格搜索；论文方法本体改为 scenery_route_clustering.py
（粒球 + 规则亲和力 + 网格选 δ + eigen-gap 自动 k）。本文件仅供回溯。

若直接运行需先: sys.path.insert(0, 目录上一级)。
"""
import gc  # 导入垃圾回收模块
import os  # 导入操作模块
import time  # 导入时间模块
from math import exp, sqrt  # 导入指数和平方根函数
import numpy as np  # 导入 numpy 库
from sklearn import metrics  # 导入 sklearn 库的指标模块
from sklearn.cluster import SpectralClustering, KMeans  # 导入 sklearn 库的谱聚类和 KMeans 聚类模块
import traceback  # 导入 traceback 模块，用于异常处理

# 导入粒球生成算法 
try:
    from Granular_Spherical_Clustering import generate_granular_balls
except ImportError:
    print("ERROR: Ensure 'Granular_Spherical_Clustering.py' is in the same directory or accessible.")
    exit()


class BKOAOptimizer:
    """
    BKOA (Breeding Krill Optimization Algorithm) 优化器
    用于自动搜索最优的聚类参数：聚类数量 k 和 Affinity 计算参数 delta
    
    优化原理：
    - 将参数搜索问题转化为优化问题
    - 使用群体智能算法在参数空间中搜索最优解
    - 适应度函数：-ARI (因为要最大化 ARI，所以最小化 -ARI)
    
    参数说明：
    - objective_function: 适应度函数，输入参数，输出适应度值
    - bounds: 参数搜索范围 [[k_min, k_max], [delta_min, delta_max]]
    - pop_size: 种群大小，个体数量
    - max_iter: 最大迭代次数
    - args: 传递给适应度函数的额外参数 (ball_dict, features, ground_truth)
    - timeout: 超时时间 (秒)，防止优化时间过长
    """
    
    def __init__(self, objective_function, bounds, pop_size=20, max_iter=50, args=(), timeout=300):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)  # 参数边界
        self.pop_size = pop_size  # 种群大小
        self.max_iter = max_iter  # 最大迭代次数
        self.args = args  # 适应度函数的额外参数
        self.dimension = len(bounds)  # 参数维度 (这里是 2: k 和 delta)
        self.timeout = timeout  # 超时时间
        
        # 全局最优解
        self.global_best_position = None  # 最优参数组合
        self.global_best_fitness = float('inf')  # 最优适应度值
        
        # BKOA 算法参数 (控制搜索行为)
        self.alpha = 0.6  # 趋向最优个体的权重 (降低，从 0.8)
        self.beta = 0.7   # 趋向随机个体的权重 (增加，从 0.5)
        self.gamma = 0.2  # 随机扰动权重 (增加，从 0.1)
        
        # 早停机制参数
        self.early_stop_threshold = 1e-4  # 收敛阈值 (放宽 100 倍)
        self.early_stop_count = 0  # 收敛计数器
        self.early_stop_max = 10  # 最大收敛次数 (增加)
        self.min_iterations = 15  # 最小迭代次数 (新增，防止过早停止)

    def _ensure_bounds(self, position):
        """
        确保参数在合法范围内
        
        作用：
        - 防止参数越界
        - 对 k 值取整 (聚类数量必须是整数)
        - 对 delta 设置最小值 (防止除零错误)
        
        参数：
        - position: 参数向量 [k, delta]
        
        返回：
        - 处理后的参数向量
        """
        # 截断到边界范围内
        position = np.clip(position, self.bounds[:, 0], self.bounds[:, 1])
        # k 值必须是整数且至少为 2
        position[0] = max(2, int(round(position[0])))
        # delta 必须大于 0
        position[1] = max(1e-6, position[1])
        return position

    def run(self):
        """
        运行 BKOA 优化算法
        
        算法流程：
        1. 初始化种群：随机生成一组参数组合
        2. 评估适应度：计算每个参数组合的 ARI 值
        3. 迭代优化：
           - 排序：按适应度排序种群
           - 更新活跃个体：前 50% 的个体向最优解和其他优秀个体学习
           - 更新非活跃个体：在后 50% 的个体在最优解附近探索
        4. 早停判断：如果连续多次迭代没有改进，提前终止
        5. 超时判断：如果超过指定时间，提前终止
        6. 返回最优参数组合
        
        返回：
        - best_params_continuous: 最优参数 [k, delta]
        - best_fitness_value: 最优适应度值 (-ARI)
        """
        # 步骤 1: 初始化种群 - 在参数范围内随机生成初始解
        population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(self.pop_size, self.dimension))
        fitness = np.full(self.pop_size, float('inf'))  # 适应度数组，初始为无穷大
        
        # 步骤 2: 初始化部分种群并评估适应度
        init_pop_size = min(self.pop_size, 10)  # 先初始化 10 个个体
        print(f"  Initializing BKOA population ({init_pop_size}/{self.pop_size})...")
        for i in range(init_pop_size):
            population[i] = self._ensure_bounds(population[i])  # 确保参数合法
            # 计算适应度 (调用 bkoa_objective，实际执行聚类并计算 ARI)
            fitness[i] = self.objective_function(population[i], *self.args)
            # 更新全局最优
            if fitness[i] < self.global_best_fitness:
                self.global_best_fitness = fitness[i]
                self.global_best_position = population[i].copy()
            print(f"  Init progress: {i + 1}/{init_pop_size}, Current Fitness: {fitness[i]:.4f}   ", end='\r')
        print()
        
        # 步骤 3: 初始化剩余种群 - 基于当前最优解添加扰动，加速收敛
        if self.global_best_position is not None:
            print(f"  Guiding remaining population initialization using initial best.")
            bound_range = self.bounds[:, 1] - self.bounds[:, 0]  # 参数范围
            for i in range(init_pop_size, self.pop_size):
                # 在最优解附近添加高斯噪声
                noise = np.random.normal(0, 0.1 * bound_range, self.dimension)
                population[i] = self._ensure_bounds(self.global_best_position + noise)
                fitness[i] = float('inf')  # 暂时不评估，留待后续迭代评估
        else:
            print(f"  Randomly initializing remaining population.")
            # 如果没有最优解，则完全随机初始化
            for i in range(init_pop_size, self.pop_size):
                population[i] = self._ensure_bounds(
                    np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dimension)))
                fitness[i] = float('inf')
        
        start_time = time.time()  # 记录开始时间
        previous_best_fitness = self.global_best_fitness  # 记录上一次的最优适应度

        # 步骤 4: 迭代优化
        for t in range(self.max_iter):
            print(f"  BKOA Iteration: {t + 1}/{self.max_iter}, Best Fitness (-ARI): {self.global_best_fitness:.6f}    ",
                  end='\r')
            
            # 检查是否超时
            if time.time() - start_time > self.timeout:
                print(f"\n  BKOA optimization timed out ({self.timeout}s), terminating early.")
                break
            
            # 早停判断：至少运行 min_iterations 次迭代
            if t + 1 >= self.min_iterations:
                fitness_change = abs(previous_best_fitness - self.global_best_fitness)
                if fitness_change < self.early_stop_threshold:
                    self.early_stop_count += 1
                    if self.early_stop_count >= self.early_stop_max:
                        print(
                            f"\n  BKOA converged, terminating early (Threshold={self.early_stop_threshold}, Count={self.early_stop_count}).")
                        break
            previous_best_fitness = self.global_best_fitness
            
            # 评估未评估的个体 (适应度为 inf 的个体)
            for i in range(self.pop_size):
                if fitness[i] == float('inf'):
                    population[i] = self._ensure_bounds(population[i])
                    fitness[i] = self.objective_function(population[i], *self.args)
            
            # 按适应度排序 (从小到大，因为适应度 = -ARI，越小越好)
            sort_indices = np.argsort(fitness)
            population = population[sort_indices]
            fitness = fitness[sort_indices]
            
            # 更新全局最优解
            if fitness[0] < self.global_best_fitness:
                self.global_best_fitness = fitness[0]
                self.global_best_position = population[0].copy()
            
            current_iter_best_pos = population[0]  # 当前迭代的最优个体
            active_size = max(3, self.pop_size // 2)  # 活跃个体数量 (前 50%)
            
            # 步骤 5: 更新活跃个体 (前 50%，不包括最优个体)
            # 策略：向当前最优个体和其他随机优秀个体学习
            for i in range(1, active_size):
                # 从活跃个体中随机选择另一个个体
                other_active_indices = np.delete(np.arange(active_size), i)
                if not len(other_active_indices): continue  # 应该不会发生
                rand_idx = np.random.choice(other_active_indices)
                rand_pos = population[rand_idx]
                
                # BKOA 位置更新公式：
                # 新位置 = 当前位置 
                #        + α*r1*(当前最优 - 当前位置)  # 趋向最优
                #        + β*r2*(随机个体 - 当前位置)  # 趋向随机优秀个体
                #        + γ*随机扰动                # 增加多样性
                r1, r2 = np.random.rand(2)
                term1 = self.alpha * r1 * (current_iter_best_pos - population[i])
                term2 = self.beta * r2 * (rand_pos - population[i])
                term3 = self.gamma * (np.random.rand(self.dimension) - 0.5) * (self.bounds[:, 1] - self.bounds[:, 0])
                
                new_position = population[i] + term1 + term2 + term3
                new_position = self._ensure_bounds(new_position)
                new_fitness = self.objective_function(new_position, *self.args)
                
                # 贪婪选择：只有新位置更好时才接受
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness
            
            # 步骤 6: 更新非活跃个体 (后 50%)
            # 策略：在最优解附近进行局部搜索，增加探索性
            bound_range = self.bounds[:, 1] - self.bounds[:, 0]
            for i in range(active_size, self.pop_size):
                # 在最优解附近添加较大的高斯噪声
                noise = np.random.normal(0, 0.2 * bound_range, self.dimension)
                new_position = self._ensure_bounds(current_iter_best_pos + noise)
                population[i] = new_position
                fitness[i] = float('inf')  # 标记为未评估，留待下次迭代评估

        # 步骤 7: 最终检查 - 确保找到了最优解
        if self.global_best_position is None:
            print("\nWarning: Global best position was not updated. Selecting best from final population.")
            if len(population) > 0:
                # 找出适应度最小的个体 (不是 inf)
                valid_fitness_indices = np.where(fitness != float('inf'))[0]
                if len(valid_fitness_indices) > 0:
                    final_best_idx = valid_fitness_indices[np.argmin(fitness[valid_fitness_indices])]
                    self.global_best_position = population[final_best_idx].copy()
                    self.global_best_fitness = fitness[final_best_idx]
                else:
                    print("ERROR: All final fitness values are infinity. Cannot determine best.")
                    # 返回边界内的默认值
                    default_pos = self._ensure_bounds(self.bounds.mean(axis=1))
                    return default_pos, float('inf')
            else:
                print("ERROR: Population is empty. Cannot determine best.")
                default_pos = self._ensure_bounds(self.bounds.mean(axis=1))
                return default_pos, float('inf')
        
        print(f"\nBKOA Optimization Finished. Final Best Fitness (-ARI): {self.global_best_fitness:.6f}")
        # 确保最终参数在合法范围内
        final_best_position = self._ensure_bounds(self.global_best_position)
        return final_best_position, self.global_best_fitness


class GranularBallRepresentation:
    """
    粒球表示类 - 用于存储球体的几何特征
    
    作用：
    - 将一组数据点表示为一个球体
    - 计算球体的中心 (质心)
    - 计算球体的半径 (最远点距离)
    
    参数：
    - points: 球体包含的所有数据点 (numpy 数组)
    - label: 球体的标签 (用于追踪)
    
    属性：
    - center: 球体中心 (所有点的均值)
    - radius: 球体半径 (中心到最远点的距离)
    - label: 球体标签
    """
    
    def __init__(self, points, label):
        self.points = points
        # 计算中心：所有点的均值向量
        self.center = self.points.mean(0) if len(points) > 0 else np.array([])
        self.label = label
        # 计算半径
        self.radius = self._calculate_radius()
    
    def _calculate_radius(self):
        """
        计算球体半径
        
        定义：半径 = 中心点到所有点的最大距离
        特殊情况：
        - 点数 <= 1: 半径为 0 (单个点或空集)
        - 中心为空：半径为 0
        
        返回：
        - 半径值 (非负数)
        """
        if self.points.shape[0] <= 1 or self.center.size == 0:
            return 0.0
        # 计算所有点到中心的欧氏距离
        distances = np.linalg.norm(self.points - self.center, axis=1)
        # 取最大距离作为半径
        return np.max(distances) if len(distances) > 0 else 0.0


def calculate_affinity(center1, center2, radius1, radius2, delta_squared_term):
    """
    计算两个球体之间的相似度 (Affinity)
    
    原理：
    - 基于球体的几何关系计算相似度
    - 使用高斯核函数将距离转换为相似度
    - 重叠的球体相似度为 1，距离越远相似度越低
    
    参数：
    - center1, center2: 两个球体的中心 (向量)
    - radius1, radius2: 两个球体的半径
    - delta_squared_term: delta 参数的平方项 (2 * delta^2)
      控制相似度衰减的速度，delta 越大，衰减越慢
    
    返回：
    - affinity: 相似度值 [0, 1]，1 表示完全相同，0 表示完全无关
    
    计算逻辑：
    1. 计算两个球体中心的欧氏距离
    2. 计算表面距离 gap = 中心距离 - 半径 1 - 半径 2
       - gap < 0: 球体重叠
       - gap = 0: 球体相切
       - gap > 0: 球体分离
    3. 使用指数核函数计算相似度：
       - 重叠时：affinity = 1.0
       - 分离时：affinity = exp(-gap / delta_squared_term)
    """
    # 验证中心是否有效
    if center1.size == 0 or center2.size == 0: 
        return 0.0
    
    # 计算中心距离
    distance = np.linalg.norm(center1 - center2)
    # 计算表面距离 (gap)
    gap = distance - radius1 - radius2
    
    # 处理 delta 为 0 的情况 (避免除零错误)
    if delta_squared_term <= 1e-12:
        return 1.0 if gap < 1e-9 else 0.0
    
    # 使用指数核函数计算相似度
    # gap > 0 时：指数衰减
    # gap <= 0 时：完全相似 (重叠)
    affinity = exp(-gap / delta_squared_term) if gap > 0 else 1.0
    # 确保非负
    return max(0, affinity)


def calculate_affinity_improved(center1, center2, radius1, radius2, delta_squared_term, delta_param=None):
    """
    改进的 Affinity 计算函数 - 多因子融合版本
    
    改进点：
    1. 考虑重叠度 (Overlap Degree) - 区分部分重叠和完全重叠
    2. 考虑半径差异 (Radius Imbalance) - 大球和小球的关系 vs 同等大小球的关系
    3. 考虑相对距离 (Relative Distance) - 归一化的距离，而非绝对距离
    4. 自适应 delta - 基于球体半径自动调整
    
    核心思想：
    - 重叠的球体：根据重叠程度给予 0.7-1.0 的相似度
    - 分离的球体：根据相对距离和半径平衡计算相似度
    - 包含关系：小球在大球内部给予更高的相似度
    
    参数：
    - center1, center2: 两个球体的中心 (numpy 向量)
    - radius1, radius2: 两个球体的半径
    - delta_squared_term: delta 参数的平方项 (2 * delta^2)
    - delta_param: delta 原始值 (可选，用于自适应计算)
    
    返回：
    - affinity: 相似度值 [0, 1]
    """
    # 步骤 0: 验证输入有效性
    if center1.size == 0 or center2.size == 0:
        return 0.0
    
    # 处理半径为 0 的情况 (单点)
    if radius1 <= 0 or radius2 <= 0:
        # 退化为点之间的距离
        distance = np.linalg.norm(center1 - center2)
        # 快速衰减
        return exp(-distance * 10)
    
    # 步骤 1: 计算基础几何关系
    distance = np.linalg.norm(center1 - center2)  # 中心距离
    sum_radii = radius1 + radius2  # 半径和
    gap = distance - sum_radii  # 表面距离
    
    # 步骤 2: 计算重叠度 (Overlap Ratio)
    if gap < 0:
        # 重叠情况：计算重叠比例
        overlap_depth = -gap  # 重叠深度
        max_possible_overlap = 2 * min(radius1, radius2)  # 最大可能重叠
        overlap_ratio = min(1.0, overlap_depth / max_possible_overlap)
    else:
        overlap_ratio = 0.0
    
    # 步骤 3: 计算半径平衡因子 (Radius Balance Factor)
    radius_min = min(radius1, radius2)
    radius_max = max(radius1, radius2)
    radius_balance = radius_min / radius_max if radius_max > 0 else 0.0
    # radius_balance = 1.0: 半径相等
    # radius_balance = 0.0: 极度不等
    
    # 步骤 4: 检测包含关系
    # 如果小球中心 + 小球半径 <= 大球半径，则小球完全在大球内部
    if distance + radius_min <= radius_max:
        containment = 1.0  # 完全包含
    elif gap < 0:
        containment = 0.5  # 部分重叠 (可能包含)
    else:
        containment = 0.0  # 不包含
    
    # 步骤 5: 计算相对距离 (Relative Gap)
    relative_gap = gap / sum_radii if sum_radii > 0 else 0.0
    
    # 步骤 6: 多因子融合的 Affinity 计算
    if overlap_ratio > 0:
        # 重叠情况
        # 基础相似度 0.7 + 重叠奖励 (最多 0.3)
        base_affinity = 0.7
        overlap_bonus = 0.3 * overlap_ratio
        
        # 包含关系额外奖励 (最多 0.1)
        containment_bonus = 0.1 * containment
        
        # 半径差异大时略微增加 (包含关系可能性大)
        radius_bonus = 0.05 * (1 - radius_balance)
        
        affinity = min(1.0, base_affinity + overlap_bonus + containment_bonus + radius_bonus)
    else:
        # 不重叠情况
        # 使用改进的指数衰减
        
        # 自适应 delta
        if delta_param is None:
            delta = np.sqrt(delta_squared_term / 2)
        else:
            delta = delta_param
        
        # 考虑半径平衡的衰减因子
        # 半径相近的球体，衰减更慢
        balance_factor = 0.5 + 0.5 * radius_balance  # 范围：0.5~1.0
        effective_delta = delta * balance_factor
        
        # 相对距离的指数衰减
        # 使用相对距离而非绝对距离
        if relative_gap > 0:
            distance_penalty = exp(-relative_gap / (effective_delta + 1e-9))
        else:
            distance_penalty = 1.0
        
        # 半径平衡的额外权重
        # 半径相近的球体关系更紧密
        radius_weight = 0.7 + 0.3 * radius_balance
        
        # 包含关系的额外权重
        containment_weight = 1.0 + 0.2 * containment
        
        affinity = distance_penalty * radius_weight * containment_weight
        
        # 确保范围 [0, 1]
        affinity = max(0.0, min(1.0, affinity))
    
    return affinity


def create_ball_dictionary(ball_data_list):
    """
    创建球体字典 - 将原始球体数据转换为带几何特征的表示
    
    作用：
    - 为每个球体计算中心、半径等几何特征
    - 过滤掉无效球体 (没有中心点的球体)
    - 建立球体索引，方便后续处理
    
    参数：
    - ball_data_list: 球体数据列表，每个元素是一个 numpy 数组 (包含该球的所有点)
    
    返回：
    - ball_dict: 字典，key 为球体索引，value 为 GranularBallRepresentation 对象
    
    示例：
    ball_dict = {
        0: GranularBallRepresentation(points=array([[...]]), label=0),
        1: GranularBallRepresentation(points=array([[...]]), label=1),
        ...
    }
    """
    ball_dict = {}
    valid_ball_count = 0
    
    # 遍历所有球体
    for i, points in enumerate(ball_data_list):
        # 只处理包含点的球体
        if len(points) > 0:
            # 创建球体表示对象 (自动计算中心和半径)
            gb_repr = GranularBallRepresentation(points, valid_ball_count)
            
            # 验证球体是否有效 (中心必须存在)
            if gb_repr.center.size > 0:
                ball_dict[valid_ball_count] = gb_repr
                valid_ball_count += 1
            else:
                print(f"Warning: Skipping ball {i} with {len(points)} points as it has no center.")
    
    return ball_dict


def perform_clustering_and_evaluate(ball_dict, num_clusters, delta_param, original_features, original_ground_truth):
    """
    执行谱聚类并评估性能 - 核心聚类函数
    
    作用：
    1. 基于球体字典构建 Affinity Matrix
    2. 使用谱聚类对球体进行聚类
    3. 将球体聚类标签分配给原始数据点
    4. 计算 ARI 指标评估聚类质量
    
    参数：
    - ball_dict: 球体字典 (key: 索引，value: GranularBallRepresentation)
    - num_clusters: 聚类数量 k
    - delta_param: Affinity 计算参数，控制相似度衰减速度
    - original_features: 原始数据特征 (用于标签分配)
    - original_ground_truth: 原始真实标签 (用于评估 ARI)
    
    返回：
    - ari_score: Adjusted Rand Index，衡量聚类与真实标签的一致性
    - final_point_labels: 每个数据点的聚类标签
    
    算法流程：
    1. 提取球体中心和半径
    2. 计算 Affinity Matrix (相似度矩阵)
    3. 应用谱聚类
    4. 将球体标签分配给原始数据点 (基于最近邻)
    5. 计算 ARI 评估指标
    """
    clustering_method_used = "Spectral"
    try:
        # 步骤 1: 获取所有球体的键
        ball_keys = list(ball_dict.keys())
        num_balls = len(ball_keys)
        
        # 验证输入有效性
        if num_balls == 0:
            print("ERROR: No valid granular balls provided for clustering.")
            return 0.0, np.full(len(original_features), -1, dtype=int)
        if num_clusters <= 0:
            print(f"ERROR: Invalid number of clusters requested: {num_clusters}")
            return 0.0, np.full(len(original_features), -1, dtype=int)
        
        # 确保聚类数不超过球体数量
        if num_clusters > num_balls:
            num_clusters = num_balls
        if num_clusters == 1:
            pass  # 只有一个聚类时，所有点都属于同一类
        
        # 步骤 2: 提取球体的几何特征 (中心和半径)
        ball_centers = np.array([ball_dict[key].center for key in ball_keys])
        ball_radii = np.array([ball_dict[key].radius for key in ball_keys])
        valid_ball_keys = ball_keys  # 所有键都是有效的
        num_valid_balls = len(valid_ball_keys)
        
        # 再次检查聚类数合法性
        if num_valid_balls < num_clusters:
            num_clusters = num_valid_balls
            if num_clusters <= 0:
                print("ERROR: No valid balls left after filtering.")
                return 0.0, np.full(len(original_features), -1, dtype=int)
        
        # 步骤 3: 计算 Affinity Matrix (相似度矩阵)
        # Affinity Matrix 是谱聚类的核心输入，表示球体之间的相似度
        affinity_matrix = np.zeros((num_valid_balls, num_valid_balls))
        # delta_squared_term = 2 * delta^2，用于高斯核函数
        delta_squared_term = 2 * (delta_param ** 2) if delta_param > 1e-9 else 1e-12
        
        # 填充 Affinity Matrix (对称矩阵)
        for i in range(num_valid_balls):
            affinity_matrix[i, i] = 1.0  # 自相似度为 1
            for j in range(i + 1, num_valid_balls):
                # 计算两个球体之间的相似度
                # 使用改进的 Affinity 函数 (多因子融合)
                affinity = calculate_affinity_improved(
                    ball_centers[i], ball_centers[j],
                    ball_radii[i], ball_radii[j],
                    delta_squared_term,
                    delta_param  # 传递 delta 参数用于自适应计算
                )
                affinity_matrix[i, j] = affinity
                affinity_matrix[j, i] = affinity  # 对称矩阵
        
        # 步骤 4: 执行谱聚类
        if num_clusters == 1:
            # 只有一个聚类，所有球体都属于同一类
            ball_cluster_labels = np.zeros(num_valid_balls, dtype=int)
        else:
            try:
                # 使用谱聚类
                spectral = SpectralClustering(
                    n_clusters=num_clusters,
                    affinity="precomputed",  # 使用预计算的 Affinity Matrix
                    assign_labels="discretize",  # 标签分配方法
                    random_state=42,  # 随机种子，保证可重复性
                    n_init=10,  # 初始化次数
                    n_jobs=-1  # 使用所有 CPU 核心
                )
                
                # 数据清洗：确保 Affinity Matrix 合法
                affinity_matrix = np.nan_to_num(affinity_matrix)  # 将 NaN 转为 0
                affinity_matrix = np.maximum(affinity_matrix, 0)  # 确保非负
                affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)  # 确保对称
                
                # 执行聚类
                ball_cluster_labels = spectral.fit_predict(affinity_matrix)
                clustering_method_used = "Spectral"
            
            except Exception as spectral_error:
                # 谱聚类失败，回退到 K-Means
                print(
                    f"\nSpectral Clustering failed (n_clusters={num_clusters}, delta={delta_param:.4f}): {spectral_error}. Trying K-Means fallback...")
                try:
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                    # K-Means 直接使用球体中心作为特征
                    ball_cluster_labels = kmeans.fit_predict(ball_centers)
                    clustering_method_used = "KMeans_Fallback"
                except Exception as kmeans_error:
                    print(f"K-Means fallback also failed: {kmeans_error}")
                    traceback.print_exc()
                    return 0.0, np.full(len(original_features), -1, dtype=int)
        
        # 步骤 5: 将球体聚类标签分配给原始数据点
        final_point_labels = np.full(len(original_features), -1, dtype=int)
        
        # 建立球体索引到聚类标签的映射
        valid_key_to_cluster_map = {valid_ball_keys[i]: ball_cluster_labels[i] for i in range(num_valid_balls)}
        
        # 为每个原始数据点分配标签
        if num_valid_balls > 0:
            for point_idx, point in enumerate(original_features):
                # 找到距离该点最近的球体 (基于欧氏距离)
                distances_sq = np.sum((ball_centers - point) ** 2, axis=1)
                nearest_ball_idx = np.argmin(distances_sq)
                nearest_ball_key = valid_ball_keys[nearest_ball_idx]
                
                # 如果最近的球体有聚类标签，则分配给该点
                if nearest_ball_key in valid_key_to_cluster_map:
                    final_point_labels[point_idx] = valid_key_to_cluster_map[nearest_ball_key]
        
        # 步骤 6: 计算 ARI 评估指标
        # 只考虑有效标签 (ground_truth != -1 且 predicted_label != -1)
        valid_indices = np.where((original_ground_truth != -1) & (final_point_labels != -1))[0]
        if len(valid_indices) < 2:
            # 样本太少，无法计算 ARI
            ari_score = 0.0
        else:
            try:
                # Adjusted Rand Index: 衡量两个标签分布的一致性
                # ARI = 1 表示完全一致，ARI = 0 表示随机一致
                ari_score = metrics.adjusted_rand_score(
                    original_ground_truth[valid_indices],
                    final_point_labels[valid_indices]
                )
            except Exception as ari_error:
                print(f"Error calculating ARI: {ari_error}")
                ari_score = 0.0
        
        return ari_score, final_point_labels
    
    except Exception as e:
        # 捕获所有异常，防止程序崩溃
        print(f"\nSevere error during clustering process: {e}")
        traceback.print_exc()
        return 0.0, np.full(len(original_features), -1, dtype=int)


def bkoa_objective(params, ball_dict, original_features, original_ground_truth):
    """
    BKOA 优化算法的适应度函数 - 评估参数组合的质量
    
    作用：
    - 将 BKOA 搜索的参数组合转换为实际的聚类结果
    - 计算 ARI 作为评估指标
    - 返回 -ARI 作为适应度值 (因为 BKOA 是最小化优化)
    
    参数：
    - params: 参数向量 [k, delta]
      - k: 聚类数量 (整数)
      - delta: Affinity 计算参数 (连续值，会被离散化)
    - ball_dict: 球体字典
    - original_features: 原始数据特征
    - original_ground_truth: 原始真实标签
    
    返回：
    - fitness: 适应度值 = -ARI
      - 越小越好 (因为要最大化 ARI)
      - ARI = 1 时，fitness = -1 (最优)
      - ARI = 0 时，fitness = 0 (最差)
    
    为什么使用 -ARI？
    - BKOA 是最小化优化算法
    - 我们希望最大化 ARI
    - 因此使用 -ARI 作为适应度，最小化 -ARI 等价于最大化 ARI
    """
    # 提取参数
    num_clusters = int(params[0])  # 聚类数量
    continuous_delta = float(params[1])  # delta 参数 (连续值)
    
    # 将 delta 离散化到 [0.1, 1.0]，步长 0.1
    # 原因：
    # 1. delta 对聚类结果影响很大，需要精细控制
    # 2. 离散化可以减小搜索空间，加速优化
    # 3. 实验表明 0.1 的步长已经足够
    snapped_delta = round(continuous_delta * 10.0) / 10.0
    min_discrete_delta = 0.1
    max_discrete_delta = 1.0
    final_discrete_delta = max(min_discrete_delta, min(max_discrete_delta, snapped_delta))
    
    # 验证聚类数量
    if num_clusters < 1:
        return 1.0  # 返回最差适应度
    
    # 执行聚类并计算 ARI
    ari_score, _ = perform_clustering_and_evaluate(
        ball_dict, num_clusters, final_discrete_delta, original_features, original_ground_truth)
    
    # 返回 -ARI 作为适应度 (最小化优化)
    fitness = -ari_score
    return fitness


def main():
    original_datasets = {'峨眉山_cleaned': 3}
    
    datasets_to_run = ['峨眉山_cleaned']
    
    BKOA_POP_SIZE = 3
    BKOA_MAX_ITER = 2
    BKOA_TIMEOUT_SECONDS = 30
    
    optimization_results = {}
    
    DATASET_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data-project', 'cleaned_labeled_data')) 
    
    if not os.path.isdir(DATASET_DIRECTORY):
        print(f"ERROR: Dataset directory not found: '{DATASET_DIRECTORY}'")
        print("Please modify the DATASET_DIRECTORY variable in the script.")
        return
    
    for dataset_name in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"Processing Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        n_clusters_hint = original_datasets.get(dataset_name, 3)
        delta_hint = 0.1
        
        continuous_delta_lower = 0.05
        continuous_delta_upper = 1.05
        
        n_clusters_lower = max(2, int(n_clusters_hint * 0.7))
        n_clusters_upper = max(n_clusters_lower + 2, int(n_clusters_hint * 1.5) + 1)
        
        parameter_bounds = [
            [n_clusters_lower, n_clusters_upper],
            [continuous_delta_lower, continuous_delta_upper]
        ]
        print(
            f"Parameter Bounds: n_clusters={parameter_bounds[0]}, continuous_delta=[{parameter_bounds[1][0]:.3f}, {parameter_bounds[1][1]:.3f}]")
        print(f"  (delta will be discretized to [0.1, 1.0] step 0.1 for evaluation)")
        
        gc.collect()
        time_total_start = time.time()
        
        # === Step 1: Generate Granular Balls ===
        print("\nStep 1: Generating Granular Balls...")
        time_gb_start = time.time()
        try:
            dataset_file_path_prefix = os.path.join(DATASET_DIRECTORY, "")
            ground_truth, features, ball_data_list, num_data_points = generate_granular_balls(
                dataset_name,
                data_path_prefix=dataset_file_path_prefix
            )
        except FileNotFoundError:
            print(f"Skipping dataset {dataset_name} due to file error.")
            continue
        except Exception as e:
            print(f"Error during granular ball generation for {dataset_name}: {e}")
            traceback.print_exc()
            continue
        time_gb_end = time.time()
        print(f"  Granular ball generation time: {time_gb_end - time_gb_start:.2f} seconds")
        print(f"  Data points: {num_data_points}, Generated raw balls: {len(ball_data_list)}")

        # === Step 2: Prepare Ball Dictionary ===
        print("Step 2: Preparing Ball Dictionary...")
        ball_dict = create_ball_dictionary(ball_data_list)
        num_valid_balls = len(ball_dict)
        if num_valid_balls == 0:
            print("ERROR: No valid granular balls were created. Skipping optimization.")
            continue
        print(f"  Number of valid balls for clustering: {num_valid_balls}")

        # Adjust cluster upper bound if necessary
        original_upper_bound = parameter_bounds[0][1]
        parameter_bounds[0][1] = min(parameter_bounds[0][1], num_valid_balls)
        # Ensure lower bound is not greater than adjusted upper bound
        parameter_bounds[0][0] = min(parameter_bounds[0][0], parameter_bounds[0][1])
        if parameter_bounds[0][1] < original_upper_bound:
            print(f"  Adjusted n_clusters upper bound to {parameter_bounds[0][1]} (number of valid balls)")
        if parameter_bounds[0][0] > parameter_bounds[0][1]:
            print(
                f"ERROR: Cluster lower bound ({parameter_bounds[0][0]}) > upper bound ({parameter_bounds[0][1]}) after adjustment.")
            continue

        # === Step 3: Optimize Parameters using BKOA ===
        print(
            f"Step 3: Optimizing Parameters with BKOA (Pop={BKOA_POP_SIZE}, Iter={BKOA_MAX_ITER}, Timeout={BKOA_TIMEOUT_SECONDS}s)...")
        time_opt_start = time.time()

        optimizer = BKOAOptimizer(
            objective_function=bkoa_objective,
            bounds=parameter_bounds,
            pop_size=BKOA_POP_SIZE,
            max_iter=BKOA_MAX_ITER,
            args=(ball_dict, features, ground_truth),  # Pass necessary args to objective
            timeout=BKOA_TIMEOUT_SECONDS
        )

        best_params_continuous, best_fitness_value = optimizer.run()

        time_opt_end = time.time()
        print(f"  BKOA optimization time: {time_opt_end - time_opt_start:.2f} seconds")

        # Extract and finalize optimized parameters
        optimized_n_clusters = int(best_params_continuous[0])
        optimized_continuous_delta = float(best_params_continuous[1])

        # Final discretization of the best delta found by BKA
        snapped_best_delta = round(optimized_continuous_delta * 10.0) / 10.0
        final_optimized_delta = max(0.1, min(1.0, snapped_best_delta))

        optimized_ari_score = -best_fitness_value  # Fitness was -ARI

        print("\n--- Optimization Results ---")
        print(f"Optimized n_clusters: {optimized_n_clusters}")
        print(
            f"Optimized discrete delta: {final_optimized_delta:.1f} (from continuous: {optimized_continuous_delta:.4f})")
        print(f"Best ARI found during optimization: {optimized_ari_score:.6f}")

        # === Step 4: Final Evaluation ===
        print("\nStep 4: Final Evaluation using optimized discrete parameters...")
        final_ari, final_cluster_labels = perform_clustering_and_evaluate(
            ball_dict, optimized_n_clusters, final_optimized_delta, features, ground_truth)

        

        # Calculate NMI for the final clustering
        valid_indices_final = np.where((ground_truth != -1) & (final_cluster_labels != -1))[0]
        if len(valid_indices_final) < 2:
            final_nmi = 0.0
        else:
            try:
                # Specify average_method to avoid future warnings, 'arithmetic' is common
                final_nmi = metrics.normalized_mutual_info_score(
                    ground_truth[valid_indices_final],
                    final_cluster_labels[valid_indices_final],
                    average_method='arithmetic'
                )
            except Exception as nmi_error:
                print(f"Error calculating NMI: {nmi_error}")
                final_nmi = 0.0

        print(f"Final ARI (using optimized discrete params): {final_ari:.6f}")
        print(f"Final NMI (using optimized discrete params): {final_nmi:.6f}")

        time_total_end = time.time()
        total_duration = time_total_end - time_total_start
        print(f"\nTotal time for dataset {dataset_name}: {total_duration:.2f} seconds")
        print("-" * 40 + "\n")

        # Store results
        optimization_results[dataset_name] = {
            'optimized_n_clusters': optimized_n_clusters,
            'optimized_delta': final_optimized_delta,
            'best_optimization_ari': optimized_ari_score,
            'final_ari': final_ari,
            'final_nmi': final_nmi,
            'total_time_s': total_duration,
            'num_valid_balls': num_valid_balls
        }
        # Optional: Clear memory intensive objects if running many datasets
        del ball_dict, features, ground_truth, ball_data_list, optimizer
        gc.collect()

    # === Final Summary ===
    print("\n=== Final Optimization Results Summary ===")
    if not optimization_results:
        print("No datasets were successfully processed.")
    else:
        for name, result in optimization_results.items():
            print(f"Dataset: {name} (#Valid Balls: {result['num_valid_balls']})")
            print(
                f"  Optimized Params: n_clusters={result['optimized_n_clusters']}, delta={result['optimized_delta']:.1f}")
            print(f"  Best ARI (Optimization): {result['best_optimization_ari']:.4f}")
            print(f"  Final ARI (Evaluation):  {result['final_ari']:.4f}")
            print(f"  Final NMI (Evaluation):  {result['final_nmi']:.4f}")
            print(f"  Total Time: {result['total_time_s']:.2f}s")
            print("-" * 25)


if __name__ == '__main__':
    # Basic system info printout
    print("Starting Granular Spherical Clustering Optimization...")
    main()

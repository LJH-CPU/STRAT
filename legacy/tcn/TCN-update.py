import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from collections import defaultdict

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_data(file_path, features, target, window_size, horizon):
    """加载数据，缩放特征和目标，并创建滑动窗口。"""
    try:
        data = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError:
        print(f"错误: 在 {file_path} 未找到文件")
        raise
    except KeyError:
        print(f"错误: 在 {file_path} 中找不到 'timestamp' 列或无法解析为日期")
        raise

    missing_cols = [col for col in features + [target] if col not in data.columns]
    if missing_cols:
        print(f"错误: CSV 文件中缺少以下列: {missing_cols}")
        raise ValueError("输入文件中缺少列")

    X = data[features].values
    y = data[target].values

    if not np.issubdtype(y.dtype, np.number):
        print(f"警告: 目标列 '{target}' 不是数值类型。尝试转换。")
        try:
            y = pd.to_numeric(y, errors='coerce')
            if np.isnan(y).any():
                print(f"错误: 无法将目标列 '{target}' 中的所有值转换为数值类型。检查非数值条目。")
                raise ValueError("目标列包含非数值")
        except Exception as e:
            print(f"错误: 将目标列 '{target}' 转换为数值类型时出错: {e}")
            raise

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    def create_sliding_windows(data, target, window_size, horizon):
        Xs, ys = [], []
        if len(data) <= window_size + horizon - 1:
            print(f"错误: 数据不足（{len(data)} 行），无法创建窗口大小={window_size} 和 预测范围={horizon} 的窗口。")
            raise ValueError("数据不足，无法进行窗口化")
        for i in range(len(data) - window_size - horizon + 1):
            v = data[i:(i + window_size)]
            labels = target[(i + window_size):(i + window_size + horizon)]
            Xs.append(v)
            ys.append(labels)
        return np.array(Xs), np.array(ys)

    X_windows, y_windows = create_sliding_windows(X_scaled, y_scaled, window_size, horizon)

    if X_windows.shape[0] == 0:
        print("错误: 无法创建滑动窗口。检查数据长度、window_size 和 horizon。")
        raise ValueError("窗口创建导致零样本")

    X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=0.2, random_state=42,
                                                        shuffle=False)

    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, scaler_y, scaler_X


def load_data_with_user_ids(file_path, features, target, user_id_col, window_size, horizon):
    """
    加载数据，缩放特征和目标，创建滑动窗口，并保留用户ID信息用于留一用户交叉验证。
    
    参数:
        file_path (str): CSV文件路径
        features (list): 特征列名列表
        target (str): 目标列名
        user_id_col (str): 用户ID列名
        window_size (int): 滑动窗口大小
        horizon (int): 预测范围
        
    返回:
        dict: 包含每个用户的数据集和数据加载器
        StandardScaler: 用于目标的缩放器
        StandardScaler: 用于特征的缩放器
    """
    try:
        data = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError:
        print(f"错误: 在 {file_path} 未找到文件")
        raise
    except KeyError:
        print(f"错误: 在 {file_path} 中找不到 'timestamp' 列或无法解析为日期")
        raise

    if user_id_col not in data.columns:
        print(f"错误: CSV 文件中缺少用户ID列: {user_id_col}")
        raise ValueError("输入文件中缺少用户ID列")

    missing_cols = [col for col in features + [target] if col not in data.columns]
    if missing_cols:
        print(f"错误: CSV 文件中缺少以下列: {missing_cols}")
        raise ValueError("输入文件中缺少列")

    # 获取所有唯一用户ID
    unique_users = data[user_id_col].unique()
    if len(unique_users) < 2:
        print(f"错误: 数据中只有 {len(unique_users)} 个用户，无法进行留一用户交叉验证")
        raise ValueError("用户数量不足，无法进行留一用户交叉验证")
    
    print(f"数据中共有 {len(unique_users)} 个用户")

    X = data[features].values
    y = data[target].values
    user_ids = data[user_id_col].values

    if not np.issubdtype(y.dtype, np.number):
        print(f"警告: 目标列 '{target}' 不是数值类型。尝试转换。")
        try:
            y = pd.to_numeric(y, errors='coerce')
            if np.isnan(y).any():
                print(f"错误: 无法将目标列 '{target}' 中的所有值转换为数值类型。检查非数值条目。")
                raise ValueError("目标列包含非数值")
        except Exception as e:
            print(f"错误: 将目标列 '{target}' 转换为数值类型时出错: {e}")
            raise

    # 全局缩放器
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    def create_sliding_windows_with_user_ids(data, target, user_ids, window_size, horizon):
        Xs, ys, uids = [], [], []
        if len(data) <= window_size + horizon - 1:
            print(f"错误: 数据不足（{len(data)} 行），无法创建窗口大小={window_size} 和 预测范围={horizon} 的窗口。")
            raise ValueError("数据不足，无法进行窗口化")
            
        for i in range(len(data) - window_size - horizon + 1):
            # 确保窗口内的用户ID一致
            window_user_ids = user_ids[i:(i + window_size)]
            if len(set(window_user_ids)) != 1:
                continue  # 跳过包含多个用户的窗口
                
            v = data[i:(i + window_size)]
            labels = target[(i + window_size):(i + window_size + horizon)]
            Xs.append(v)
            ys.append(labels)
            uids.append(window_user_ids[0])  # 使用窗口的第一个用户ID
            
        return np.array(Xs), np.array(ys), np.array(uids)

    X_windows, y_windows, window_user_ids = create_sliding_windows_with_user_ids(
        X_scaled, y_scaled, user_ids, window_size, horizon
    )

    if X_windows.shape[0] == 0:
        print("错误: 无法创建滑动窗口。检查数据长度、window_size 和 horizon。")
        raise ValueError("窗口创建导致零样本")

    # 按用户ID组织数据
    user_data = defaultdict(dict)
    for uid in unique_users:
        mask = window_user_ids == uid
        user_X = X_windows[mask]
        user_y = y_windows[mask]
        
        if len(user_X) == 0:
            print(f"警告: 用户 {uid} 没有有效的窗口数据，将被跳过")
            continue
            
        user_data[uid]['X'] = user_X
        user_data[uid]['y'] = user_y
        
        # 创建数据集
        user_data[uid]['dataset'] = TimeSeriesDataset(user_X, user_y)

    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    return user_data, scaler_y, scaler_X


class LeaveOneUserOut:
    """
    留一用户交叉验证模块，用于在用户级别进行数据分割。
    
    该模块确保每次验证时保留一个用户作为测试集，其余用户作为训练集，
    支持多轮交叉验证循环。
    
    参数:
        user_data (dict): 包含每个用户数据的字典
        batch_size (int, optional): 数据加载器的批次大小，默认为32
    """
    
    def __init__(self, user_data, batch_size=32):
        """初始化留一用户交叉验证模块。"""
        self.user_data = user_data
        self.users = list(user_data.keys())
        self.batch_size = batch_size
        
        if len(self.users) < 2:
            raise ValueError("至少需要2个用户才能进行留一用户交叉验证")
            
        print(f"留一用户交叉验证将使用 {len(self.users)} 个用户")
        
    def __len__(self):
        """返回交叉验证的折数（等于用户数）。"""
        return len(self.users)
        
    def get_fold(self, test_user_idx):
        """
        获取指定折的训练和测试数据加载器。
        
        参数:
            test_user_idx (int): 测试用户的索引
            
        返回:
            DataLoader: 训练数据加载器
            DataLoader: 测试数据加载器
            str: 测试用户ID
        """
        if test_user_idx < 0 or test_user_idx >= len(self.users):
            raise ValueError(f"测试用户索引必须在0到{len(self.users)-1}之间")
            
        test_user = self.users[test_user_idx]
        train_users = [user for user in self.users if user != test_user]
        
        # 创建测试集
        test_dataset = self.user_data[test_user]['dataset']
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 合并所有训练用户的数据
        train_X = np.vstack([self.user_data[user]['X'] for user in train_users])
        train_y = np.vstack([self.user_data[user]['y'] for user in train_users])
        
        train_dataset = TimeSeriesDataset(train_X, train_y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"折 {test_user_idx+1}/{len(self.users)}: 测试用户 = {test_user}, "
              f"训练样本 = {len(train_dataset)}, 测试样本 = {len(test_dataset)}")
        
        return train_loader, test_loader, test_user
        
    def run_cross_validation(self, model_class, model_params, train_params, scaler_y):
        """
        执行完整的留一用户交叉验证。
        
        参数:
            model_class: 模型类
            model_params (dict): 模型参数
            train_params (dict): 训练参数
            scaler_y: 目标变量的缩放器
            
        返回:
            dict: 包含每个折的评估结果
        """
        results = {}
        
        for fold_idx in range(len(self)):
            print(f"\n--- 开始第 {fold_idx+1}/{len(self)} 折交叉验证 ---")
            
            # 获取当前折的数据
            train_loader, test_loader, test_user = self.get_fold(fold_idx)
            
            # 初始化模型
            model = model_class(**model_params)
            model.to(device)
            
            # 训练和评估
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=train_params.get('learning_rate', 0.001),
                weight_decay=train_params.get('weight_decay', 0.001)  # L2正则化
            )
            
            # 训练模型
            fold_results = train_and_evaluate(
                model, 
                optimizer, 
                criterion, 
                train_loader, 
                test_loader, 
                scaler_y, 
                num_epochs=train_params.get('num_epochs', 10)
            )
            
            results[test_user] = fold_results
            print(f"--- 完成第 {fold_idx+1}/{len(self)} 折交叉验证 ---")
            
        return results


class TemporalConvNet(nn.Module):
    """
    时间卷积网络 (TCN) 模块，具有可配置的正则化功能。
    
    该模块实现了带有扩张卷积的时间卷积网络，并集成了 Dropout 和 L2 正则化技术，
    以提高模型的泛化能力和防止过拟合。
    
  
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, weight_decay=0.001):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        self.weight_decay = weight_decay  # 存储 L2 正则化系数
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # 使用带有权重衰减的卷积层实现 L2 正则化
            # 注意：实际的 L2 正则化是在优化器中通过 weight_decay 参数实现的
            layers += [
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size
                ),
                nn.BatchNorm1d(out_channels),  # 添加批归一化以提高训练稳定性
                nn.ReLU(),
                nn.Dropout(dropout)  # 在每层后添加 Dropout
            ]
        self.network = nn.Sequential(*layers)
        
        print(f"TCN 初始化: dropout={dropout}, weight_decay={weight_decay}")

    def forward(self, x):
        """
        前向传播函数。
        
        参数:
            x (Tensor): 输入张量，形状为 [batch_size, num_inputs, seq_len]
            
        返回:
            Tensor: 输出张量，形状为 [batch_size, num_channels[-1], seq_len]
        """
        return self.network(x)
        
    def get_l2_regularization_loss(self):
        """
        计算模型的 L2 正则化损失。
        
        返回:
            Tensor: L2 正则化损失
        """
        l2_reg = torch.tensor(0., device=next(self.parameters()).device)
        for param in self.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param, 2)
        return self.weight_decay * l2_reg


class ChannelAttentionMechanism(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttentionMechanism, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        reduced_channels = max(1, channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        channel_attention_weights = self.fc(avg_out + max_out)

        return channel_attention_weights.unsqueeze(-1)


class TransformerBlock(nn.Module):

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        attention_output, _ = self.attention(query, key, value, attn_mask=mask)

        x = self.dropout(self.norm1(attention_output + query))

        forward_output = self.feed_forward(x)

        out = self.dropout(self.norm2(forward_output + x))
        return out


class TCNDCATransformer(nn.Module):
    def __init__(self, input_dim, window_size, num_channels, tcn_kernel_size, dca_reduction_ratio, transformer_heads,
                 transformer_dropout, transformer_forward_expansion, output_dim):
        super(TCNDCATransformer, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.tcn_output_channels = num_channels[-1]
        self.output_dim = output_dim

        self.tcn = TemporalConvNet(input_dim, num_channels, tcn_kernel_size)

        self.tcn_output_seq_len = window_size

        self.dca = ChannelAttentionMechanism(self.tcn_output_channels, dca_reduction_ratio)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.tcn_output_channels, transformer_heads, transformer_dropout,
                             transformer_forward_expansion)
            for _ in range(2)
        ])

        self.fc = nn.Linear(self.tcn_output_channels * self.tcn_output_seq_len, output_dim)

    def forward(self, x):

        x = x.transpose(1, 2)

        x_tcn = self.tcn(x)

        attention_weights = self.dca(x_tcn)
        x_dca = x_tcn * attention_weights

        x_transformer_input = x_dca.permute(0, 2, 1)

        transformer_output = x_transformer_input
        for block in self.transformer_blocks:
            transformer_output = block(transformer_output, transformer_output, transformer_output, None)

        x_flat = transformer_output.reshape(transformer_output.size(0), -1)

        if self.fc.in_features != x_flat.shape[1]:
            print(f"警告: 正在将 FC 层输入大小从 {self.fc.in_features} 调整为 {x_flat.shape[1]}")
            self.fc = nn.Linear(x_flat.shape[1], self.output_dim).to(x.device)
        x_out = self.fc(x_flat)

        return x_out


def train_and_evaluate(model, optimizer, criterion, train_loader, test_loader, scaler_y, num_epochs=10):
    model.to(device)
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f'轮次 [{epoch + 1}/{num_epochs}], 训练损失: {epoch_loss:.6f}')

        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        print(f'轮次 [{epoch + 1}/{num_epochs}], 测试损失: {avg_test_loss:.6f}')

    model.eval()
    all_predictions_scaled = []
    all_actuals_scaled = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_predictions_scaled.extend(outputs.cpu().numpy())
            all_actuals_scaled.extend(labels.cpu().numpy())

    predictions_scaled = np.array(all_predictions_scaled)
    actuals_scaled = np.array(all_actuals_scaled)

    predictions = scaler_y.inverse_transform(predictions_scaled)
    actuals = scaler_y.inverse_transform(actuals_scaled)

    if predictions.shape[1] == 1:
        predictions = predictions.flatten()
    if actuals.shape[1] == 1:
        actuals = actuals.flatten()

    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    mask = actuals != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((predictions[mask] - actuals[mask]) / actuals[mask])) * 100
    else:
        mape = np.nan

    if actuals.ndim == 1:
        r2 = 1 - (np.sum((predictions - actuals) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))
    else:

        r2_per_step = []
        for i in range(actuals.shape[1]):
            ss_res = np.sum((predictions[:, i] - actuals[:, i]) ** 2)
            ss_tot = np.sum((actuals[:, i] - np.mean(actuals[:, i])) ** 2)
            if ss_tot == 0:
                r2_step = np.nan
            else:
                r2_step = 1 - (ss_res / ss_tot)
            r2_per_step.append(r2_step)
        r2 = np.nanmean(r2_per_step)
        print(f'R² 分数 (每步): {r2_per_step}')

    print(f'\n--- 评估指标 (原始尺度) ---')
    print(f'测试损失 (缩放后 MSE): {avg_test_loss:.6f}')
    print(f'R² 分数: {r2:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'MAPE: {mape:.4f}%' if not np.isnan(mape) else 'MAPE: N/A (由于实际值中存在零)')

    results = f"测试损失 (缩放后 MSE): {avg_test_loss:.6f}\n"
    results += f"R² 分数: {r2:.4f}\n"
    results += f"RMSE: {rmse:.4f}\n"
    results += f"MAE: {mae:.4f}\n"
    results += f"MAPE: {mape:.4f}%" if not np.isnan(mape) else 'MAPE: N/A (由于实际值中存在零)' + "\n"
    if actuals.ndim > 1:
        results += f"R² 分数 (每步): {r2_per_step}\n"

    results_dir = "dataset/results"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'results.txt')
    plot_path = os.path.join(results_dir, 'prediction_vs_actual.png')
    loss_plot_path = os.path.join(results_dir, 'loss_curve.png')

    with open(results_path, 'w') as f:
        f.write(results)
    print(f"结果已保存到 {results_path}")

    plt.figure(figsize=(15, 7))

    plot_actual = actuals[:, 0] if actuals.ndim > 1 else actuals
    plot_pred = predictions[:, 0] if predictions.ndim > 1 else predictions
    plt.plot(plot_actual, label='实际值 (第一个预测步)', color='blue')
    plt.plot(plot_pred, label='预测值 (第一个预测步)', color='red', linestyle='--')
    plt.title('实际值 vs 预测值 (测试集 - 原始尺度)')
    plt.xlabel('时间步 (测试集中)')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"图表已保存到 {plot_path}")

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='训练损失')
    plt.plot(range(1, num_epochs + 1), test_losses, label='测试损失')
    plt.title('训练和测试损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失 (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path)
    print(f"损失曲线图已保存到 {loss_plot_path}")
    plt.show()


if __name__ == "__main__":

    file_path = r""  # put your url

    features = ['', '', '', '', ''] # Place tags, multiple entries allowed

    target = '' # Only one output is allowed.
    window_size = 300
    horizon = 50

    num_channels = [64, 128]
    tcn_kernel_size = 3
    dca_reduction_ratio = 8
    transformer_heads = 4
    transformer_dropout = 0.1
    transformer_forward_expansion = 4
    num_epochs = 20
    learning_rate = 0.01

    input_dim = len(features)
    output_dim = horizon

    print("--- 开始数据加载 ---")
    try:

        train_loader, test_loader, scaler_y, _ = load_data(file_path, features, target, window_size, horizon)
        print("--- 数据加载成功 ---")

        print("--- 初始化模型 ---")
        model = TCNDCATransformer(
            input_dim=input_dim,
            window_size=window_size,
            num_channels=num_channels,
            tcn_kernel_size=tcn_kernel_size,
            dca_reduction_ratio=dca_reduction_ratio,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
            transformer_forward_expansion=transformer_forward_expansion,
            output_dim=output_dim
        )
        model.to(device)
        print(f"模型已使用 {input_dim} 个输入特征进行初始化。")
        print(model)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print("--- 开始训练和评估 ---")
        train_and_evaluate(model, optimizer, criterion, train_loader, test_loader, scaler_y, num_epochs=num_epochs)
        print("--- 训练和评估完成 ---")

    except FileNotFoundError:
        print(f"严重错误: 在 {file_path} 未找到输入文件。请检查路径。")
    except ValueError as ve:
        print(f"严重错误: 在数据处理或模型设置过程中出错: {ve}")
    except Exception as e:
        print(f"发生意外的严重错误: {e}")

        import traceback

        traceback.print_exc()

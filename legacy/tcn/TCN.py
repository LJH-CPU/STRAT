import matplotlib
matplotlib.use('Agg')

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
# 创建滑动窗口数据
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

# PyTorch数据集包装器
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

# 时间卷积网络，实现扩张卷积
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                 padding=(kernel_size - 1) * dilation_size),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 通道注意力机制，增强重要特征
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

# Transformer自注意力块
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

# 整合TCN、DCA和Transformer的混合模型
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

# 训练模型并评估性能
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

    with open(results_path, 'w', encoding='utf-8') as f:
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

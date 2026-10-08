#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from TCN import TCNDCATransformer, load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_classification_data(file_path, features, target, window_size):
    data = pd.read_csv(file_path)
    X = data[features].values
    y = data[target].values

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    def create_sliding_windows(data, target, window_size):
        Xs, ys = [], []
        for i in range(len(data) - window_size):
            v = data[i : (i + window_size)]
            label = target[i + window_size]
            Xs.append(v)
            ys.append(label)
        return np.array(Xs), np.array(ys)

    X_windows, y_windows = create_sliding_windows(X_scaled, y, window_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X_windows, y_windows, test_size=0.2, random_state=42, shuffle=False
    )

    return X_train, X_test, y_train, y_test, scaler_X


def train_and_evaluate_classification(
    model, optimizer, criterion, train_loader, test_loader, num_epochs=20
):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).long()
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).long()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        accuracy = 100 * correct / total
        print(
            f"轮次 [{epoch+1}/{num_epochs}], 训练损失: {train_loss:.4f}, 测试损失: {test_loss:.4f}, 准确率: {accuracy:.2f}%"
        )

    print(f"\n最终测试准确率: {accuracy:.2f}%")
    return accuracy


file_path = os.path.join(script_dir, "predict_data", "青城山_with_cluster.csv")
features = ["longitude", "latitude", "altitude", "speed"]
target = "cluster_label"
window_size = 10
num_classes = 5

num_channels = [64, 128]
tcn_kernel_size = 3
dca_reduction_ratio = 8
transformer_heads = 4
transformer_dropout = 0.1
transformer_forward_expansion = 4
num_epochs = 5
learning_rate = 0.001

print("--- 完整版TCN-DCA-Transformer 分类测试 ---")
print(f"特征: {features}")
print(f"预测目标: {target} (共{num_classes}类)")
print(f"窗口大小: {window_size}")
print(f"模型: TCN + DCA + Transformer")
print("-" * 50)

X_train, X_test, y_train, y_test, scaler_X = load_classification_data(
    file_path, features, target, window_size
)
print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
print(f"类别分布: {np.bincount(y_train.astype(int))}")

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_dim = len(features)
model = TCNDCATransformer(
    input_dim=input_dim,
    window_size=window_size,
    num_channels=num_channels,
    tcn_kernel_size=tcn_kernel_size,
    dca_reduction_ratio=dca_reduction_ratio,
    transformer_heads=transformer_heads,
    transformer_dropout=transformer_dropout,
    transformer_forward_expansion=transformer_forward_expansion,
    output_dim=num_classes,
)
model.to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("\n--- 开始训练 ---")
accuracy = train_and_evaluate_classification(
    model, optimizer, criterion, train_loader, test_loader, num_epochs=num_epochs
)

print("\n--- 评估完成 ---")
print(f"最终准确率: {accuracy:.2f}%")
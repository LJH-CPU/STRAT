#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 数据合并脚本 ，用于将轨迹数据与聚类标签合并。
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

predict_file = os.path.join(script_dir, "predict_data", "青城山_for_prediction.csv")
labels_file = os.path.join(script_dir, "..", "result", "cluster_labels_青城山_自动标签_20260413_200333.csv")
output_file = os.path.join(script_dir, "predict_data", "青城山_with_cluster.csv")

print("读取预测数据...")
df_predict = pd.read_csv(predict_file)
print(f"预测数据: {len(df_predict)} 行, 轨迹数: {df_predict['track_id'].nunique()}")

print("\n读取聚类标签...")
df_labels = pd.read_csv(labels_file)
print(f"聚类标签: {len(df_labels)} 条")

df_labels.columns = ['track_id', 'cluster_label']

print("\n合并数据...")
df_merged = df_predict.merge(df_labels, on='track_id', how='left')

print("\n检查合并结果...")
print(f"合并后数据: {len(df_merged)} 行")
print(f"聚类标签分布:")
print(df_merged['cluster_label'].value_counts().sort_index())

print("\n保存合并后的数据...")
df_merged.to_csv(output_file, index=False)
print(f"已保存到: {output_file}")

print("\n合并后的数据预览:")
print(df_merged.head(10))
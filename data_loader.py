import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler  # 导入 SMOTE 和 RandomOverSampler
from imblearn.pipeline import Pipeline  # 导入 Pipeline 用于组合
import numpy as np
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler

class TableDataset(Dataset):
    def __init__(self, file_path):
        # 读取表格数据
        df = pd.read_csv(file_path, header=None)  # 无表头

        # 提取数据 (从第二行、第二列开始，到倒数第二列)
        self.features = df.iloc[1:, 1:-1].values

        # 提取标签 (从第二行开始，最后一列)
        self.labels = df.iloc[1:, -1].values

        # 转换数据类型，处理异常值
        self.features = self.clean_data(self.features)
        if self.labels is not None:
            self.labels = self.clean_data(self.labels, is_label=True)

        # 标准化特征（均值为 0，标准差为 1）
        self.features = self.standardize_features(self.features)

    def clean_data(self, data, is_label=False):
        """ 处理 NaN 和 Infinity，并确保标签是整数 """
        data = np.array(data, dtype=np.float32)  # 转换为 float32，避免数据异常
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)  # 处理 NaN 和无穷大
        if is_label:
            data = data.astype(int)  # 确保标签是整数（分类任务）
        return data

    def standardize_features(self, features):
        """标准化特征到均值为 0，标准差为 1"""
        self.mean = np.mean(features, axis=0, keepdims=True)  # 按特征维度计算均值
        self.std = np.std(features, axis=0, keepdims=True)  # 按特征维度计算标准差
        # 避免除以 0，std 为 0 的特征保持不变
        features_standardized = (features - self.mean) / (self.std + 1e-8)
        return features_standardized

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.int64)  # 分类任务用 int64
            return x, y
        return x


# ROS + SMOTE 混合采样
def load_training_with_oversampling(root_path, file_name, batch_size, kwargs):
    print(f"Loading dataset from: {root_path + file_name}")
    try:
        dataset = TableDataset(root_path + file_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # 获取特征和标签
    X = dataset.features
    y = dataset.labels

    # 检查 NaN 和 Infinity
    nan_rows_X = np.isnan(X).any(axis=1)
    nan_rows_y = np.isnan(y) if y is not None else np.array([])
    inf_rows_X = np.isinf(X).any(axis=1)
    inf_rows_y = np.isinf(y) if y is not None else np.array([])

    # 打印有问题的数据索引
    print("NaN in X at rows:", np.where(nan_rows_X)[0])
    print("NaN in y at index:", np.where(nan_rows_y)[0])
    print("Infinity in X at rows:", np.where(inf_rows_X)[0])
    print("Infinity in y at index:", np.where(inf_rows_y)[0])

    # 确保 y 是整数
    if y is not None:  # Source domain with labels
        y = y.astype(int)
        # 使用 Pipeline 组合 RUS 和 SMOTE
        pipeline = Pipeline([
            ('ros', RandomOverSampler(sampling_strategy=0.5, random_state=42)),  # 少数类增加到多数类的 50%
            ('smote', SMOTE(sampling_strategy=1.0, random_state=42))  # 再将少数类增加到与多数类相等
        ])
        try:
            X_resampled, y_resampled = pipeline.fit_resample(X, y)
        except ValueError as e:
            print(f"Error in RUS+SMOTE sampling: {e}")
            return None

        y_resampled = y_resampled.astype(int)
        print("Original class distribution:", np.bincount(y))
        print("Resampled class distribution:", np.bincount(y_resampled))

        resampled_dataset = TensorDataset(
            torch.tensor(X_resampled, dtype=torch.float32),
            torch.tensor(y_resampled, dtype=torch.int64)
        )
    else:  # Target domain without labels
        resampled_dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32)
        )

    # Create DataLoader for both cases
    train_loader = DataLoader(resampled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    print(f"Created DataLoader with length: {len(train_loader.dataset)}")
    return train_loader

# 加载测试集
def load_testing(root_path, file_name, batch_size, kwargs):
    dataset = TableDataset(root_path + file_name)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader
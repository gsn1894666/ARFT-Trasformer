import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import numpy as np


# 全局统计量计算函数
def compute_global_stats(root_path, src_file, tgt_file):
    """
    计算源域和目标域所有数据的全局均值和标准差
    Args:
        root_path (str): 数据文件路径
        src_file (str): 源域文件名（如 'mysql.csv'）
        tgt_file (str): 目标域文件名（如 'httpd.csv'）
    Returns:
        global_mean (np.ndarray): 全局均值
        global_std (np.ndarray): 全局标准差
    """
    # 加载源域和目标域数据（不标准化）
    src_dataset = TableDataset(root_path + src_file, global_mean=None, global_std=None)
    tgt_dataset = TableDataset(root_path + tgt_file, global_mean=None, global_std=None)

    # 合并特征
    all_features = np.vstack([src_dataset.features, tgt_dataset.features])

    # 计算全局均值和标准差
    global_mean = np.mean(all_features, axis=0, keepdims=True)
    global_std = np.std(all_features, axis=0, keepdims=True)

    return global_mean, global_std


class TableDataset(Dataset):
    def __init__(self, file_path,global_mean=None, global_std=None):
        # 读取表格数据
        df = pd.read_csv(file_path, header=None)  # 无表头

        # 提取数据 (从第二行、第二列开始，到倒数第二列)
        self.features = df.iloc[1:, 1:-1].values

        # 提取标签 (从第二行开始，最后一列)
        self.labels = df.iloc[1:, -1].values

        # 转换数据类型，处理异常值
        self.features = self.clean_data(self.features)
        self.labels = self.clean_data(self.labels, is_label=True)

        # 标准化特征
        if global_mean is not None and global_std is not None:
            # 使用全局均值和标准差标准化
            self.features = self.standardize_features(self.features, global_mean, global_std)
        else:
            # 使用分别标准化（每个数据集独立）
            self.features = self.standardize_features(self.features)


    def clean_data(self, data, is_label=False):
        """ 处理 NaN 和 Infinity，并确保标签是整数 """
        data = np.array(data, dtype=np.float32)  # 转换为 float32，避免数据异常
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)  # 处理 NaN 和无穷大
        if is_label:
            data = data.astype(int)  # 确保标签是整数（分类任务）
        return data

    def standardize_features(self, features, global_mean=None, global_std=None):
        """标准化特征到均值为 0，标准差为 1"""
        if global_mean is not None and global_std is not None:
            # 全局标准化
            features_standardized = (features - global_mean) / (global_std + 1e-8)
        else:
            # 分别标准化
            self.mean = np.mean(features, axis=0, keepdims=True)
            self.std = np.std(features, axis=0, keepdims=True)
            features_standardized = (features - self.mean) / (self.std + 1e-8)
        return features_standardized


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.int64)  # 分类任务用 int64
        return x, y  # 返回特征和标签


# 加载训练集，仅使用 ROS 过采样
def load_training_with_ros(root_path, file_name, batch_size, kwargs, global_mean=None, global_std=None):
    """
    加载训练集，仅使用 ROS 过采样
    """
    dataset = TableDataset(root_path + file_name, global_mean, global_std)
    X = dataset.features
    y = dataset.labels

    # 检查 NaN 和 Infinity
    nan_rows_X = np.isnan(X).any(axis=1)
    nan_rows_y = np.isnan(y)
    inf_rows_X = np.isinf(X).any(axis=1)
    inf_rows_y = np.isinf(y)
    print("NaN in X at rows:", np.where(nan_rows_X)[0])
    print("NaN in y at index:", np.where(nan_rows_y)[0])
    print("Infinity in X at rows:", np.where(inf_rows_X)[0])
    print("Infinity in y at index:", np.where(inf_rows_y)[0])

    y = y.astype(int)
    print("采样前的类别分布:", np.bincount(y))

    # 使用 ROS 过采样
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    print("ROS 采样后的类别分布:", np.bincount(y_resampled))

    y_resampled = y_resampled.astype(int)
    resampled_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_resampled, dtype=torch.float32),
        torch.tensor(y_resampled, dtype=torch.int64)
    )
    train_loader = DataLoader(resampled_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_raw_training(root_path, file_name, batch_size, kwargs, global_mean=None, global_std=None):
    """
    加载未过采样的训练数据
    """
    dataset = TableDataset(root_path + file_name, global_mean, global_std)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)

def load_testing(root_path, file_name, batch_size, kwargs, global_mean=None, global_std=None):
    """
    加载测试集
    """
    dataset = TableDataset(root_path + file_name, global_mean, global_std)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler

class TableDataset(Dataset):
    def __init__(self, file_path):
        # Reading table data
        df = pd.read_csv(file_path, header=None)  #No header

        # extract data
        self.features = df.iloc[1:, 1:-1].values

        # extract label
        self.labels = df.iloc[1:, -1].values

        # data type
        self.features = self.clean_data(self.features)
        if self.labels is not None:
            self.labels = self.clean_data(self.labels, is_label=True)

        # standardize
        self.features = self.standardize_features(self.features)

    def clean_data(self, data, is_label=False):
        """ make sure label is Integer """
        data = np.array(data, dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if is_label:
            data = data.astype(int)
        return data

    def standardize_features(self, features):
        """standardize feature"""
        self.mean = np.mean(features, axis=0, keepdims=True)
        self.std = np.std(features, axis=0, keepdims=True)
        # Avoid division by zero
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


# ROS
def load_training_with_oversampling(root_path, file_name, batch_size, kwargs):
    print(f"Loading dataset from: {root_path + file_name}")
    try:
        dataset = TableDataset(root_path + file_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Get features and labels
    X = dataset.features
    y = dataset.labels

    # Checking for NaN and Infinity
    nan_rows_X = np.isnan(X).any(axis=1)
    nan_rows_y = np.isnan(y) if y is not None else np.array([])
    inf_rows_X = np.isinf(X).any(axis=1)
    inf_rows_y = np.isinf(y) if y is not None else np.array([])

    # Make sure y is an integer
    if y is not None:  # Source domain with labels
        y = y.astype(int)
        ros = RandomOverSampler(sampling_strategy=0.5, random_state=42)

        try:
            X_resampled, y_resampled = ros.fit_resample(X, y)
        except ValueError as e:
            print(f"Error in ROS sampling: {e}")
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

# load test dataset
def load_testing(root_path, file_name, batch_size, kwargs):
    dataset = TableDataset(root_path + file_name)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader
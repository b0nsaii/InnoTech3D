# data_loader.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def load_and_clean_csv(file_path):
    try:
        df = pd.read_csv(file_path, sep=',')
        df.columns = [col.strip() for col in df.columns]
        return df
    except Exception as e:
        print(f"Error loading and cleaning CSV from {file_path}: {e}")
        return pd.DataFrame()

def load_point_cloud(file_path):
    try:
        point_cloud = np.loadtxt(file_path, delimiter=' ')
        if point_cloud.ndim == 1:
            point_cloud = point_cloud.reshape(1, -1)
    except Exception as e:
        print(f"Error loading point cloud from {file_path}: {e}")
        point_cloud = np.empty((0, 7))  # 空数组，形状为 (0, 7)
    return point_cloud  # 形状: (N, 7)

def normalize_point_cloud(pc):
    if pc.shape[0] == 0:
        return pc
    min_vals = pc.min(axis=0)
    max_vals = pc.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1e-8
    normalized_pc = (pc - min_vals) / ranges
    return normalized_pc

def load_bounding_boxes(csv_path):
    try:
        df = load_and_clean_csv(csv_path)
        if df.empty:
            print(f"Warning: Loaded bounding boxes DataFrame is empty for {csv_path}")
        else:
            print(f"Loaded CSV file: {csv_path}")
            print(f"Columns: {df.columns.tolist()}")
            print(df.head())
            unique_labels_in_csv = df['Label'].unique()
            print(f"CSV 文件中的唯一标签：{unique_labels_in_csv}")
    except Exception as e:
        print(f"Error loading bounding boxes from {csv_path}: {e}")
        df = pd.DataFrame()
    return df

def assign_labels_to_points(point_cloud, bbox_df, label_mapping):
    labels = np.zeros(point_cloud.shape[0], dtype=int)  # 背景标签为0

    for _, row in bbox_df.iterrows():
        label = row.get('Label')
        if pd.isnull(label):
            print("Error: 'Label' 列不存在于 CSV 文件中。")
            print(f"Available columns: {bbox_df.columns.tolist()}")
            continue
        label = label.strip().lower()  # 标准化标签名称
        min_x, min_y, min_z = row['BB.Min.X'], row['BB.Min.Y'], row['BB.Min.Z']
        max_x, max_y, max_z = row['BB.Max.X'], row['BB.Max.Y'], row['BB.Max.Z']

        # 创建掩码
        mask = (
            (point_cloud[:, 0] >= min_x) & (point_cloud[:, 0] <= max_x) &
            (point_cloud[:, 1] >= min_y) & (point_cloud[:, 1] <= max_y) &
            (point_cloud[:, 2] >= min_z) & (point_cloud[:, 2] <= max_z)
        )
        # 赋值标签，使用label_mapping映射标签
        mapped_label = label_mapping.get(label, 0)
        labels[mask] = mapped_label

    unique_labels = np.unique(labels)
    print(f"分配的唯一标签：{unique_labels}")
    return labels

class PointCloudDataset(Dataset):
    def __init__(self, project_paths, label_mapping, transforms=None, num_points=1024):
        self.file_paths = []
        self.transforms = transforms
        self.label_mapping = label_mapping
        self.num_points = num_points

        for project, path in project_paths.items():
            if not os.path.isdir(path):
                print(f"Warning: Project path {path} is not a directory.")
                continue

            # 获取所有 .xyz 和 .csv 文件
            xyz_files = sorted([
                os.path.join(path, f) for f in os.listdir(path)
                if f.endswith('.xyz')
            ])
            csv_files = sorted([
                os.path.join(path, f) for f in os.listdir(path)
                if f.endswith('.csv')
            ])

            # 确保每个 .xyz 文件有对应的 .csv 文件
            for xyz in xyz_files:
                base_name = os.path.splitext(os.path.basename(xyz))[0]
                corresponding_csv = os.path.join(path, f"{base_name}.csv")
                if corresponding_csv in csv_files:
                    self.file_paths.append((xyz, corresponding_csv))
                else:
                    print(f"Warning: No corresponding CSV file for {xyz}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        xyz_path, csv_path = self.file_paths[idx]
        pc = load_point_cloud(xyz_path)
        bbox_df = load_bounding_boxes(csv_path)
        labels = assign_labels_to_points(pc, bbox_df, self.label_mapping)

        # 在归一化之前分配标签
        pc = normalize_point_cloud(pc)

        # 确保点云数量一致
        if pc.shape[0] > self.num_points:
            indices = np.random.choice(pc.shape[0], self.num_points, replace=False)
            pc = pc[indices]
            labels = labels[indices]
        elif pc.shape[0] < self.num_points:
            padding_size = self.num_points - pc.shape[0]
            padding = np.zeros((padding_size, pc.shape[1]), dtype=np.float32)
            pc = np.vstack((pc, padding))
            labels = np.concatenate((labels, np.zeros(padding_size, dtype=int)))

        # 转置点云数据，确保形状为 [7, num_points]
        pc = pc.T

        # 转换为张量
        pc = torch.tensor(pc, dtype=torch.float32)  # 形状：[7, num_points]
        labels = torch.tensor(labels, dtype=torch.long)  # 形状：[num_points]

        return pc, labels

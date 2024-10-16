import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

point_cloud_file = 'Project1.xyz'
bounding_box_file = 'Project1.csv'

# 加载点云数据和 bounding box 数据
# Load point cloud data and bounding box data
points = pd.read_csv(point_cloud_file, header=None, names=['x', 'y', 'z', 'i', 'r', 'g', 'b'])
boxes = pd.read_csv(bounding_box_file)

# 检查列名，去除可能的空格
# Check column names and remove any potential spaces
print("Columns before stripping:", boxes.columns.tolist())
boxes.columns = boxes.columns.str.strip()
print("Columns after stripping:", boxes.columns.tolist())

# 将涉及到比较的列转换为数值类型
# Convert columns involved in comparisons to numeric types
points[['x', 'y', 'z', 'i', 'r', 'g', 'b']] = points[['x', 'y', 'z', 'i', 'r', 'g', 'b']].apply(pd.to_numeric,
                                                                                                errors='coerce')
boxes[['BB.Min.X', 'BB.Min.Y', 'BB.Min.Z', 'BB.Max.X', 'BB.Max.Y', 'BB.Max.Z']] = boxes[
    ['BB.Min.X', 'BB.Min.Y', 'BB.Min.Z', 'BB.Max.X', 'BB.Max.Y', 'BB.Max.Z']].apply(pd.to_numeric, errors='coerce')

# 处理缺失值（可选，视情况而定）
# Handle missing values (optional, depending on the situation)
points = points.dropna()
boxes = boxes.dropna()

# 定义一个函数来获取每个 bounding box 内的点并提取特征
# Define a function to get the points within each bounding box and extract features
def extract_point_features(box, points):
    # 筛选出位于该 bounding box 内的所有点
    # Filter points that are inside the bounding box
    mask = (
            (points['x'] >= box['BB.Min.X']) & (points['x'] <= box['BB.Max.X']) &
            (points['y'] >= box['BB.Min.Y']) & (points['y'] <= box['BB.Max.Y']) &
            (points['z'] >= box['BB.Min.Z']) & (points['z'] <= box['BB.Max.Z'])
    )
    selected_points = points[mask]

    # 如果没有点，则返回空特征
    # If there are no points, return empty features
    if len(selected_points) == 0:
        return pd.Series([0] * 10)

    # 提取统计特征
    # Extract statistical features
    feature_vector = [
        len(selected_points),  # 点的数量
        # Number of points
        selected_points[['x', 'y', 'z']].mean().values,  # 平均位置 (x, y, z)
        # Average position (x, y, z)
        selected_points[['x', 'y', 'z']].std().values,  # 标准差 (x, y, z)
        # Standard deviation (x, y, z)
        selected_points['i'].mean(),  # 强度均值
        # Average intensity
        selected_points[['r', 'g', 'b']].mean().values,  # 颜色均值 (r, g, b)
        # Average color (r, g, b)
    ]
    return pd.Series(np.concatenate(feature_vector))

# 为每个 bounding box 提取点云特征
# Extract point cloud features for each bounding box
point_features = boxes.apply(lambda row: extract_point_features(row, points), axis=1)

# 合并 bounding box 几何特征和点云特征
# Merge bounding box geometric features and point cloud features
boxes['length'] = boxes['BB.Max.X'] - boxes['BB.Min.X']
boxes['width'] = boxes['BB.Max.Y'] - boxes['BB.Min.Y']
boxes['height'] = boxes['BB.Max.Z'] - boxes['BB.Min.Z']
combined_features = pd.concat([boxes[['length', 'width', 'height']], point_features], axis=1)

# 将列名转换为字符串类型，以确保 scikit-learn 兼容性
# Convert column names to string type to ensure scikit-learn compatibility
combined_features.columns = combined_features.columns.astype(str)

# 准备模型的输入和输出
# Prepare inputs and outputs for the model
X = combined_features
y = boxes['Label']

# 划分训练集和测试集
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用随机森林分类器
# Use Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 预测和评估
# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


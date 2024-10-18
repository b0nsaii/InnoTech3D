import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

point_cloud_file = 'data/TrainingSet/Project1/Project1.xyz'
bounding_box_file = 'data/TrainingSet/Project1/Project1.csv'

# 加载点云数据和 bounding box 数据
# Load point cloud data and bounding box data
points = pd.read_csv(point_cloud_file, header=None, names=['x', 'y', 'z', 'i', 'r', 'g', 'b'], delimiter=' ')
boxes = pd.read_csv(bounding_box_file)

# 检查列名，去除可能的空格
# Check column names and remove any potential spaces
print("Columns before stripping:", boxes.columns.tolist())
boxes.columns = boxes.columns.str.strip()
print("Columns after stripping:", boxes.columns.tolist())

# 将涉及到比较的列转换为数值类型
# Convert columns involved in comparisons to numeric type
points[['x', 'y', 'z', 'i', 'r', 'g', 'b']] = points[['x', 'y', 'z', 'i', 'r', 'g', 'b']].apply(pd.to_numeric, errors='coerce')
boxes[['BB.Min.X', 'BB.Min.Y', 'BB.Min.Z', 'BB.Max.X', 'BB.Max.Y', 'BB.Max.Z']] = boxes[
    ['BB.Min.X', 'BB.Min.Y', 'BB.Min.Z', 'BB.Max.X', 'BB.Max.Y', 'BB.Max.Z']].apply(pd.to_numeric, errors='coerce')

# 处理缺失值（可选，视情况而定）
# Handle missing values (optional, depending on the situation)
points = points.dropna()
boxes = boxes.dropna()


# 定义一个函数来获取每个 bounding box 内的点并提取特征
# Define a function to get the points within each bounding box and extract features
def extract_point_features(box, points):
    arrayofobjects = []

    # 筛选出位于该 bounding box 内的所有点
    # Filter points that are inside the bounding box
    mask = (
            (points['x'] >= box['BB.Min.X']) & (points['x'] <= box['BB.Max.X']) &
            (points['y'] >= box['BB.Min.Y']) & (points['y'] <= box['BB.Max.Y']) &
            (points['z'] >= box['BB.Min.Z']) & (points['z'] <= box['BB.Max.Z'])
    )
    selected_points = points[mask]
    id = str(box['ID']) 
    xyzfile = open(id + ".xyz", "w")

    # Write the selected points to the file
    selected_points.to_csv(xyzfile, sep=' ', index=False, header=False)
    xyzfile.close()

point_features = boxes.apply(lambda row: extract_point_features(row, points), axis=1)




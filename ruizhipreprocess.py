import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 定义项目文件的列表
projects = ['Project1', 'Project2', 'Project3', 'Project4']

# 初始化空的 DataFrame 用来存放所有项目的数据
all_points = pd.DataFrame()
all_boxes = pd.DataFrame()

# 循环加载每个项目的数据并合并
for project in projects:
    point_cloud_file = f'{project}.xyz'
    bounding_box_file = f'{project}.csv'

    print(f"Loading {project}...")

    # 加载点云数据和 bounding box 数据
    points = pd.read_csv(point_cloud_file, header=None, names=['x', 'y', 'z', 'i', 'r', 'g', 'b'], delimiter=' ')
    boxes = pd.read_csv(bounding_box_file)

    # 检查列名，去除可能的空格
    boxes.columns = boxes.columns.str.strip()

    # 将涉及到比较的列转换为数值类型
    points[['x', 'y', 'z', 'i', 'r', 'g', 'b']] = points[['x', 'y', 'z', 'i', 'r', 'g', 'b']].apply(pd.to_numeric, errors='coerce')
    boxes[['BB.Min.X', 'BB.Min.Y', 'BB.Min.Z', 'BB.Max.X', 'BB.Max.Y', 'BB.Max.Z']] = boxes[
        ['BB.Min.X', 'BB.Min.Y', 'BB.Min.Z', 'BB.Max.X', 'BB.Max.Y', 'BB.Max.Z']].apply(pd.to_numeric, errors='coerce')

    # 处理缺失值
    points = points.dropna()
    boxes = boxes.dropna()

    # 将每个项目的数据追加到总的 DataFrame 中
    all_points = pd.concat([all_points, points], ignore_index=True)
    all_boxes = pd.concat([all_boxes, boxes], ignore_index=True)

print("All projects loaded and merged.")

# 定义一个函数来获取每个 bounding box 内的点并提取特征
def extract_point_features(box, points):
    mask = (
            (points['x'] >= box['BB.Min.X']) & (points['x'] <= box['BB.Max.X']) &
            (points['y'] >= box['BB.Min.Y']) & (points['y'] <= box['BB.Max.Y']) &
            (points['z'] >= box['BB.Min.Z']) & (points['z'] <= box['BB.Max.Z'])
    )
    selected_points = points[mask]

    pointmatrix = np.array([selected_points['x'], selected_points['y'], selected_points['z']])
    rotatingmatrix = np.random.rand(3, 3)

    np.matmul(pointmatrix, rotatingmatrix)
        

    if len(selected_points) == 0:
        return pd.Series([0] * 10)

    num_points = len(selected_points)
    mean_position = selected_points[['x', 'y', 'z']].mean().values.flatten()
    std_position = selected_points[['x', 'y', 'z']].std().values.flatten()
    intensity_mean = selected_points['i'].mean()
    color_mean = selected_points[['r', 'g', 'b']].mean().values.flatten()

    feature_vector = [num_points, *mean_position, *std_position, intensity_mean, *color_mean]

    return pd.Series(feature_vector)

# 为每个 bounding box 提取点云特征
point_features = all_boxes.apply(lambda row: extract_point_features(row, all_points), axis=1)

# 合并 bounding box 几何特征和点云特征
all_boxes['length'] = all_boxes['BB.Max.X'] - all_boxes['BB.Min.X']
all_boxes['width'] = all_boxes['BB.Max.Y'] - all_boxes['BB.Min.Y']
all_boxes['height'] = all_boxes['BB.Max.Z'] - all_boxes['BB.Min.Z']
combined_features = pd.concat([all_boxes[['length', 'width', 'height']], point_features], axis=1)

# 将列名转换为字符串类型，以确保 scikit-learn 兼容性
combined_features.columns = combined_features.columns.astype(str)

# 准备模型的输入和输出
X = combined_features
y = all_boxes['Label']

# 从所有项目的数据中划分训练集和测试集，80%训练，20%测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 预测和评估
y_pred = clf.predict(X_test)

# 输出所有测试集里的预测值和实际值
test_results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

# 打印结果
print(test_results)

# 输出分类报告
print(classification_report(y_test, y_pred))

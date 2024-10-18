import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_loader import PointCloudDataset  # 确保data_loader.py与train.py在同一目录下
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 定义 PointNetSegmentation 模型（不变，和你的版本一致）
class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes=5):
        super(PointNetSegmentation, self).__init__()
        self.num_classes = num_classes

        # PointNet Encoder
        self.conv1 = nn.Conv1d(7, 64, 1)  # 将输入通道数从 6 修改为 7，因为点云有 7 个属性
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        # Fully connected layers for global features
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)

        # Decoder for per-point segmentation
        self.conv4 = nn.Conv1d(1152, 512, 1)
        self.bn7 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.bn8 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.bn9 = nn.BatchNorm1d(128)
        self.conv7 = nn.Conv1d(128, self.num_classes, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Input x: [B, 7, N]
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        x = self.relu(self.bn2(self.conv2(x)))  # [B, 128, N]
        x = self.relu(self.bn3(self.conv3(x)))  # [B, 1024, N]

        # Global feature
        x_global = torch.max(x, 2, keepdim=False)[0]  # [B, 1024]

        # Fully connected layers
        x_global = self.relu(self.bn4(self.fc1(x_global)))  # [B, 512]
        x_global = self.dropout1(x_global)
        x_global = self.relu(self.bn5(self.fc2(x_global)))  # [B, 256]
        x_global = self.dropout2(x_global)
        x_global = self.relu(self.bn6(self.fc3(x_global)))  # [B, 128]

        # Repeat global features for each point
        x_global = x_global.unsqueeze(2).repeat(1, 1, x.size(2))  # [B, 128, N]

        # Concatenate global and local features
        x = torch.cat([x, x_global], dim=1)  # [B, 1152, N]

        # Decoder
        x = self.relu(self.bn7(self.conv4(x)))  # [B, 512, N]
        x = self.relu(self.bn8(self.conv5(x)))  # [B, 256, N]
        x = self.relu(self.bn9(self.conv6(x)))  # [B, 128, N]
        x = self.conv7(x)  # [B, num_classes, N]

        return x


def main():
    # 定义标签映射
    label_mapping = {
        'HVAC_Duct': 1,
        'Structural_IBeam': 2,
        'Structural_ColumnBeam': 3,
        'Pipe': 4,

        # 如果 CSV 中有其他标签，添加到这里
    }

    # 定义项目路径映射
    project_paths = {
        'Project1': '/Users/laolao/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/course/perception/InnoTech3D-master/data/TrainingSet/Project1',
        'Project2': '/Users/laolao/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/course/perception/InnoTech3D-master/data/TrainingSet/Project2',
        'Project3': '/Users/laolao/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/course/perception/InnoTech3D-master/data/TrainingSet/Project3'
    }

    # 初始化数据集
    dataset = PointCloudDataset(project_paths, label_mapping, num_points=1024)

    # 统计标签分布
    all_labels = []
    for idx in range(len(dataset)):
        _, labels = dataset[idx]
        all_labels.extend(labels.numpy())

    unique_labels, counts = np.unique(all_labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    print(f"数据集中各类别的样本数量：{label_counts}")

    # 划分训练集和验证集
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # 初始化模型
    model = PointNetSegmentation(num_classes=len(label_mapping) + 1)  # +1 for background
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 定义损失函数和优化器
    class_weights = torch.tensor([1.0 if i in label_counts else 0 for i in range(len(label_mapping) + 1)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (points, labels) in enumerate(train_loader):
            points = points.to(device)  # [B, 7, 1024]
            labels = labels.to(device)  # [B, 1024]

            optimizer.zero_grad()

            # Forward pass
            outputs = model(points)  # [B, num_classes, 1024]

            # Reshape outputs and labels for loss computation
            outputs = outputs.permute(0, 2, 1).contiguous()  # [B, 1024, num_classes]
            outputs = outputs.view(-1, model.num_classes)  # [B*1024, num_classes]
            labels = labels.view(-1)  # [B*1024]

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")

        # 保存模型
        torch.save(model.state_dict(), f'pointnet_segmentation_epoch_{epoch + 1}.pth')

        # 验证模型
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for points, labels in val_loader:
                points = points.to(device)
                labels = labels.to(device)

                outputs = model(points)
                outputs = outputs.permute(0, 2, 1).contiguous()
                preds = outputs.view(-1, model.num_classes).argmax(dim=1)
                labels = labels.view(-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        print(f"Epoch [{epoch + 1}/{num_epochs}] 验证集混淆矩阵：\n{cm}")

    print("Training completed.")

if __name__ == "__main__":
    main()
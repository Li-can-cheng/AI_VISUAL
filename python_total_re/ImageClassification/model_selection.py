import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import random

def CNN_model(x_train, y_train, num_class, epoch=1, lr=0.001, batch_size=64):
    """
    搭建CNN(卷积神经网络)模型
    ... ...
    参数：
    x_train: 传入的images_list
    y_train: 传入的labels_list


    返回：
    训练好的网络模型
    """

    # 使用这个网络要经过的必要形状转化
    x_train_tensor = torch.tensor(x_train).float().unsqueeze(1).view(-1, 1, 28, 28)
    y_train_tensor = torch.tensor(y_train).long()

    # 定义卷积神经网络模型
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.fc = nn.Linear(32 * 7 * 7, num_class)  # 次数为n类

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    # 创建模型实例、定义损失函数和优化器
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)

    # 训练模型
    for epo in range(epoch):
        running_loss = 0.0
        correct = 0
        total = 0
        # 假设 x_train 和 y_train 已经是正确的张量
        dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return model

def train_mlp_model(x_train, y_train, json_config, epochs=5):
    """
    训练一个多层感知器模型。

    参数:
    x_train (list): 存放多个灰度矩阵的列表。
    y_train (list): 存放对应于x_train中每个矩阵的标签的列表。
    epochs (int): 训练的轮数。
    json_config (dict): 包含网络层配置的字典。

    返回:
    model (CustomMLP): 经过训练的MLP模型。
    """

    # 使用示例
    # 假设x_train和y_train已经被加载和预处理
    # epochs = 5
    # user_config = {'linear1': 128, 'sigmoid1': '', 'linear2': 64, 'ReLU1': '', 'linear3': 10}
    # trained_model = train_mlp_model(x_train, y_train, epochs, user_config)

    # 将灰度矩阵列表转换为向量
    x_train_tensors = [torch.tensor(matrix, dtype=torch.float32).view(-1) for matrix in x_train]
    # 将向量堆叠成一个张量
    x_train_tensor = torch.stack(x_train_tensors)
    # 获取输入特征的维度
    input_features = x_train_tensor.size(1)

    # 将标签列表转换为张量
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    # 创建自定义MLP模型
    class CustomMLP(nn.Module):
        def __init__(self, config, input_features):
            super(CustomMLP, self).__init__()
            layers = []
            input_dim = input_features
            for layer_name, layer_info in config.items():
                if 'linear' in layer_name:
                    output_dim = layer_info
                    layers.append(nn.Linear(input_dim, output_dim))
                    input_dim = output_dim  # 更新input_dim为下一层的输入维度
                elif 'ReLU' in layer_name:
                    layers.append(nn.ReLU())
                elif 'sigmoid' in layer_name:
                    layers.append(nn.Sigmoid())
            self.layers = nn.Sequential(*layers)

        def forward(self, x):
            return self.layers(x)

    # 实例化模型
    model = CustomMLP(json_config, input_features)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 创建数据加载器
    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 训练模型
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()  # 清除梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

        # # 打印训练进度
        # print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    # 返回训练好的模型
    return model

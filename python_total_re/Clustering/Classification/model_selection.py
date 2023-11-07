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

def classify_with_lightgbm(data, num_class, epoch=1, params=None):
    """
    使用 LightGBM 进行多分类模型训练

    参数：
    data (DataFrame): 包含特征和目标列'y'的数据集
    num_class (int): 目标类别的数量
    epoch (int): 模型训练的轮数，默认为1
    params (dict): LightGBM模型参数，默认为None

    返回：
    model (lightgbm.Booster): 训练好的 LightGBM 模型
    """
    # 数据集划分
    y = data['y']
    x = data.drop(columns=['y'])

    # 分割数据集，80%用于训练，20%用于验证
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # 如果没有提供参数，则定义默认参数
    if params is None:
        params = {
            'objective': 'multiclass',  # 多分类问题
            'num_class': num_class,     # 类别数目
            'metric': 'multi_logloss',  # 多分类的对数损失
            'verbose': -1              # 控制打印输出
        }

    # 创建 LightGBM 数据集
    dtrain = lgb.Dataset(x_train, label=y_train)
    dval = lgb.Dataset(x_val, label=y_val)

    # 模型训练
    model = lgb.train(params, dtrain, num_boost_round=epoch, valid_sets=[dtrain, dval], early_stopping_rounds=10)

    return model


def classify_with_catboost(data, num_class, epoch=1, params=None):
    """
    使用 CatBoost 进行多分类模型训练

    参数：
    data (DataFrame): 包含特征和目标列'y'的数据集
    num_class (int): 目标类别的数量
    epoch (int): 模型训练的轮数，默认为1
    params (dict): CatBoost 模型参数，默认为None

    返回：
    model (CatBoostClassifier): 训练好的 CatBoost 分类器模型
    """
    # 数据集划分
    y = data['y']
    x = data.drop(columns=['y'])

    # 分割数据集，80%用于训练，20%用于验证
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # 如果没有提供参数，则定义默认参数
    if params is None:
        params = {
            'loss_function': 'MultiClass',  # 多分类问题
            'iterations': epoch,           # 训练轮数
            'random_seed': 42,
            'verbose': False
        }
    else:
        params['iterations'] = epoch  # 确保传入的参数中有epoch

    # 初始化模型
    model = CatBoostClassifier(**params)

    # 模型训练
    model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=10)

    return model

def classify_with_xgboost(data, num_class, epoch=1, params=None):
    """
    使用 XGBoost 进行多分类模型训练

    参数：
    data (DataFrame): 包含特征和目标列'y'的数据集
    num_class (int): 目标类别的数量
    epoch (int): 模型训练的轮数，默认为1
    params (dict): XGBoost 模型参数，默认为None

    返回：
    model (xgboost.Booster): 训练好的 XGBoost 模型
    """
    # 数据集划分
    y = data['y']
    x = data.drop(columns=['y'])

    # 分割数据集，80%用于训练，20%用于测试
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # 如果没有提供参数，则定义默认参数
    if params is None:
        params = {
            'objective': 'multi:softmax',  # 多分类问题
            'num_class': num_class,        # 类别数目
            'eval_metric': 'mlogloss',     # 多分类的对数损失
            'seed': 42
        }

    # 创建 DMatrix
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)

    # 初始化模型
    model = None

    # 模型训练
    for i in range(epoch):
        model = xgb.train(params, dtrain, num_boost_round=10, evals=[(dtrain, 'train'), (dval, 'val')],
                          early_stopping_rounds=10, xgb_model=model)

    return model

def CNN_model(data, num_class, epoch=1, lr=0.001, batch_size=64):
    """
    搭建CNN(卷积神经网络)模型
    ... ...
    参数：
    x_train: 传入的传入


    返回：

    """

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



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

def Lightgbm(data, num_class, epoch=1, params=None):
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


def Catboost(data, num_class, epoch=1, params=None):
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

def Xgboost(data, num_class, epoch=1, params=None):
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


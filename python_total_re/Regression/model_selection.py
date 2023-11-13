from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split


def Random_forest(data, epoch=10, n_estimators=100, max_depth=None, random_state=None):
    """
    使用随机森林进行回归模型训练，并使用检验集进行模型评估

    参数：
    data(DataFrame)
    n_estimators (int): 构建的决策树数量，默认为100
    max_depth (int): 决策树的最大深度，如果为None，则树将在所有叶子都纯净或者达到最小样本数时停止生长
    random_state (int): 控制随机数生成器的种子，默认为None

    返回：
    model (RandomForestRegressor): 训练完成的回归模型
    score (float): 检验集上的R^2评分
    """
    # 数据集划分
    y_train = data['y']
    x_train = data.drop(columns=['y'])

    # 导入随机森林回归器
    # 创建随机森林回归器实例
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # 对模型进行训练
    for i in range(epoch):
        model.fit(x_train, y_train)

    return model


def Lightgbm(data, epoch=1, params=None):
    """
    使用LightGBM进行模型选择

    参数：
    data(DataFrame)
    epoch (int): 模型训练的轮数，默认为1
    params (dict): LightGBM模型参数，默认为None

    返回：
    model (lightgbm.Booster): 训练好的LightGBM模型
    """
    # 数据集划分
    y = data['y']
    x = data.drop(columns=['y'])

    # 分割数据集，80%用于训练，20%用于测试
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    if params is None:
        params = {}

    dtrain = lgb.Dataset(x_train, label=y_train)
    dval = lgb.Dataset(x_val, label=y_val)

    # 初始无模型，防止下面init_model报错
    model = None

    for i in range(epoch):
        model = lgb.train(params, dtrain, valid_sets=[dtrain, dval], early_stopping_rounds=10, init_model=model)

    return model

def Xgboost(data, epoch=1, params=None):
    """
    使用XGBoost进行模型选择

    参数：
    data
    epoch (int): 模型训练的轮数，默认为1
    params (dict): XGBoost模型参数，默认为None

    返回：
    model (xgboost.Booster): 训练好的XGBoost模型
    """
    # 数据集划分
    y = data['y']
    x = data.drop(columns=['y'])

    # 分割数据集，80%用于训练，20%用于测试
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    if params is None:
        params = {}

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)

    model = None

    for i in range(epoch):
        model = xgb.train(params, dtrain, evals=[(dtrain, 'train'), (dval, 'val')], early_stopping_rounds=10, xgb_model=model)

    return model


def Catboost(data, epoch=1, params=None):
    """
    使用CatBoost进行模型选择
    ... ...
    参数：
    data(DataFrame)
    params (dict): CatBoost模型参数，默认为None

    返回：
    model (catboost.CatBoostClassifier or catboost.CatBoostRegressor): 训练好的CatBoost模型
    """
    # 数据集划分
    y = data['y']
    x = data.drop(columns=['y'])

    # 分割数据集，80%用于训练，20%用于测试
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    if params is None:
        params = {}

    # 创建分类模型
    model = cb.CatBoostRegressor(**params)

    for i in range(epoch):
        model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=10)

    return model

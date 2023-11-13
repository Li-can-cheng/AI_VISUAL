import cv2
import pandas as pd
import sys
import os
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 定义模型的层和参数

def import_csv_data(file_path):
    """
    从本地CSV文件导入数据并返回DataFrame对象
    由于只有x_train,没有y_train,用于无监督学习

    参数：
    file_path (str): CSV文件的路径

    返回：
    x_train, y_train (DataFrame, Series): 读取的Excel文件数据的DataFrame对象
    x_train (DataFrame): 表头为'y'的一列数据，如果不存在则为None
    """
    data = pd.read_csv(file_path)
    if 'y' not in data.columns:
        raise ValueError("列'y'在文件中不存在，请检查所给表格文件哦")
        sys.exit(1)

    return data

def import_excel_data(file_path):
    """
    从本地Excel文件导入数据并返回DataFrame对象
    若任务为聚类等无监督学习,那么读取的表格仅为DataFrame,没有y_train
    若任务为有监督学习，那么在读取的表格的'y'这一列为label,读取后一并返回(元组形式)

    参数：
    file_path (str): Excel文件的路径

    返回：
    data(DataFrame): 读取的Excel文件数据的DataFrame对象
    """
    data = pd.read_excel(file_path)
    if 'y' not in data.columns:
        raise ValueError("列'y'在文件中不存在，请检查所给表格文件哦")
        sys.exit(1)

    return data


def import_pic_data(folder_path):
    # 创建一个空列表用于存放灰度矩阵
    gray_images = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 读取图像并将其转换为灰度图像
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # 将灰度图像添加到列表中
        gray_images.append(image)

    # 返回灰度图像列表
    return gray_images


# 假设data_dict是一个包含数据的字典，其中包含了"username"键和对应的值
data_dict = {
    "username": "cloud"
}

# 提取用户名
username = data_dict["username"]

# 搜索含有对应用户名字符串的文件
folder_path = "model_save"
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if username in file:
            file_path = os.path.join(root, file)

# 判断关键字并选择加载方式
if "Xgboost" in file_path:
    # 使用xgboost加载模型
    model = xgb.Booster(model_file=file_path)

elif "Lightgbm" in file_path:
    # 使用lightgbm加载模型
    model = lgb.Booster(model_file=file_path)

elif "Catboost" in file_path:
    # 使用catboost加载模型
    model = cb.CatBoost()
    model.load_model(file_path)

elif "CNN" in file_path or "MLP" in file_path:
    # 使用torch加载模型

    model = MyModel()
    state_dict = torch.load(file_path)
    model.load_state_dict(state_dict)

# 开始预测数据
prediction = model(data)

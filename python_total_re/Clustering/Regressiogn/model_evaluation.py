from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def calculate_mse(y_true, y_pred):
    """
    ... 计算模型的均方误差（MSE）

    参数：
    y_true (array-like): 真实的目标变量值
    y_pred (array-like): 模型预测的目标变量值

    返回：
    mse (float): 计算得到的均方误差（MSE）
    """
    mse = mean_squared_error(y_true, y_pred)
    return mse

def calculate_mape(y_true, y_pred):
    """
    计算模型的平均绝对百分比误差（MAPE）

    参数：
    y_true (array-like): 真实的目标变量值
    y_pred (array-like): 模型预测的目标变量值

    返回：
    mape (float): 计算得到的平均绝对百分比误差（MAPE）
    """
    errors = abs((y_true - y_pred) / y_true)
    mape = np.mean(errors) * 100
    return mape

def calculate_rmse(y_true, y_pred):
    """
    计算模型的均方根误差（RMSE）

    参数：
    y_true (array-like): 真实的目标变量值
    y_pred (array-like): 模型预测的目标变量值

    返回：
    rmse (float): 计算得到的均方根误差（RMSE）
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    return rmse

def calculate_r2(y_true, y_pred):
    """
    计算模型的R平方值（R^2）

    参数：
    y_true (array-like): 真实的目标变量值
    y_pred (array-like): 模型预测的目标变量值

    返回：
    r2 (float): 计算得到的R平方值
    """
    r2 = r2_score(y_true, y_pred)
    return r2

def calculate_max_error(y_true, y_pred):
    """
    计算模型的最大残差

    参数：
    y_true (array-like): 真实的目标变量值
    y_pred (array-like): 模型预测的目标变量值

    返回：
    max_error (float): 计算得到的最大残差
    """
    max_error = max(abs(y_true - y_pred))
    return max_error

from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import numpy as np

def calculate_precision(y_true, y_pred):
    """
    计算模型的精确率

    参数：
    y_true (array-like): 真实的目标变量值
    y_pred (array-like): 模型预测的目标变量值

    返回：
    precision (float): 计算得到的精确率
    """
    precision = precision_score(y_true, y_pred)
    return precision

def calculate_f1_score(y_true, y_pred):
    """
    计算模型的F1值

    参数：
    y_true (array-like): 真实的目标变量值
    y_pred (array-like): 模型预测的目标变量值

    返回：
    f1_score (float): 计算得到的F1值
    """
    f1 = f1_score(y_true, y_pred)
    return f1

def calculate_auc_score(y_true, y_pred):
    """
    计算模型的AUC值

    参数：
    y_true (array-like): 真实的目标变量值
    y_pred (array-like): 模型预测的目标变量值

    返回：
    auc_score (float): 计算得到的AUC值
    """
    auc_score = roc_auc_score(y_true, y_pred)
    return auc_score

def calculate_roc(y_true, y_pred):
    """
    计算模型的ROC曲线和AUC值

    参数：
    y_true (array-like): 真实的目标变量值
    y_pred (array-like): 模型预测的目标变量值

    返回：
    fpr (array): ROC曲线的假正例率
    tpr (array): ROC曲线的真正例率
    roc_auc (float): 计算得到的AUC值
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

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


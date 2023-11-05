import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import jieba
import cv2
import sys

def handle_missing_values(data, method='mean'):
    """
    根据指定的方法来填充或删除数据中的缺失值

    参数：
    data (pandas.DataFrame): 需要处理缺失值的数据
    method (str): 处理缺失值的方法，可选'mean'、'median'、'interpolate'或'knn'

    返回：
    cleaned_data (pandas.DataFrame): 处理后的数据
    """
    # 检查每列的缺失值数量
    missing_values = data.isnull().sum()

    # 根据指定的方法处理缺失值
    if method == 'mean':
        # 用均值填充缺失值
        cleaned_data = data.fillna(data.mean())
    elif method == 'median':
        # 用中位数填充缺失值
        cleaned_data = data.fillna(data.median())
    elif method == 'interpolate':
        # 使用线性插值填充缺失值
        cleaned_data = data.interpolate()
    elif method == 'knn':
        # 使用KNN最近邻方法填充缺失值
        imputer = KNNImputer(n_neighbors=5)
        filled_data = imputer.fit_transform(data)
        cleaned_data = pd.DataFrame(filled_data, columns=data.columns)
    else:
        raise ValueError("Method not recognized. Please input 'mean', 'median', 'interpolate', or 'knn'.")
        sys.exit(1)

    return cleaned_data

def handle_outliers(data, method='z-score', replace_method='extremes', threshold=3):
    """
    处理数据中的异常值，可以选择用极值或均值/中位数替换异常值。

    参数：
    data (pandas.DataFrame): 需要处理异常值的数据。
    method (str, 可选): 异常值检测方法，默认为'z-score'，可选值为'z-score'和'IQR'。
    replace_method (str, 可选): 异常值替换方法，'extremes'替换为极值，'mean'替换为均值，'median'替换为中位数。
    threshold (float, 可选): 异常值的阈值，默认为3。

    返回：
    cleaned_data (pandas.DataFrame): 处理后的数据，异常值根据选择的替换方法被替换。
    """
    cleaned_data = data.copy()

    # 根据选择的方法检测异常值
    if method == 'z-score':
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = z_scores > threshold
    elif method == 'IQR':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (data < lower_bound) | (data > upper_bound)

    # 根据选择的替换方法替换异常值
    if replace_method == 'extremes':
        if method == 'z-score':
            upper_extreme = data[z_scores <= threshold].max()
            lower_extreme = data[z_scores <= threshold].min()
            cleaned_data[outliers] = np.where(cleaned_data > upper_extreme, upper_extreme, lower_extreme)
        elif method == 'IQR':
            cleaned_data[data < lower_bound] = lower_bound
            cleaned_data[data > upper_bound] = upper_bound
    elif replace_method == 'mean':
        mean_value = data.mean()
        cleaned_data[outliers] = mean_value
    elif replace_method == 'median':
        median_value = data.median()
        cleaned_data[outliers] = median_value
    else:
        raise ValueError("Replace method not recognized. Use 'extremes', 'mean', or 'median'.")
        sys.exit(1)

    return cleaned_data

def filter_data(data, condition):
    """
    根据条件筛选数据

    参数：
    data (DataFrame): 需要筛选的数据
    condition (str): 筛选条件，使用类似于SQL的语法

    返回：
    filtered_data (pandas.DataFrame): 筛选后的数据
    """
    filtered_data = data.query(condition)
    return filtered_data

def modify_data(data, operation, specifier=None):
    """
    增加或删除数据

    参数：
    data (pandas.DataFrame): 需要修改的数据
    operation (str): 操作类型，可选值为'add'和'delete'
    specifier (str or None): 新数据的文件路径或删除数据的行数指定，取决于操作类型

    返回：
    modified_data (pandas.DataFrame): 修改后的数据
    """
    modified_data = data.copy()

    if operation == 'add':
        if specifier is None:
            raise ValueError("添加操作需要提供新数据的文件路径。")
            sys.exit(1)

        # 读取新数据
        new_data = pd.read_csv(specifier)

        # 检查列数是否一致
        if new_data.shape[1] != data.shape[1]:
            raise ValueError("新数据的列数必须与原数据一致。")
            exit(1)

        # 拼接数据
        modified_data = pd.concat([modified_data, new_data], ignore_index=True)

    elif operation == 'delete':
        if specifier is None:
            raise ValueError("删除操作需要提供行数指定。")
            sys.exit(1)

        # 解析行数指定
        if '~' in specifier:
            start, end = map(int, specifier.split('~'))
            rows_to_delete = range(start, end + 1)
        else:
            rows_to_delete = [int(specifier)]

        # 删除指定行
        modified_data = modified_data.drop(rows_to_delete, axis=0).reset_index(drop=True)

    else:
        raise ValueError("操作类型必须是'add'或'delete'。")
        sys.exit(1)

    return modified_data

def merge_data(data1, data2, on=None):
    """
    合并数据

    参数：
    data1 (pandas.DataFrame): 第一个数据集
    data2 (pandas.DataFrame): 第二个数据集
    on (str, 可选): 合并的列名，如果不提供，则默认使用两个数据集中的所有列

    返回：
    merged_data (pandas.DataFrame): 合并后的数据
    """
    merged_data = pd.merge(data1, data2, on=on)
    return merged_data

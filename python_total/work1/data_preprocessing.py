import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import jieba
import cv2


def multiply(data, multiply_factor: int):
    try:
        data = data.astype(int)  # 尝试将所有列转换为 int 类型
    except ValueError as e:
        return str(e)  # 或者你可以选择抛出一个异常

    processed_data = data * int(multiply_factor)  # 防止爆炸
    return processed_data


def handle_missing_values1(data):
    """
    填充或删除数据中的缺失值

    参数：
    data (pandas.DataFrame): 需要处理缺失值的数据

    返回：
    cleaned_data (pandas.DataFrame): 处理后的数据，缺失值被填充或删除
    """
    # 检查每列的缺失值数量
    missing_values = data.isnull().sum()

    # 填充缺失值或删除缺失值所在的行
    cleaned_data = data.fillna(method='ffill').dropna()

    return cleaned_data


def handle_missing_values2(data):
    """
    使用KNN最近邻方法填充缺失值或删除缺失值所在的行
    Copy

    参数：
    data (pandas.DataFrame): 需要处理缺失值的数据

    返回：
    cleaned_data (pandas.DataFrame): 处理后的数据，缺失值使用KNN进行填补
    """
    # 检查每列的缺失值数量
    missing_values = data.isnull().sum()

    # 使用KNN最近邻方法填充缺失值
    imputer = KNNImputer(n_neighbors=5)
    filled_data = imputer.fit_transform(data)

    # 将填充后的数据转换为DataFrame
    cleaned_data = pd.DataFrame(filled_data, columns=data.columns)

    return cleaned_data


def handle_outliers1(data, method='z-score', threshold=3):
    """
    处理数据中的异常值

    参数：
    data (pandas.DataFrame): 需要处理异常值的数据
    method (str, 可选): 异常值处理方法，默认为'z-score'，可选值为'z-score'和'IQR'
    threshold (float, 可选): 异常值的阈值，默认为3

    返回：
    cleaned_data (pandas.DataFrame): 处理后的数据，异常值被替换为NaN
    """
    cleaned_data = data.copy()

    if method == 'z-score':
        z_scores = np.abs((data - data.mean()) / data.std())
        cleaned_data[z_scores > threshold] = np.nan
    elif method == 'IQR':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        cleaned_data[(data < Q1 - threshold * IQR) | (data > Q3 + threshold * IQR)] = np.nan

    return cleaned_data


def handle_outliers2(data, method='mean', threshold=3):
    """
    处理数据中的异常值
    Copy

    参数：
    data (pandas.DataFrame): 需要处理异常值的数据
    method (str, 可选): 异常值处理方法，默认为'mean'，可选值为'mean'和'median'
    threshold (float, 可选): 异常值的阈值，默认为3

    返回：
    cleaned_data (pandas.DataFrame): 处理后的数据，异常值被替换为均值或中位数
    """
    cleaned_data = data.copy()

    if method == 'mean':
        mean_value = data.mean()
        cleaned_data[(data - mean_value).abs() > threshold * data.std()] = mean_value
    elif method == 'median':
        median_value = data.median()
        cleaned_data[(data - median_value).abs() > threshold * data.std()] = median_value

    return cleaned_data


def filter_data(data, condition):
    """
    根据条件筛选数据

    参数：
    data (pandas.DataFrame): 需要筛选的数据
    condition (str): 筛选条件，使用类似于SQL的语法

    返回：
    filtered_data (pandas.DataFrame): 筛选后的数据
    """
    filtered_data = data.query(condition)
    return filtered_data


def modify_data(data, operation, new_data=None):
    """
    增加或删除数据

    参数：
    data (pandas.DataFrame): 需要修改的数据
    operation (str): 操作类型，可选值为'add'和'delete'
    new_data (pandas.DataFrame, 可选): 新数据，仅在操作类型为'add'时需要提供

    返回：
    modified_data (pandas.DataFrame): 修改后的数据
    """
    modified_data = data.copy()

    if operation == 'add':
        modified_data = pd.concat([modified_data, new_data], ignore_index=True)
    elif operation == 'delete':
        modified_data = modified_data.dropna()

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


def split_dataset(data, test_size=0.2, random_state=None):
    """
    划分数据集为训练集和测试集

    参数：
    data (pandas.DataFrame): 需要划分的数据集
    test_size (float, 可选): 测试集的比例，默认为0.2
    random_state (int, 可选): 随机种子，用于重现随机划分的结果

    返回：
    train_data (pandas.DataFrame): 训练集数据
    test_data (pandas.DataFrame): 测试集数据
    """
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data


def tokenize_text(text):
    """
    对文本进行分词

    参数：
    text (str): 需要进行分词的文本

    返回：
    tokens (list): 分词后的结果，以列表形式返回
    """
    tokens = jieba.lcut(text)
    return tokens


def segment_image(image_path):
    """
    对图像进行分割

    参数：
    image_path (str): 图像文件的路径

    返回：
    segmented_image (numpy.ndarray): 分割后的图像，以NumPy数组形式返回
    """
    image = cv2.imread(image_path)
    # 进行图像分割的处理步骤...
    segmented_image = ...  # 分割后的图像

    return segmented_image


def convert_to_gray(image_path):
    """
    将图像转换为灰度图像

    参数：
    image_path (str): 图像文件的路径

    返回：
    gray_image (numpy.ndarray): 转换后的灰度图像，以NumPy数组形式返回
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray_image

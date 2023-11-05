import sys
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler, Normalizer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler

def standardize_data(data, method='z_score'):
    """
    根据指定的方法对数据进行标准化处理

    参数：
    data (pandas.DataFrame): 需要进行标准化的数据
    method (str): 标准化的方法，可选'z_score'、'mean_normalization'或'scale_to_unit_length'

    返回：
    standardized_data (pandas.DataFrame): 标准化后的数据
    """
    if method == 'z_score':
        # 使用Z-Score标准化
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(data)
        standardized_data = pd.DataFrame(standardized_data, columns=data.columns)
    elif method == 'mean_normalization':
        # 使用Mean Normalization
        standardized_data = (data - data.mean()) / (data.max() - data.min())
    elif method == 'scale_to_unit_length':
        # 使用Scale to Unit Length
        scaler = Normalizer()
        standardized_data = scaler.fit_transform(data)
        standardized_data = pd.DataFrame(standardized_data, columns=data.columns)
    else:
        raise ValueError("Method not recognized. Please input 'z_score', 'mean_normalization', or 'scale_to_unit_length'.")
        sys.exit(1)

    return standardized_data

def normalize_data(data, method='min_max'):
    """
    根据指定的方法对数据进行归一化处理

    参数：
    data (pandas.DataFrame): 需要进行归一化的数据
    method (str): 归一化的方法，可选'min_max'、'max_abs'或'robust'

    返回：
    normalized_data (pandas.DataFrame): 归一化后的数据
    """
    # 根据指定的方法处理归一化
    if method == 'min_max':
        # 使用Min-Max归一化
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data)
        normalized_data = pd.DataFrame(normalized_data, columns=data.columns)
    elif method == 'max_abs':
        # 使用绝对最大值归一化
        scaler = MaxAbsScaler()
        normalized_data = scaler.fit_transform(data)
        normalized_data = pd.DataFrame(normalized_data, columns=data.columns)
    elif method == 'robust':
        # 使用Robust Scaler归一化
        scaler = RobustScaler()
        normalized_data = scaler.fit_transform(data)
        normalized_data = pd.DataFrame(normalized_data, columns=data.columns)
    else:
        raise ValueError("Method not recognized. Please input 'min_max', 'max_abs', or 'robust'.")
        sys.exit(1)

    return normalized_data

def discretize_data(data, n_bins):
    """
    对数据进行离散化处理

    参数：
    data (pandas.DataFrame): 需要进行离散化的数据
    n_bins (int): 离散化的分箱数目

    返回：
    discretized_data (pandas.DataFrame): 离散化后的数据
    """
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    discretized_data = discretizer.fit_transform(data)

    return discretized_data

def onehot_encode(data):
    """
    对数据进行OneHot编码

    参数：
    data (pandas.DataFrame): 需要进行OneHot编码的数据

    返回：
    encoded_data (pandas.DataFrame): OneHot编码后的数据
    """
    encoder = OneHotEncoder(sparse=False)
    encoded_data = encoder.fit_transform(data)

    return encoded_data

def label_encode(data):
    """
    对数据进行Label编码

    参数：
    data (pandas.DataFrame): 需要进行Label编码的数据

    返回：
    encoded_data (pandas.DataFrame): Label编码后的数据
    """
    encoder = LabelEncoder()
    encoded_datas = encoder.fit_transform(data)

    return encoded_datas

def descriptive_statistics(data):
    """
    对x_train数据进行描述性统计

    参数：
    data (pandas.DataFrame): 需要进行描述性统计的数据

    返回：
    statistics (pandas.DataFrame): 描述性统计结果
    """
    # 数据拆分
    y_train = data['y']
    x_train = data.drop(columns=['y'])
    # 描述性统计
    statistics = x_train.describe()
    # 数据还原
    x_train['y'] = y_train
    data = x_train

    print(statistics)

    return data

def calculate_similarity(data):
    """
    计算x_train之间的相似度

    参数：
    data (numpy.ndarray): 需要计算相似度的数据，以NumPy数组形式表示

    返回：
    similarity_matrix (numpy.ndarray): 相似度矩阵，以NumPy数组形式返回
    """
    # 数据拆分
    y_train = data['y']
    x_train = data.drop(columns=['y'])
    # 描述性统计
    similarity_matrix = cosine_similarity(x_train)
    # 数据还原
    x_train['y'] = y_train
    data = x_train

    print(similarity_matrix)

    return data


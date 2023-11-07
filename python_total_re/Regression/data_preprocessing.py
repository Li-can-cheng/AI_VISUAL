import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import jieba
import cv2

def handle_missing_values(data, method='mean'):
    """
    根据指定的方法来填充或删除数据中的缺失值

    参数：
    data (pandas.DataFrame): 需要处理缺失值的数据
    method (str): 处理缺失值的方法，可选'mean'、'median'、'interpolate'或'knn'

    返回：
    cleaned_data (pandas.DataFrame): 处理后的data
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
    cleaned_data, y_train (pandas.DataFrame): 处理后的数据，异常值根据选择的替换方法被替换。
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
    data (pandas.DataFrame): 需要筛选的数据
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


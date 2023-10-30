from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2

def standardize_data(data):
    """
    对数据进行标准化处理

    参数：
    data (pandas.DataFrame): 需要进行标准化的数据

    返回：
    standardized_data (pandas.DataFrame): 标准化后的数据
    """
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)

    return standardized_data

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
    对数据进行描述性统计

    参数：
    data (pandas.DataFrame): 需要进行描述性统计的数据

    返回：
    statistics (pandas.DataFrame): 描述性统计结果
    """
    statistics = data.describe()

    return statistics

def convert_to_word_vectors(data):
    """
    将文本数据转换为词向量

    参数：
    data (list): 需要转换为词向量的文本数据，以列表形式表示

    返回：
    word_vectors (numpy.ndarray): 转换后的词向量，以NumPy数组形式返回
    """
    vectorizer = TfidfVectorizer()
    word_vectors = vectorizer.fit_transform(data)

    return word_vectors

def calculate_similarity(data):
    """
    计算数据之间的相似度

    参数：
    data (numpy.ndarray): 需要计算相似度的数据，以NumPy数组形式表示

    返回：
    similarity_matrix (numpy.ndarray): 相似度矩阵，以NumPy数组形式返回
    """
    similarity_matrix = cosine_similarity(data)

    return similarity_matrix

def enhance_image(image):
    """
    对图像进行增强处理

    参数：
    image (numpy.ndarray): 需要进行增强处理的图像，以NumPy数组形式表示

    返回：
    enhanced_image (numpy.ndarray): 增强后的图像，以NumPy数组形式返回
    """
    # 转换图像为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 对灰度图像进行直方图均衡化
    equalized_image = cv2.equalizeHist(gray_image)

    # 对均衡化后的图像进行高斯模糊
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

    # 对模糊后的图像进行边缘增强
    enhanced_image = cv2.Laplacian(blurred_image, cv2.CV_64F)

    # 将图像像素值归一化到0-255范围
    enhanced_image = cv2.normalize(enhanced_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return enhanced_image


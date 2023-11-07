import random
import numpy as np
import sys

def shuffle_images(x_train, y_train):
    """
    将图像列表和标签列表进行相同顺序的随机打乱

    参数：
    x_train (list): 包含图像数据的列表
    y_train (list): 对应图像的标签列表

    返回：
    shuffled_x_train (list): 打乱顺序后的图像数据列表
    shuffled_y_train (list): 打乱顺序后的标签列表
    """
    combined = list(zip(x_train, y_train))

    # Randomly shuffle the combined list in place
    random.shuffle(combined)

    shuffled_x_train, shuffled_y_train = zip(*combined)

    return list(shuffled_x_train), list(shuffled_y_train)

def standardize_images(images_list, labels_list, method='z_score'):
    """
    根据指定的方法对图像的灰度矩阵进行标准化处理

    参数：
    images_list (list): 包含灰度矩阵的列表
    labels_list (list): 对应图像的标签列表
    method (str): 标准化的方法，可选'z_score'、'mean_normalization'或'scale_to_unit_length'

    返回：
    standardized_images (list): 标准化后的灰度矩阵列表
    labels_list (list): 未改变的标签列表
    """

    if method not in ['z_score', 'mean_normalization', 'scale_to_unit_length']:
        raise ValueError(
            "Method not recognized. Please input 'z_score', 'mean_normalization', or 'scale_to_unit_length'.")
        sys.exit(1)

    def z_score_standardization(image):
        return (image - np.mean(image)) / np.std(image)

    def mean_normalization(image):
        return (image - np.mean(image)) / (np.max(image) - np.min(image))

    def scale_to_unit_length(image):
        norm = np.linalg.norm(image)
        return image if norm == 0 else image / norm

    # Choose the standardization function based on the method
    if method == 'z_score':
        standardization_function = z_score_standardization
    elif method == 'mean_normalization':
        standardization_function = mean_normalization
    elif method == 'scale_to_unit_length':
        standardization_function = scale_to_unit_length

    # Apply the chosen standardization method to each image
    standardized_images = [standardization_function(img) for img in images_list]

    return standardized_images, labels_list

def normalize_images(images_list, labels_list, method='min_max'):
    """
    根据指定的方法对图像的灰度矩阵进行归一化处理

    参数：
    images_list (list): 包含灰度矩阵的列表
    labels_list (list): 对应图像的标签列表
    method (str): 归一化的方法，可选'min_max'、'max_abs'或'robust'

    返回：
    normalized_images (list): 归一化后的灰度矩阵列表
    labels_list (list): 未改变的标签列表
    """

    def min_max_normalization(image):
        return (image - np.min(image)) / (np.max(image) - np.min(image)) if np.max(image) != np.min(image) else image

    def max_abs_normalization(image):
        return image / np.max(np.abs(image)) if np.max(np.abs(image)) != 0 else image

    def robust_normalization(image):
        median = np.median(image)
        quantile_range = np.quantile(image, 0.75) - np.quantile(image, 0.25)
        return (image - median) / quantile_range if quantile_range != 0 else image

    # Choose the normalization function based on the method
    if method == 'min_max':
        normalization_function = min_max_normalization
    elif method == 'max_abs':
        normalization_function = max_abs_normalization
    elif method == 'robust':
        normalization_function = robust_normalization
    else:
        raise ValueError("Method not recognized. Please input 'min_max', 'max_abs', or 'robust'.")
        sys.exit(1)
    # Apply the chosen normalization method to each image
    normalized_images = [normalization_function(img) for img in images_list]

    return normalized_images, labels_list


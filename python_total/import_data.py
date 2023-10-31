import cv2
import pandas as pd

def import_csv_data(file_path):
    """
    从本地CSV文件导入数据并返回DataFrame对象

    参数：
    file_path (str): CSV文件的路径

    返回：
    data (pandas.DataFrame): 读取的CSV文件数据的DataFrame对象
    """
    data = pd.read_csv(file_path)
    return data

def import_excel_data(file_path, sheet_name):
    """
    从本地Excel文件导入数据并返回DataFrame对象

    参数：
    file_path (str): Excel文件的路径
    sheet_name (str)(可选参数): Excel文件中要读取的工作表名称

    返回：
    data (pandas.DataFrame): 读取的Excel文件数据的DataFrame对象
    """
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

def read_image(image_path):
    """
    从给定的图片路径读取图片并转化为灰度矩阵

    参数：
    image_path (str): 图片的绝对路径

    返回：
    image (numpy.ndarray): 图片的灰度矩阵
    """

    image = pd.DataFrame(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
    return image

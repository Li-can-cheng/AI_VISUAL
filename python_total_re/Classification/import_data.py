import cv2
import pandas as pd
import sys

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

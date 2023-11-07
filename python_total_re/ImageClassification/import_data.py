import zipfile
import cv2
import numpy as np

def extract_images_from_zip(zip_path):
    """
    从ZIP文件中提取图像并将它们转换为灰度矩阵，同时记录每个图像对应的标签。

    参数：
    zip_path (str): ZIP文件的路径

    返回：
    images_list, labels_list (list, list): 图像的灰度矩阵列表和相应的标签列表
    """
    images_list = []
    labels_list = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for folder in z.namelist():
            if folder.endswith('/') and folder[:-1].isdigit():
                label = int(folder[:-1])
                for filename in z.namelist():
                    if filename.startswith(folder) and not filename.endswith('/'):
                        with z.open(filename) as file:
                            content = file.read()
                            img_array = np.frombuffer(content, np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                            images_list.append(img)
                            labels_list.append(label)

    return images_list, labels_list

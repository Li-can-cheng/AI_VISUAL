from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN


def kmeans_model(X, n_clusters=3, random_state=None):
    """
    使用K-means进行模型选择，并绘制聚类结果图。
    注意：这个函数现在返回数据点的聚类标签，而不是模型本身。如果X是二维或三维的，它还会绘制出相应的聚类结果图。
    如果X有更多维度，代码会引发一个错误，因为无法绘制超过三维的散点图。

    参数：
    X (array-like): 特征数据，可以是二维或三维。
    n_clusters (int): 聚类的簇数， 默认为3。
    random_state (int): 随机种子， 默认为None。

    返回：
    labels (array): 数据点的聚类标签。
    """
    # 创建并拟合模型
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(X)

    # 获取聚类标签
    labels = model.labels_

    # 检查数据维度
    if X.shape[1] == 3:
        # 绘制三维散点图
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', marker='o')
        ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], model.cluster_centers_[:, 2], c='red',
                   marker='x', s=100)
        ax.set_title('K-means Clustering in 3D')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
    elif X.shape[1] == 2:
        # 绘制二维散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', marker='x', s=100)
        plt.title('K-means Clustering in 2D')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    else:
        raise ValueError("X must be 2D or 3D. Other dimensions are not supported for plotting.")

    plt.show()

    return labels

def select_hierarchical_clustering(data, n_clusters=None, linkage_method='ward'):
    """
    对提供的数据执行层次聚类并返回拟合模型。

    :param data: DataFrame
        待聚类的数据。它应该是一个二维数组或一个DataFrame。

    :param n_clusters: int, 可选
        要找到的聚类数量。如果未提供，模型将不会对树状图进行剪辑
        并且不会分配聚类标签。

    :param linkage_method: str, 可选
        使用的链接算法。选项包括 'ward', 'complete', 'average', 和 'single'。

    :return: dict
        一个字典，包含 'model'，即拟合的AgglomerativeClustering模型，
        如果n_clusters为None，还包含 'dendrogram'，即用于绘制树状图的链接矩阵。
    """

    if n_clusters is not None:
        # If n_clusters is provided, fit the model with the number of clusters
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        model.fit(data)
        result = {'model': model}
    else:
        # If n_clusters is not provided, perform clustering without cutting the dendrogram
        linkage_matrix = linkage(data, method=linkage_method)
        result = {'dendrogram': linkage_matrix}

        # Plot the dendrogram
        plt.figure()
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.show()

    return result

def dbscan_clustering(X, eps=0.5, min_samples=5):
    """
    使用DBSCAN进行密度聚类，并绘制聚类结果图。
    注意：这个函数现在返回数据点的聚类标签，而不是模型本身。如果X是二维或三维的，它还会绘制出相应的聚类结果图。
    如果X有更多维度，代码会引发一个错误，因为无法绘制超过三维的散点图。

    参数：
    X (array-like): 特征数据，可以是二维或三维。
    eps (float): DBSCAN算法中的邻域半径。
    min_samples (int): 形成稠密区域所需的最小样本数。

    返回：
    labels (array): 数据点的聚类标签。
    """
    # 创建并拟合DBSCAN模型
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)

    # 获取聚类标签
    labels = dbscan.labels_

    # 检查数据维度并绘图
    if X.shape[1] == 3:
        # 绘制三维散点图
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', marker='o')
        ax.set_title('DBSCAN Clustering in 3D')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
    elif X.shape[1] == 2:
        # 绘制二维散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
        plt.title('DBSCAN Clustering in 2D')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    else:
        raise ValueError("X must be 2D or 3D. Other dimensions are not supported for plotting.")

    plt.show()

    return labels



from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

def evaluate_model_silhouette(X, labels):
    # 示例用法（这需要您提供数据集 X 和对应的簇标签 labels）
    # X = ... # 数据集特征
    # labels = ... # 由聚类模型分配的簇标签
    # silhouette_avg = evaluate_model_silhouette(X, labels)
    # print(f'平均轮廓系数为: {silhouette_avg}')

    """
    计算聚类模型的轮廓系数。

    参数:
    X : 传入的DataFrame训练数据。
    labels : array-like of shape (n_samples,)
        聚类标签。

    返回:
    silhouette_avg : float
        轮廓系数的平均值。
    """
    # 计算所有样本的轮廓系数的平均值
    silhouette_avg = silhouette_score(X, labels)
    return silhouette_avg

def evaluate_davies_bouldin(X, labels):
    """
    计算并返回Davies-Bouldin指数以评估聚类模型。

    参数:
    X: array-like, shape (n_samples, n_features), 聚类数据集。
    labels: array-like, shape (n_samples,), 数据点的簇标签。

    返回:
    db_index: float, Davies-Bouldin指数。
    """
    db_index = davies_bouldin_score(X, labels)
    return db_index


def evaluate_calinski_harabasz(X, labels):
    """
    计算并返回Calinski-Harabasz指数以评估聚类模型。

    参数:
    X: array-like, shape (n_samples, n_features), 聚类数据集。
    labels: array-like, shape (n_samples,), 数据点的簇标签。

    返回:
    ch_index: float, Calinski-Harabasz指数。
    """
    ch_index = calinski_harabasz_score(X, labels)
    return ch_index
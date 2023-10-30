from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.cluster import KMeans


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=None):
    """
    使用随机森林进行分类模型训练

    参数：
    X_train (array-like): 训练集特征数据
    y_train (array-like): 训练集目标变量
    n_estimators (int): 决策树的数量，默认为100
    max_depth (int): 决策树的最大深度，默认为None
    random_state (int): 随机种子，默认为None

    返回：
    model (RandomForestClassifier): 训练好的随机森林分类器模型
    """
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def select_lightgbm_model(X_train, y_train, X_val, y_val, params=None):
    """
    使用LightGBM进行模型选择

    参数：
    X_train (array-like): 训练集特征数据
    y_train (array-like): 训练集目标变量
    X_val (array-like): 验证集特征数据
    y_val (array-like): 验证集目标变量
    params (dict): LightGBM模型参数，默认为None

    返回：
    model (lightgbm.Booster): 训练好的LightGBM模型
    """
    if params is None:
        params = {}

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(params, dtrain, valid_sets=[dtrain, dval], early_stopping_rounds=10)

    return model

def select_xgboost_model(X_train, y_train, X_val, y_val, params=None):
    """
    使用XGBoost进行模型选择
    ...
    参数：
    X_train (array-like): 训练集特征数据
    y_train (array-like): 训练集目标变量
    X_val (array-like): 验证集特征数据
    y_val (array-like): 验证集目标变量
    params (dict): XGBoost模型参数，默认为None

    返回：
    model (xgboost.Booster): 训练好的XGBoost模型
    """
    if params is None:
        params = {}

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(params, dtrain, evals=[(dtrain, 'train'), (dval, 'val')], early_stopping_rounds=10)

    return model

def select_catboost_model(X_train, y_train, X_val, y_val, params=None):
    """
    使用CatBoost进行模型选择
    ... ...
    参数：
    X_train (array-like): 训练集特征数据
    y_train (array-like): 训练集目标变量
    X_val (array-like): 验证集特征数据
    y_val (array-like): ... 验证集目标变量
    params (dict): CatBoost模型参数，默认为None

    返回：
    model (catboost.CatBoostClassifier or catboost.CatBoostRegressor): 训练好的CatBoost模型
    """
    if params is None:
        params = {}

    if isinstance(y_train[0], str):
        model = cb.CatBoostClassifier(**params)
    else:
        model = cb.CatBoostRegressor(**params)

    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)

    return model

def select_kmeans_model(X, n_clusters=8, random_state=None):
    """
    使用K-means进行模型选择
    ... ...
    参数：
    X (array-like): 特征数据
    n_clusters (int): 聚类的簇数，默认为8
    random_state (int): 随机种子，默认为None

    返回：
    model (sklearn.cluster.KMeans): 训练好的K-means模型
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(X)

    return model


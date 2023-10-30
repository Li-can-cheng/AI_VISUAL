import pickle

def predict(model, data):
    """
    对输入的数据进行预测

    参数：
    model: 预训练的模型对象
    data: 输入的数据，可以是单个样本或多个样本

    返回：
    predictions: 预测结果
    """
    predictions = model.predict(data)
    return predictions

def save_model(model, file_path):
    """
    保存训练好的模型到指定文件路径

    参数：
    model: 训练好的模型对象
    file_path: 保存模型的文件路径

    返回：
    无
    """
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)


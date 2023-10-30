import time
import psutil


def monitor_model(model, data):
    """
    监测模型的运行结果、运行时间和运行资源使用情况，并进行相应告警

    参数：
    model: 训练好的模型对象
    data: 用于模型评估的数据

    返回：
    result: 模型的运行结果
    runtime: 模型的运行时间
    resource_usage: 模型的运行资源使用情况
    """
    start_time = time.time()
    result = model.predict(data)
    end_time = time.time()
    runtime = end_time - start_time

    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent

    # 进行告警逻辑，根据需要自行实现

    return result, runtime, (cpu_usage, memory_usage)


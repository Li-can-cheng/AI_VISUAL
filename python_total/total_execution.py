import json
import importlib

# 大概的json格式
json_str = "{ 'module' : [a, b, c], function : [a, b, c], arguments : [[a], [b], [c]]}"

# 将json格式数据转换为字典
json_dict = json.loads(json_str)

# 第一步：读取数据
data_total = json_dict['data']

# 非第一步：对数据进行循环处理
for i in range(2, 8):
    if f'module{i}' in json_dict:
        module = importlib.import_module(json_dict[f'module{i}'])  # 通过字符串获取模块
        function = getattr(module, json_dict[f'function{i}'])  # 通过字符串获取函数
        data_total = function(*json_dict[f'arguments{i}'])  # 更新迭代后的数据


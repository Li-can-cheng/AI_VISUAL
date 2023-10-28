import json
import numpy as np

# 创建一个随机28x28数组，数值范围从0到255（通常用于灰度图像）
array_28x28 = np.random.randint(0, 256, (28, 28)).tolist()

# 转换为JSON对象
data = {
    "data": array_28x28
}

# 转换为JSON字符串
json_data = json.dumps(data, indent=4)
print(json_data)


import importlib
from typing import Any
import pandas as pd
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from typing import List, Dict

# 创建一个名为router的APIRouter实例，用于定义和组织路由。
# 路由是指将HTTP请求映射到相应的处理函数或方法的过程。
router = APIRouter()

# 下面三个类的具体实现结构：
#JsonInfo
# └── Command（指令，一个指令中只有一个模块）
#   └── Module（模块，也称步骤）
#     └── Function（一个模块中调用的函数）
#         └── Parameters（函数的参数）

# Function表示传入的函数，一共包括了“函数名”和“函数参数”，注意函数的参数为dict对象
class Function(BaseModel):
    name: str
    arguments: Dict[str, Any]


# Command表示发送的指令，一共包括了“模块名”和“该模块涉及的所有函数列表”，注意一个模块中可以调用多个函数进行处理
class Command(BaseModel):
    module: str
    functions: List[Function]  # functions现在是一个Function对象的列表


# JsonInfo表示json信息集合，一共包括了“指令列表”，就是把多个步骤的指令全部汇总起来了
class JsonInfo(BaseModel):
    commands: List[Command]


# 通过路由将json发送到execute函数中
@router.post("/execute")

# 定义了一个名为execute的异步函数，用于处理POST请求
async def execute(json_info: JsonInfo):
    data_total = pd.DataFrame()  # 初始化为空的 DataFrame

    # 开始对指令进行遍历
    for command in json_info.commands:
        mod = command.module  # 读取模块名称

        # 对一个指令中模块下的函数进行遍历
        for function in command.functions:
            func_name = function.name  #取出函数名
            args = function.arguments  # args是一个字典

            # 如果 data_total 有数据，将其传递给下一个函数
            if not data_total.empty:
                args['data'] = data_total

            # 导入模块
            try:
                imported_module = importlib.import_module(mod)
            # 如果无法导入模块，将状态码设为400，并报错
            except ImportError as e:
                raise HTTPException(status_code=400, detail=f"Module {mod} not found: {str(e)}")

            # 获取函数
            try:
                function_to_call = getattr(imported_module, func_name)
            # 如果无法获取得到函数，将状态码设置为400，并报错
            except AttributeError as e:
                raise HTTPException(status_code=400, detail=f"Function {func_name} not found in module {mod}: {str(e)}")

            # 调用函数
            try:
                data_total = function_to_call(**args)
            # 如果无法调用函数，将状态码设置为400，并报错
            except Exception as e:
                raise HTTPException(status_code=400,
                                    detail=f"Error calling function {func_name} with args {args}: {str(e)}")

    return {"status": "success", "data": data_total.to_dict(orient='records')}


#
# # 大概的json格式
# json_str = "{ 'module' : [a, b, c], function : [a, b, c], arguments : [[a], [b], [c]]}"

# # 将json格式数据转换为字典
# json_dict = json.loads(json_str)
#
# # 第一步：读取数据
# data_total = json_dict['data']
#
# # 非第一步：对数据进行循环处理
# for i in range(2, 8):
#     if f'module{i}' in json_dict:
#         module = importlib.import_module(json_dict[f'module{i}'])  # 通过字符串获取模块
#         function = getattr(module, json_dict[f'function{i}'])  # 通过字符串获取函数
#         data_total = function(*json_dict[f'arguments{i}'])  # 更新迭代后的数据

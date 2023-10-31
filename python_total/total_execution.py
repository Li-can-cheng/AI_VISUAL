import importlib
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List, Union

router = APIRouter()


class Command(BaseModel):
    module: str
    function: str
    arguments: List[Union[str, int]]  # 修改以接受字符串和整数参数


class JsonInfo(BaseModel):
    commands: List[Command]


@router.post("/execute")
async def execute(json_info: JsonInfo):
    data_total = None  # 初始化数据

    for command in json_info.commands:
        mod = command.module
        func = command.function
        args = command.arguments

        # 如果 data_total 不是 None，将它添加到参数列表的开始
        if data_total is not None:
            args.insert(0, data_total)

        # 从字符串导入模块
        try:
            imported_module = importlib.import_module(mod)
        except ImportError:
            raise HTTPException(status_code=400, detail=f"Module {mod} not found")

        # 从模块获取函数
        try:
            function_to_call = getattr(imported_module, func)
        except AttributeError:
            raise HTTPException(status_code=400, detail=f"Function {func} not found in module {mod}")

        # 调用函数
        try:
            data_total = function_to_call(*args)  # 更新迭代后的数据
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return {"status": "success", "data": data_total.to_dict() if data_total is not None else None}

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

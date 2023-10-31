import importlib
import inspect  # 新导入
from typing import Any, List, Dict
import pandas as pd
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel

router = APIRouter()


class Function(BaseModel):
    name: str
    arguments: Dict[str, Any]


class Command(BaseModel):
    module: str
    functions: List[Function]


class JsonInfo(BaseModel):
    commands: List[Command]


@router.post("/execute")
async def execute(json_info: JsonInfo):
    data_total = pd.DataFrame()

    for command in json_info.commands:
        mod = command.module

        for function in command.functions:
            func_name = function.name
            args = function.arguments

            if 'data' in args and not isinstance(args['data'], pd.DataFrame):
                try:
                    args['data'] = pd.DataFrame(args['data'])
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=f"Error converting data to DataFrame: {str(e)}")

            if not data_total.empty and func_name != ('request_train' or 'predict'):
                args['data'] = data_total

            try:
                imported_module = importlib.import_module(mod)
            except ImportError as e:
                raise HTTPException(status_code=400, detail=f"Module {mod} not found: {str(e)}")

            try:
                function_to_call = getattr(imported_module, func_name)
            except AttributeError as e:
                raise HTTPException(status_code=400, detail=f"Function {func_name} not found in module {mod}: {str(e)}")

            # 获取函数的参数信息
            sig = inspect.signature(function_to_call)
            func_args = {k: args[k] for k in sig.parameters if k in args}

            try:
                data_total = function_to_call(**func_args)
            except Exception as e:
                raise HTTPException(status_code=400,
                                    detail=f"Error calling function {func_name} with args {func_args}: {str(e)}")

            if not isinstance(data_total, pd.DataFrame):
                try:
                    data_total = pd.DataFrame(data_total)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=f"Error converting data to DataFrame: {str(e)}")

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

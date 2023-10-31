import importlib
from typing import Any
import pandas as pd
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel

router = APIRouter()

from typing import List, Dict


class Function(BaseModel):
    name: str
    arguments: Dict[str, Any]


class Command(BaseModel):
    module: str
    functions: List[Function]  # functions现在是一个Function对象的列表


class JsonInfo(BaseModel):
    commands: List[Command]


@router.post("/execute")
async def execute(json_info: JsonInfo):
    data_total = pd.DataFrame()  # 初始化为空的 DataFrame

    for command in json_info.commands:
        mod = command.module

        for function in command.functions:
            func_name = function.name
            args = function.arguments  # args是一个字典

            # 如果 data 不是 DataFrame，尝试将其转换为 DataFrame
            if 'data' in args and not isinstance(args['data'], pd.DataFrame):
                try:
                    args['data'] = pd.DataFrame(args['data'])
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=f"Error converting data to DataFrame: {str(e)}")

            # 如果 data_total 有数据，将其传递给下一个函数
            if not data_total.empty:
                args['data'] = data_total

            # 导入模块
            try:
                imported_module = importlib.import_module(mod)
            except ImportError as e:
                raise HTTPException(status_code=400, detail=f"Module {mod} not found: {str(e)}")

            # 获取函数
            try:
                function_to_call = getattr(imported_module, func_name)
            except AttributeError as e:
                raise HTTPException(status_code=400, detail=f"Function {func_name} not found in module {mod}: {str(e)}")

            # 调用函数
            try:
                data_total = function_to_call(**args)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error calling function {func_name} with args {args}: {str(e)}")

            # 确保 data_total 是 DataFrame 类型
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

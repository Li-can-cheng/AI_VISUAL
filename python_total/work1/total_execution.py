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


data_total = pd.DataFrame()


@router.post("/execute")
async def execute(json_info: JsonInfo):
    global data_total  # 使用全局变量来存储累积的数据
    results = []  # 用于存储每个函数调用的结果

    for command in json_info.commands:
        mod = command.module
        try:
            imported_module = importlib.import_module(mod)  # 导入模块
        except ImportError as e:
            raise HTTPException(status_code=400, detail=f"Module {mod} not found: {str(e)}")

        for function in command.functions:
            func_name = function.name
            args = function.arguments

            # 如果参数中有'data'，并且其类型不是pd.DataFrame，则尝试将其转换
            if 'data' in args and not isinstance(args['data'], pd.DataFrame):
                try:
                    args['data'] = pd.DataFrame(args['data'])
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error converting 'data' to DataFrame: {str(e)}")

            # 如果data_total不为空，则将其作为'data'参数
            if not data_total.empty and 'data' not in args:
                args['data'] = data_total

            try:
                function_to_call = getattr(imported_module, func_name)  # 获取模块中的函数
                result = function_to_call(**args)  # 调用函数
                results.append(result)  # 存储结果
                if isinstance(result, pd.DataFrame):  # 如果结果是DataFrame，更新data_total
                    data_total = result
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error calling function {func_name}: {str(e)}")

    # 根据结果类型构建响应
    if results:
        if all(isinstance(res, pd.DataFrame) for res in results):
            # 所有结果都是DataFrames，合并它们（如果有多个）
            combined_df = pd.concat(results, ignore_index=True)
            response_data = combined_df.to_dict(orient='records')
        else:
            # 结果类型不一，返回一个列表
            response_data = [res.to_dict(orient='records') if isinstance(res, pd.DataFrame) else res for res in results]
    else:
        response_data = None

    return {"status": "success", "data": response_data}




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

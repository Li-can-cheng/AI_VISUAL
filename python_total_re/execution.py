import json
import importlib

# 模拟读取JSON字符串
json_string ='''
{
    "task":"ImageClassification",    
    "import_data":{
    "method":"import_zip_data",
    "file_path":"handwriting_digits.zip"
    },
    "data_preprocessing":[
        {
            "name":"normalize_images",
            "arguments":{
                "mean":""
            }
        },
        {
            "name":"standardize_images",
            "arguments":{
            "mean" : ""
            }    
        }
    ],
    "model_selection":{
        "name":"CNN",
        "arguments":{
            "epochs":15,
        "layer":{
        "conv2d1":[32, 2],
        "ReLU1": -1,
        "conv2d2":[16, 2] ,
        "ReLU": -1,
        "maxpool2d":2,
        "linear1":10
    }  
        }
    }, 
    "model_evaluation":["Accuracy", "F1_score"]
}'''


# ① 读取JSON，存储在data_dict中
data_dict = json.loads(json_string)

# ② 确定任务类型
task = data_dict["task"]  # 由于列表中只有一个值，直接取出来
file_path = data_dict["import_data"]["file_path"]  # 读取路径参数

# 根据任务类型选择对应的目录
module_path = task

# ③ 读取"import_data"函数名称
import_function_name = data_dict["import_data"]["method"]  # 由于列表中只有一个值，直接取出来
import_module = importlib.import_module(f"{module_path}.import_data")
import_function = getattr(import_module, import_function_name)

# 使用别名读取数据
data = import_function(file_path)

# ④ 数据预处理
for preprocessing_step in data_dict["data_preprocessing"]:
    # 获取函数名称和参数
    function_name = preprocessing_step["name"]
    function_arguments = preprocessing_step["arguments"]

    # 清洗参数字典
    cleaned_arguments = {k: v for k, v in function_arguments.items() if v != '' and v is not None}

    # 添加data到参数字典中
    cleaned_arguments['data'] = data

    # 调用函数
    preprocessing_module = importlib.import_module(f"{module_path}.data_preprocessing")
    preprocessing_function = getattr(preprocessing_module, function_name)
    data = preprocessing_function(**cleaned_arguments)

# ⑤ 模型选择、训练，模型评估
model_name = data_dict["model_selection"]["name"]
model_arguments = data_dict["model_selection"]["arguments"]
cleaned_model_arguments = {k: v for k, v in model_arguments.items() if v != '' and v is not None and v != -1 and v != 0}
cleaned_model_arguments['data'] = data

# 调用模型选择模块中的模型函数
model_module = importlib.import_module(f"{module_path}.model_selection")
model_function = getattr(model_module, model_name)
evaluation_module = importlib.import_module(f"{module_path}.model_evaluation")  # 选择评估模块
evaluation_functions = [getattr(evaluation_module, fun) for fun in data_dict["model_evaluation"]]  # 获取评估函数
cleaned_model_arguments['evaluation_functions'] = evaluation_functions
model = model_function(**cleaned_model_arguments)


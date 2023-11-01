启动服务：

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```



func是一个列表，数组名:参数



```
class Function(BaseModel):
    name: str
    arguments: Dict[str, Any]


class Command(BaseModel):
    module: str
    functions: List[Function]  # functions现在是一个Function对象的列表


class JsonInfo(BaseModel):
    commands: List[Command]
```



测试的时候遇到一个问题

```json
{
    detail": "Missing optional dependency 'openpyxl'.  Use pip or conda to install openpyxl."
}
```

进入conda环境，

```bash
conda install openpyxl
```

##### total_execution.py

希望java后端发送的东西

###### @router.post("/execute")

```json
{
  "commands": [
    {
      "module": "import_data",
      "functions": [
        {
          "name": "import_excel_data",
          "arguments": {
            "file_path": "file_path.xlsx",
            "sheet_name": "Sheet1"
          }
        }
      ]
    },
    {
      "module": "data_preprocessing",
      "functions": [
        {
          "name": "process_data",
          "arguments": {
            "multiply_factor":2
          }
        }
      ]
    }
  ]
}

```



假设我们传递了一下这一坨参数。

```
{
  "commands": [
    {
      "module": "import_data",
      "functions": [
        {
          "name": "import_excel_data",
          "arguments": {
            "file_path": "file_path.xlsx",
            "sheet_name": "Sheet1"
          }
        }
      ]
    },
    {
      "module":"train",
      "functions":[
        {
          "name":"request_train",
          "arguments":{
            "input_epochs":1
          }
        }
      ]
    }
  ]
  }
```

这时候会调用train函数并且直接让epochs的值设置为1进行运行。当然，我们这里是有必要在逻辑上限制一下epoch的大小的。



之前的代码逻辑，我们

```python
if not data_total.empty:
    args['data'] = data_total
```

这样只要有参数返回，我就迭代，但是我们会发现训练的时候参数是自己调的，这会造成尴尬的局面。一种方法是接受空参数，但我懒得思考，所以直接添加了个判断，让它训练前保留着这个data_total不要出来害人，下次再用。

```python
if not data_total.empty and func_name != 'request_train':
    args['data'] = data_total
```

效果还是不错的，于是成功调用train进行训练。



这里是读数据+训练+预测

```json
{
  "commands": [
    {
      "module": "import_data",
      "functions": [
        {
          "name": "import_excel_data",
          "arguments": {
            "file_path": "file_path.xlsx",
            "sheet_name": "Sheet1"
          }
        }
      ]
    },
    {
      "module":"train",
      "functions":[
        {
          "name":"handwriting_train",
          "arguments":{
            "input_epochs":1
          }
        }
      ]
    },
    {
      "module":"predict",
      "functions":[
        {
          "name":"handwriting_predict",
          "arguments":{
          }
        }
      ]
    }
  ]
  }
```



# 11-01

跑通。

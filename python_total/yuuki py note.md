启动服务：

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

###### @router.post("/hi")

接收一个json

```
class JsonInfo(BaseModel):
    module: list
    function: list
    arguments: list
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

###### post /execute

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
    }
  ]
}

```




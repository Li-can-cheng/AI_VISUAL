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
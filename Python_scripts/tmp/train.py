import requests
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess

app = FastAPI()


class YourRequestObject(BaseModel):
    some_field: Optional[str] = None  # 这样，该字段就是可选的了


@app.post("/train")
async def train_model(request: YourRequestObject):
    print("训练开始... ")
    some_field = request.some_field

    command_list = ["python", "my_model.h5.py"]

    if some_field:
        command_list.append(some_field)

    # 根据是否有参数来调用脚本
    process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        print("训练结束...")
        return {"message": f"Model trained successfully with field: {some_field}"}
    else:
        return {"message":"6"}

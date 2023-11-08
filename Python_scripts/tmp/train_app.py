from fastapi import APIRouter
from pydantic import BaseModel
from fastapi import Body
import subprocess

router = APIRouter()


class YourRequestObject(BaseModel):
    value: str  # 假设这个字段是必需的


@router.post("/train")
async def train_model(request: YourRequestObject):
    print("训练开始... ")
    some_field = request.value

    command_list = ["python", "my_model.h5.py"]

    if some_field:
        command_list.append(some_field)

    process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        print("训练结束...")
        return {"message": f"Model trained successfully with field: {some_field}"}
    else:
        return {"message": "训练失败"}


import os
import shutil

from fastapi import *
from starlette.responses import JSONResponse

router = APIRouter()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        save_path = "uploads"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_path = os.path.join(save_path, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return JSONResponse(status_code=200, content={"message": "文件上传成功", "filename": file.filename})

    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"message": "文件上传失败"})

from fastapi import UploadFile, File, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST
from typing import Optional
import os
import shutil
import zipfile

router = APIRouter()

# 支持的图片格式
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    save_path = "uploads"
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, file.filename)

    # 检查文件扩展名
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext in IMAGE_EXTENSIONS:
        # 图片文件，直接保存
        dest_path = os.path.join(save_path, file.filename)
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": "图片上传成功。", "file_path": dest_path}

    elif file_ext == '.zip':
        # ZIP文件，解压缩
        try:
            extract_path = os.path.join("extracted_files", file.filename[:-4])
            os.makedirs(extract_path, exist_ok=True)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            training_set_folder = determine_training_set_folder(extract_path)
            return {"message": "ZIP文件上传并解压成功。", "training_set_folder": training_set_folder}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文件处理失败: {e}")

    else:
        # 不支持的文件类型
        raise HTTPException(status_code=400, detail="上传的文件类型不支持。")




def determine_training_set_folder(extract_path: str) -> Optional[str]:
    """判断解压后的文件夹结构，并找到训练集图片的文件夹。"""
    # 直接检查根目录下的第一层文件夹，返回遇到的第一个文件夹
    for entry in os.listdir(extract_path):
        potential_folder = os.path.join(extract_path, entry)
        if os.path.isdir(potential_folder):
            return potential_folder

    # 如果没有找到文件夹，返回None
    return None


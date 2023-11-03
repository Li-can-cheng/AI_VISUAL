from fastapi import APIRouter
import os

router = APIRouter()

@router.get("/files/{path:path}")
async def list_files(path: str):
    # 确保只有extracted_files目录可以被访问
    base_path = os.path.abspath("extracted_files")
    absolute_path = os.path.abspath(os.path.join(base_path, path))

    # 防止目录遍历漏洞
    if not absolute_path.startswith(base_path):
        return {"error": "Invalid directory path"}

    # 获取文件夹内容
    try:
        files = os.listdir(absolute_path)
        files_info = []
        for file in files:
            file_path = os.path.join(absolute_path, file)
            files_info.append({
                "name": file,
                "is_directory": os.path.isdir(file_path)
            })
        return files_info
    except FileNotFoundError:
        return {"error": "Directory not found"}

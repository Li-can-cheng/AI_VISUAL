import uvicorn
from fastapi import FastAPI


from total_execution import router as good_router
from upload import router as upload_router
from file_service import router as file_service

from fastapi.middleware.cors import CORSMiddleware


# 恭喜，你找到我了！请开始启动后端吧！


app = FastAPI()

app.include_router(good_router)
app.include_router(upload_router)
app.include_router(file_service)

# 设置允许的跨源请求的源列表——不然访问个der
# 你可以使用通配符 "*" 来允许所有源，或者使用具体的源列表
origins = [
    "*"
]

# 添加中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许的源列表
    allow_credentials=True,  # 支持cookies跨域
    allow_methods=["*"],  # 允许的HTTP方法
    allow_headers=["*"],  # 允许的HTTP请求头
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # 做个更新，可以用main函数直接启动——就是终端界面没有渲染，变丑了。
    # fastapi，启动！

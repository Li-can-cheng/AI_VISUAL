from fastapi import FastAPI
from yuuki_reload_execution import router as main_router
app = FastAPI()

app.include_router(main_router)
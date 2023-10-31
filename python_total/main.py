from fastapi import FastAPI
from total_execution import router as good_router
app = FastAPI()

app.include_router(good_router)
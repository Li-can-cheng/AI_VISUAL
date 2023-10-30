from fastapi import FastAPI
from predict_app import router as predict_router
from train_app import router as train_router

app = FastAPI()

app.include_router(predict_router)
app.include_router(train_router)

from fastapi import FastAPI

from yuuki_reload_execution import router as main_router
from ImageClassification.model_selection import router as websocket_router
app = FastAPI()

app.include_router(main_router)
app.include_router(websocket_router)




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
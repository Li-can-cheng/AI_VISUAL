from fastapi import FastAPI
from fastapi import Body

app = FastAPI()

@app.post("/trainModel")
async def trainModel(data=Body(None)):
    print(data)
    return {"result":data}
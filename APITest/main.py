from fastapi import FastAPI
from fastapi import Body

app = FastAPI()

@app.post("/trainModel")
async def trainModel(data=Body(None)):
    print(data)
    return {"result":data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
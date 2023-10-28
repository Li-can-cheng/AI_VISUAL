from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

model = load_model("S:\\myPython\\tmp\\my_model.h5")

class Item(BaseModel):
    data: list

@app.post("/predict/")
def predict(item: Item):
    array = np.array(item.data).reshape(1, 28, 28)
    prediction = model.predict(array).argmax()
    return {"result": int(prediction)}

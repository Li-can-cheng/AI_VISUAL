from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import cv2

router = APIRouter()

model = load_model("S:\\myPython\\tmp\\my_model.h5")

class Item(BaseModel):
    data: list

@router.post("/predict/")
def predict(item: Item):
    array = np.array(item.data)

    # 新增的缩放代码
    if array.shape != (28, 28):  # 只在需要时才缩放
        resized_array = cv2.resize(array, (28, 28))
        array = resized_array

    ready_array = array.reshape(1, 28, 28)
    prediction = model.predict(ready_array).argmax()
    return {"result": int(prediction)}

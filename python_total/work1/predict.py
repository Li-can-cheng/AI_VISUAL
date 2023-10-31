from tensorflow.keras.models import load_model
import cv2


def predictit():
    # 读取上传的文件
    model = load_model("S:\\myJAVA\\Visual-AI-Model-Development-Platform\\python_total\\work1\\my_model.h5")
    uploaded_file = cv2.imread("S:\\myJAVA\\Visual-AI-Model-Development-Platform\\upload\\4.jpg")
    if uploaded_file is None:
        raise ValueError("图像加载失败，请检查文件路径是否正确")

    # 转换图像为灰度图
    gray_img = cv2.cvtColor(uploaded_file, cv2.COLOR_BGR2GRAY)

    # 将图像缩放到28x28像素
    resized_img = cv2.resize(gray_img, (28, 28))

    # 调整图像的形状以符合模型的输入要求
    ready_array = resized_img.reshape(1, 28, 28)

    # 使用模型进行预测
    prediction = model.predict(ready_array).argmax()

    # 返回预测结果
    return {"result": [int(prediction)]}


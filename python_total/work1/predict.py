import string

import torch
from torchvision import transforms
from PIL import Image
from train import MNISTModel

model_path = 'my_pytorch_model.pt'


def handwriting_predict(image_path: string):
    print(image_path)
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 图片转换操作应与训练时相同
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载图像
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # 增加一个批次维度并转移到设备上

    # 加载模型
    model = MNISTModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切换到预测模式

    # 预测
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1, keepdim=True)  # 获取最可能的结果

    return prediction.item()


if __name__ == "__main__":
    predict_result = handwriting_predict('uploads\\4.jpg')
    print(f'Predicted class: {predict_result}')

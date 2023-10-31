import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


# 读取、调整尺寸和转换灰度
image = Image.open('AI_Sync\\Visual-AI-Model-Development-Platform\\upload\\4.jpg')  # 替换为实际的图片路径
image = image.resize((28, 28))  # 调整为MNIST数据集的图片尺寸，28x28
image = image.convert('L')  # 转换为灰度图像

# 转换为PyTorch的张量数据表示
transform = transforms.ToTensor()
input_tensor = transform(image).unsqueeze(0)  # 增加一个维度以适应模型输入的batch维度

# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 创建模型实例
model = CNN()

# 加载训练好的模型参数
path = 'model.pth'
model.load_state_dict(torch.load(path))  # 替换为实际的模型参数保存路径

# 模型预测
output = model(input_tensor)
prediction = output.argmax(dim=1).item()  # 获取预测结果的类别索引值

print('预测结果为:', prediction)

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def handwriting_train(input_epochs):
    # 设置PyTorch的device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda"
    if device == "cuda":
        print("Using GPU")
    else:
        print("Using CPU")

    # 定义数据集的转换操作
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),  # 如果需要的话，调整图片大小
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据集
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # 数据集分割
    train_size = int(0.8 * len(dataset))  # 假设80%的数据用于训练
    test_size = len(dataset) - train_size  # 剩余20%的数据用于测试
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

    # 构建模型
    model = MNISTModel().to(device)

    # 编译模型
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # 训练模型
    model.train()
    for epoch in range(input_epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())

    # 保存模型
    torch.save(model.state_dict(), 'my_pytorch_model.pt')
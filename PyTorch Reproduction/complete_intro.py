"""
Introduction to PyTorch - Complete Example
==========================================
1. Tensor basics
2. Autograd (PyTorch native)
3. Build & train CNN on FashionMNIST
4. Evaluate & visualize
5. Transfer learning with ResNet50
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# 1. Tensor Basics (演示用)
# --------------------------
def tensor_demo():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    print("x:", x)

    y = torch.zeros(2, 2)
    z = torch.ones(2, 2)
    w = torch.rand(2, 2)
    print("y:", y, "\nz:", z, "\nw:", w)

    np_array = np.array([1, 2, 3])
    x_np = torch.from_numpy(np_array)
    print("x_np:", x_np)


# -------------------------------------
# 2. PyTorch Autograd (无需 autograd 库)
# -------------------------------------
def autograd_demo():
    x = torch.ones(5)
    y = torch.zeros(3)
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)

    z = torch.matmul(x, w) + b
    loss = F.binary_cross_entropy_with_logits(z, y)
    loss.backward()
    print("w.grad:", w.grad)
    print("b.grad:", b.grad)

    # 禁用梯度追踪
    with torch.no_grad():
        z = torch.matmul(x, w) + b
    print("z.requires_grad inside no_grad:", z.requires_grad)


# --------------------------
# 3. FashionMNIST CNN 训练
# --------------------------
class FashionMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def prepare_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=512, shuffle=True, num_workers=0)  # Windows 下 num_workers=0
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=512, shuffle=False, num_workers=0)

    return train_loader, test_loader


def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader = prepare_dataloaders()
    net = FashionMNISTNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # 训练 5 个 epoch
    net.train()
    for epoch in range(5):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/5 - Loss: {running_loss/i:.4f}")

    # 评估
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100*correct/total:.2f}%")


# --------------------------
# 4. 迁移学习 ResNet50
# --------------------------
def transfer_learning_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device)

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换分类头（示例 5 类）
    n_inputs = model.fc.in_features
    n_classes = 5
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes)
    ).to(device)

    # 解冻新分类头
    for param in model.fc.parameters():
        param.requires_grad = True

    print("ResNet50 迁移学习模型已准备完毕。")
    print("可训练参数:", sum(p.numel() for p in model.parameters() if p.requires_grad))


# --------------------------
# 主入口
# --------------------------
if __name__ == '__main__':
    tensor_demo()
    autograd_demo()
    train_and_evaluate()
    transfer_learning_demo()
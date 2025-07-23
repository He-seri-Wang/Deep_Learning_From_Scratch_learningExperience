'''
Introduction to PyTorch
Pytorch is a popular neural net framework with the following features:

Automatic differentation
Compiling computation graphs
Libraries of algorithms and network primitives. Provides a high-level abstractions for working with neural networks.
Support for graphics processing units (GPU)
In this lesson, we will learn the basics of PyTorch. We will cover the following topics:

Tensors
Automatic differentation
Building a simple neural network
PyTorch Datasets and DataLoaders
Visualizing examples from the FashionMNIST Dataset
Training on CPU
Training on GPU
Using pre-trained weights
1. Tensors
Tensors are a specialized data structure very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model's parameters.

Tensors are similar to NumPy's ndarrays, except that tensors can run on GPUs or other hardware accelerators.

Initializing a Tensor
'''

import torch
import numpy as np
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights

if __name__ == '__main__':
    '''
    initializing a tensor
    '''
    # Create a tensor directly from data
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    print("x:", x)

    # Create a tensor of zeros
    y = torch.zeros(2, 2)
    print("y:", y)

    # Create a tensor of ones
    z = torch.ones(2, 2)
    print("z:", z)

    # Create a random tensor
    w = torch.rand(2, 2)
    print("w:", w)

    # Create a tensor from a NumPy array
    np_array = np.array([1, 2, 3])
    x_np = torch.from_numpy(np_array)
    print("x_np:", x_np)
    tensor = torch.tensor([[1, 2, 3], [3, 4, 5]])

    '''attributes of a tensor'''
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

    '''operations on tensors'''
    # Move the tensor to GPU if available
    if torch.cuda.is_available():
        tensor = tensor.to("cuda")

    # Standard numpy-like indexing and slicing
    tensor = torch.tensor([[1, 2, 3], [3, 4, 5]])
    print("First row: ", tensor[0])
    print("First column: ", tensor[:, 0])

    # Matrix multiplication
    tensor = torch.ones(3, 3)
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)
    print("y1: ", y1)
    print("y2: ", y2)

    # Element wise product
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)
    print("z1: ", z1)
    print("z2: ", z2)

    # common functions
    a = torch.rand(2, 4) * 2 - 1
    print('Common functions:')
    print(torch.abs(a))
    print(torch.ceil(a))
    print(torch.floor(a))
    print(torch.clamp(a, -0.5, 0.5))

    # Reshape
    a = torch.arange(4.)
    a_reshaped = torch.reshape(a, (2, 2))
    b = torch.tensor([[0, 1], [2, 3]])
    b_reshaped = torch.reshape(b, (-1,))
    print("a_reshaped", a_reshaped)
    print("b_reshaped", b_reshaped)

    '''tensor broadcasting'''
    x1 = torch.tensor([[1, 2, 3], [3, 4, 5]])
    x2 = torch.tensor([2, 2, 2])
    doubled = x1 * x2
    print(doubled)

    '''2. Automatic Differentation
    ... (same as your original text) ... '''

    '''2.1 Autograd ... (same as your original text) ...'''

    '''2.2 PyTorch Automatic Differentation ... (same as your original text) ...'''

    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

    loss.backward()
    print(w.grad)
    print(b.grad)

    '''Disabling gradient tracking ... (same as your original text) ...'''

    '''3. Building a simple neural network'''

    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(2, 8, bias=False)
            self.fc2 = nn.Linear(8, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    net = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    data = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32)
    labels = torch.tensor([[0.], [1.]], dtype=torch.float32)

    for epoch in range(500):
        outputs = net(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/500], Loss: {loss.item():.4f}")

    '''4. PyTorch Datasets and DataLoaders ... (same as your original text) ...'''

    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform)

    # Windows-safe: use num_workers=0
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=512, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=512, shuffle=False, num_workers=0)

    print("Num training examples:", len(train_dataset))
    print("Num test examples:", len(test_dataset))

    classes = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    '''5. Visualizing Examples ... (same as your original text) ...'''

    def imshow(img):
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images[:16]))
    print(" -- ".join(f"{classes[labels[j]]}" for j in range(8)))
    print(" -- ".join(f"{classes[labels[j]]}" for j in range(8, 16)))

    '''6. Training on the CPU = Slow! ... (same as your original text) ...'''

    class FashionMNISTNet(nn.Module):
        def __init__(self):
            super(FashionMNISTNet, self).__init__()
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

    net = FashionMNISTNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    print("Training finished.")

    '''7. Training on the GPU = Faster! ... (same as your original text) ...'''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    net = FashionMNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    print("Training finished.")
    '''Now that we have trained our neural network, let's evaluate its performance on the test dataset.'''
    # Test the neural network
    correct = 0
    total = 0

    # Set the model to evaluation mode
    net.eval()

    # Disable gradient calculation
    with torch.no_grad():
        for inputs, labels in test_loader:

            # Move the inputs and labels to the GPU if available
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = net(inputs)

            # Get the predicted class
            _, predicted = torch.max(outputs.data, 1)

            # Update the total number of samples and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate the accuracy
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    '''Not bad!
    Let's inspect number of total parameters and training parameters in the model:'''
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    '''Now that we have a trained model, if we want to adapt the model to another dataset with only 5 classes, we can freeze earlier layers and only train on the last fully-connected layer.'''
    # Freeze earlier layers
    for param in net.parameters():
        param.requires_grad = False

    n_inputs = net.fc3.in_features
    n_classes = 5
    net.fc3 = nn.Linear(n_inputs, n_classes)
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
        
    '''8. Pre-trained weights PyTorch has many pretrained models we can use. All of these models have been trained on Imagenet which consists of millions of images across 1000 categories. We want to freeze the early layers of these pretrained models and replace the classification module with our own.

PyTorch API for using pre-trained weights: https://pytorch.org/vision/stable/models.html

The approach for using a pre-trained image recognition model is well-established:

Load in pre-trained weights from a network trained on a large dataset
Freeze all the weights in the lower (convolutional) layers
Layers to freeze can be adjusted depending on similarity of task to large training dataset
Replace the classifier (fully connected) part of the network with a custom classifier
Number of outputs must be set equal to the number of classes
Train only the custom classifier (fully connected) layers for the task
Optimizer model classifier for smaller dataset
We will demonstrate an example of loading a pre-trained Resnet model.'''
    from torchvision import models

    model = models.resnet50(pretrained=True)

    print(model)

    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))


    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    print(model)

    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.fc.in_features
    n_classes = 5
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, n_classes), nn.LogSoftmax(dim=1)
    )
    model.fc.to(device)
    '''for p in model.parameters():
        p.requires_grad = False
    import torch.nn as nn
    num_classes = 5          # your dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)'''
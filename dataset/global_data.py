"""import pandas as pd
import numpy as np

def load_mnist_csv(file_path):
    df = pd.read_csv(file_path)  # Read CSV file into a DataFrame
    labels = df['label'].values  # Extract labels (the target column)
    images = df.drop('label', axis=1).values / 255.0  # Normalize pixel values to range [0, 1]

    # If one-hot encoding is needed:
    labels_one_hot = np.zeros((labels.size, 10))  # Initialize one-hot encoded labels
    for idx, row in enumerate(labels_one_hot):
        row[labels[idx]] = 1  # Set the appropriate index to 1 for each label

    return images, labels_one_hot

# Example usage:
file_path = r'D:\CS231nProject\DeepLearningFromScratch-master\MNIST\mnist_train.csv'
images, labels = load_mnist_csv(file_path)
print(images.shape, labels.shape)


#书上的读入MNIST数据示例
import sys, os
sys.path.append(os.pardir) # 为了导入父目录中的文件而进行的设定
from dataset.mnist import load_mnist
# 第一次调用会花费几分钟 ......
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
normalize=False)
# 输出各个数据的形状
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000,)
print(x_test.shape) # (10000, 784)
print(t_test.shape) # (10000,)
"""

#GPT重写的读取本地MNIST数据
import pandas as pd
import numpy as np
import sys, os
"""
# Function to load MNIST data from a CSV file
def load_mnist_csv(file_path):
    df = pd.read_csv(file_path)  # Read CSV file into a DataFrame
    labels = df['label'].values  # Extract labels (the target column)
    images = df.drop('label', axis=1).values / 255.0  # Normalize pixel values to range [0, 1]

    # If one-hot encoding is needed:
    labels_one_hot = np.zeros((labels.size, 10))  # Initialize one-hot encoded labels
    for idx, row in enumerate(labels_one_hot):
        row[labels[idx]] = 1  # Set the appropriate index to 1 for each label

    return images, labels_one_hot
"""
"""
# dataset/global_data.py
import pandas as pd
import numpy as np

def load_mnist_csv(file_path, flatten=True, normalize=True):
    df = pd.read_csv(file_path)  # 读取CSV文件到DataFrame
    labels = df['label'].values  # 提取标签（目标列）
    images = df.drop('label', axis=1).values  # 获取图像数据

    if normalize:
        images = images / 255.0  # 归一化像素值到[0, 1]

    if flatten:
        images = images.reshape(-1, 784)  # 将28x28的图像展平为一维数组

    # one-hot编码标签
    labels_one_hot = np.zeros((labels.size, 10))
    for idx, row in enumerate(labels_one_hot):
        row[labels[idx]] = 1

    return images, labels_one_hot

# Load data from CSV file
file_path = r'D:\CS231nProject\DeepLearningFromScratch-master\MNIST\mnist_train.csv'
images, labels = load_mnist_csv(file_path)
print(images.shape, labels.shape)

# Now, we are assuming you want to integrate this into your code that uses 'load_mnist'
# Adjusting the previous call to load data from the CSV directly instead of from a pickle

# Assuming you want to skip the loading from the original mnist.py and just use CSV for now
# Example usage for testing:
# 修复代码：返回值只需要一个变量来接收
images, labels = load_mnist_csv(r'D:\CS231nProject\DeepLearningFromScratch-master\MNIST\mnist_train.csv', flatten=True, normalize=False)

# 手动切分训练集和测试集
x_train, t_train = images[:60000], labels[:60000]  # 训练集
x_test, t_test = images[60000:], labels[60000:]  # 测试集

# 输出各个数据的形状
print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000, 10)
print(x_test.shape)   # (10000, 784)
print(t_test.shape)   # (10000, 10)


# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from global_data import load_mnist_csv
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 加载 MNIST 数据，传入 flatten 和 normalize 参数
(x_train, t_train), (x_test, t_test) = load_mnist_csv(r'D:\CS231nProject\DeepLearningFromScratch-master\MNIST\mnist_train.csv', flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 输出标签，例如 5

print(img.shape)  # 输出图像形状 (784,)
img = img.reshape(28, 28)  # 将图像恢复为28x28的形状
print(img.shape)  # 输出 (28, 28)

img_show(img)  # 显示图像
"""


import pandas as pd
import numpy as np

def load_mnist_csv(file_path, flatten=True, normalize=True, one_hot_label=False):
    df = pd.read_csv(file_path)
    labels = df['label'].values
    images = df.drop('label', axis=1).values

    if normalize:
        images = images / 255.0

    if flatten:
        images = images.reshape(-1, 784)

    if one_hot_label:
        one_hot = np.zeros((labels.size, 10))
        one_hot[np.arange(labels.size), labels] = 1
        labels = one_hot  # 替代掉原始 labels

    return (None, None), (images, labels)

def load_mnist_csv_all(file_path_train, file_path_test, flatten=True, normalize=True, one_hot_label=False):
    train_data = load_mnist_csv_single(file_path_train, flatten, normalize, one_hot_label)
    test_data = load_mnist_csv_single(file_path_test, flatten, normalize, one_hot_label)
    return train_data, test_data

def load_mnist_csv_single(file_path, flatten=True, normalize=True, one_hot_label=False):
    df = pd.read_csv(file_path)
    labels = df['label'].values
    images = df.drop('label', axis=1).values

    if normalize:
        images = images / 255.0

    if flatten:
        images = images.reshape(-1, 784)

    if one_hot_label:
        one_hot = np.zeros((labels.size, 10))
        one_hot[np.arange(labels.size), labels] = 1
        labels = one_hot  # 替代掉原始 labels

    return (images, labels)

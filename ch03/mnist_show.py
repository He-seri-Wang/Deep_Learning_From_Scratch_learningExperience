"""
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
#from dataset.mnist import load_mnist


# 添加global_data.py所在的路径
sys.path.append(os.path.abspath('D:\CS231nProject\DeepLearningFromScratch-master\dataset'))

from global_data import load_mnist_csv

from PIL import Image
 
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist_csv(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)  # (28, 28)

img_show(img)
"""
import sys
import os

# 1. 获取当前脚本目录
current_dir = os.path.dirname(__file__)

# 2. 组合 dataset 目录绝对路径（从当前脚本的父目录加上 dataset）
dataset_dir = os.path.abspath(os.path.join(current_dir, '..', 'dataset'))

# 3. 把 dataset 目录加入 sys.path
if dataset_dir not in sys.path:
    sys.path.append(dataset_dir)

# 4. 导入你自己写的模块
from global_data import load_mnist_csv

# 5. 下面正常使用
file_train = os.path.abspath(os.path.join(current_dir, '..', 'MNIST', 'mnist_train.csv'))
file_test = os.path.abspath(os.path.join(current_dir, '..', 'MNIST', 'mnist_test.csv'))

# 载入训练和测试数据
x_train, t_train = load_mnist_csv(file_train)
x_test, t_test = load_mnist_csv(file_test)

print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)

# 你的显示代码（比如用 PIL 显示第一个图像）
from PIL import Image
import numpy as np

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img.reshape(28,28)*255))  # 乘255还原像素值范围
    pil_img.show()

img_show(x_train[0])
print(t_train[0])  # 打印对应标签（one-hot编码，可以用 np.argmax 还原数字）
print(np.argmax(t_train[0]))

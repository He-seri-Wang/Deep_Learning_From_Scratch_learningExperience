
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from dataset.global_data import load_mnist_csv_all
from two_layer_net import TwoLayerNet
# 设置数据集路径
train_csv = r'D:\CS231nProject\DeepLearningFromScratch-master\MNIST\mnist_train.csv'
test_csv = r'D:\CS231nProject\DeepLearningFromScratch-master\MNIST\mnist_test.csv'

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist_csv_all(train_csv, test_csv, normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
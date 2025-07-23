# coding: utf-8
# neuralnet_mnist.py 顶部统一整理好路径
import sys, os
import numpy as np
import pickle

# 设置路径
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
dataset_dir = os.path.join(parent_dir, 'dataset')
common_dir = os.path.join(parent_dir, 'common')
sys.path.append(parent_dir)  # 加上整个项目根路径，方便 dataset 能被正确识别为模块

for path in [dataset_dir, common_dir]:
    if path not in sys.path:
        sys.path.append(path)

# 导入你自己的模块
from global_data import load_mnist_csv
from functions import sigmoid, softmax
'''
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

'''
"""
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist_csv(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test
"""
def get_data():
    file_path = os.path.abspath(os.path.join(current_dir, '..', 'MNIST', 'mnist_test.csv'))
    (_, _), (x_test, t_test) = load_mnist_csv(
        file_path=file_path,
        normalize=True,
        flatten=True,
        one_hot_label=False
    )
    return x_test, t_test
'''
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network
'''
def init_network():
    weight_path = os.path.join(current_dir, "sample_weight.pkl")
    with open(weight_path, 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100 # 批数量
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

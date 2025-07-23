# neuralnet_mnist.py 顶部统一整理好路径
import sys, os
import numpy as np
import pickle

# 设置路径
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
dataset_dir = os.path.join(parent_dir, 'dataset')
common_dir = os.path.join(parent_dir, 'common')

for path in [dataset_dir, common_dir]:
    if path not in sys.path:
        sys.path.append(path)

# 导入你自己的模块
from global_data import load_mnist_csv
from functions import sigmoid, softmax

'''
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test
'''
def get_data():
    file_test = os.path.abspath(os.path.join(current_dir, '..', 'MNIST', 'mnist_test.csv'))
    x_test, t_test = load_mnist_csv(file_test, flatten=True, normalize=True)
    t_test = np.argmax(t_test, axis=1)  # 如果是 one-hot
    return x_test, t_test

def init_network():
    with open("ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

#main
x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
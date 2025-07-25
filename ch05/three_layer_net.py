
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class ThreeLayerNet:

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, weight_init_std=0.01):        # 初始化权重
            self.params = {}
            self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size1)
            self.params['b1'] = np.zeros(hidden_size1)
            self.params['W2'] = weight_init_std * np.random.randn(hidden_size1, hidden_size2) 
            self.params['b2'] = np.zeros(hidden_size2)
            self.params['W3'] = weight_init_std * np.random.randn(hidden_size2, output_size)
            self.params['b3'] = np.zeros(output_size)

            # 生成层(1 more layer)
            self.layers = OrderedDict()
            self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
            self.layers['Relu1'] = Relu()
            self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
            self.layers['Relu2'] = Relu()
            self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])

            self.lastLayer = SoftmaxWithLoss()
            
    def predict(self, x):
            for layer in self.layers.values():
                x = layer.forward(x)
            
            return x
            
        # x:输入数据, t:监督数据
    def loss(self, x, t):
            y = self.predict(x)
            return self.lastLayer.forward(y, t)
        
    def accuracy(self, x, t):
            y = self.predict(x)
            y = np.argmax(y, axis=1)
            if t.ndim != 1 : t = np.argmax(t, axis=1)
            
            accuracy = np.sum(y == t) / float(x.shape[0])
            return accuracy
            
        # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
            loss_W = lambda W: self.loss(x, t)
            
            grads = {}
            grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
            grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
            grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
            grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
            grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
            grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
            
            return grads
            
    def gradient(self, x, t):
            # forward
            self.loss(x, t)

            # backward
            dout = 1
            dout = self.lastLayer.backward(dout)
            
            layers = list(self.layers.values())
            layers.reverse()
            for layer in layers:
                dout = layer.backward(dout)

            # 设定
            grads = {}
            grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
            grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
            grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db
            return grads

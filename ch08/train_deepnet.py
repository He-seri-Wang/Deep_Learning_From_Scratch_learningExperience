'''# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer'''


# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import matplotlib.pyplot as plt
from dataset.global_data import load_mnist_csv_all
from deep_convnet import DeepConvNet
from common.trainer import Trainer

# 设置数据集路径
train_csv = r'D:\CS231nProject\DeepLearningFromScratch-master\MNIST\mnist_train.csv'
test_csv = r'D:\CS231nProject\DeepLearningFromScratch-master\MNIST\mnist_test.csv'

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist_csv_all(train_csv, test_csv, flatten=False, normalize=True, one_hot_label=True)
x_train = x_train.reshape(-1, 1, 28, 28)
x_test  = x_test.reshape(-1, 1, 28, 28)
'''
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
'''
network = DeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 保存参数
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")

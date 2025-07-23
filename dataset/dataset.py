# dataset.py
import pandas as pd
import numpy as np

def load_mnist_csv(file_path):
    df = pd.read_csv(file_path)  # 从CSV文件中读取数据
    labels = df['label'].values  # 提取标签（目标列）
    images = df.drop('label', axis=1).values / 255.0  # 对像素值进行归一化处理，范围[0, 1]

    # 如果需要one-hot编码标签：
    labels_one_hot = np.zeros((labels.size, 10))  # 初始化one-hot编码标签
    for idx, row in enumerate(labels_one_hot):
        row[labels[idx]] = 1  # 为每个标签设置适当的索引为1

    return images, labels_one_hot

# coding: utf-8
"""
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np


url_base = 'https://web.archive.org/web/20220331130319/https://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    
def download_mnist():
    for v in key_file.values():
       _download(v)
        
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data
    
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
  #  读入MNIST数据集
    
   # Parameters
  #  ----------
  #  normalize : 将图像的像素值正规化为0.0~1.0
  #  one_hot_label : 
  #      one_hot_label为True的情况下，标签作为one-hot数组返回
   #     one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
   # flatten : 是否将图像展开为一维数组
    
   # Returns
   # -------
   # (训练图像, 训练标签), (测试图像, 测试标签)
  #  """
"""
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 

"""
"""
if __name__ == '__main__':
    init_mnist()

"""
import os
import pandas as pd
import numpy as np
import pickle

# Paths to the CSV files
train_csv = 'D:/CS231nProject/DeepLearningFromScratch-master/MNIST/mnist_train.csv'
test_csv = 'D:/CS231nProject/DeepLearningFromScratch-master/MNIST/mnist_test.csv'

save_file = 'mnist.pkl'  # The pickle file where the dataset will be saved

# Function to load CSV and return images and labels
def load_mnist_csv(file_path, normalize=True, one_hot_label=True):
    # Read CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Extract labels (the target column)
    labels = df['label'].values

    # Extract pixel values and normalize them (scale to [0, 1])
    images = df.drop('label', axis=1).values / 255.0  # Normalize images

    # One-hot encoding for labels
    if one_hot_label:
        labels_one_hot = np.zeros((labels.size, 10))
        for idx, row in enumerate(labels_one_hot):
            row[labels[idx]] = 1
    else:
        labels_one_hot = labels

    return images, labels_one_hot

# Function to convert MNIST dataset into a dictionary and save it as a pickle file
def init_mnist():
    print("Loading train data from CSV...")
    train_images, train_labels = load_mnist_csv(train_csv)
    print("Loading test data from CSV...")
    test_images, test_labels = load_mnist_csv(test_csv)

    # Prepare the dataset dictionary
    dataset = {
        'train_img': train_images,
        'train_label': train_labels,
        'test_img': test_images,
        'test_label': test_labels
    }

    # Save the dataset to a pickle file
    print("Creating pickle file...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

# Function to load the MNIST dataset from the pickle file
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    # Normalize images if required
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    # One-hot encode labels if required
    if one_hot_label:
        dataset['train_label'] = dataset['train_label']
        dataset['test_label'] = dataset['test_label']

    # Reshape images if required (flatten them into a vector)
    if flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 784)  # 28x28 image flattened to 784

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

# Example usage:
if __name__ == '__main__':
    init_mnist()  # Initialize and save the dataset to a pickle file
    # Load dataset (normalization and flattening are default)
    (train_img, train_label), (test_img, test_label) = load_mnist()
    
    print("Training images shape:", train_img.shape)
    print("Training labels shape:", train_label.shape)
    print("Test images shape:", test_img.shape)
    print("Test labels shape:", test_label.shape)


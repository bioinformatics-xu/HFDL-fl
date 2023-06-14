import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import gzip
import os
import pandas as pd
import torchvision
#import cv2
import matplotlib.pyplot as plt


"""
    批训练，把数据变成一小批一小批数据进行训练。
    DataLoader就是用来包装所使用的数据，每次抛出一批数据
"""
# import torch
# import torch.utils.data as Data
#
# import pandas as pd
# train_data = pd.read_csv('data/mnist_train.csv')
# print("训练数据的形状:", train_data.shape)
# test_data = pd.read_csv('data/mnist_test.csv')
# print("测试数据的形状:",test_data.shape)
#
# from keras.utils import to_categorical
# train_y = train_data.iloc[:,0]
# train_X = train_data.iloc[:,1:]
# train_X = train_X.values.reshape(-1,28,28)
# #train_y = to_categorical(train_y, 10)
# train_X = train_X.astype('float32')
# train_X /= 255
#
# test_y = test_data.iloc[:,0]
# test_X = test_data.iloc[:,1:]
# test_X = test_X.values.reshape(-1,28,28)
# #test_y = to_categorical(test_y, 10)
# test_X = test_X.astype('float32')
# test_X /= 255
#
# train_y = torch.tensor(train_y)
# test_y = torch.tensor(test_y)
# train_X = torch.from_numpy(train_X)
# test_X = torch.from_numpy(test_X)


# torch_dataset_train = Data.TensorDataset(train_X, train_y)
# torch_dataset_test = Data.TensorDataset(test_X, test_y)

# BATCH_SIZE = 64
# train_loader = Data.DataLoader(
#     # 从数据库中每次抽出batch size个样本
#     dataset=torch_dataset_train,
#     batch_size=BATCH_SIZE,
#     shuffle=True
#     # num_workers=2,
# )
#
# test_loader = Data.DataLoader(
#     # 从数据库中每次抽出batch size个样本
#     dataset=torch_dataset_test,
#     batch_size=BATCH_SIZE,
#     shuffle=False
#     # num_workers=2,
# )
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# 设置网络结构
class Net(nn.Module):
    def __init__(self, in_features, out_features):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 32, 3) #32,26,26
        self.max_pool1 = nn.MaxPool2d(kernel_size=2) # 32, 13, 13
        self.conv2 = nn.Conv2d(32, 64, 3) # 64, 11, 11
        self.max_pool2 = nn.MaxPool2d(kernel_size=2) # 64, 5, 5
        self.conv3 = nn.Conv2d(64, 64, 3) # 64,3,3
        self.dnn1 = nn.Linear(64*3*3, 64) #第一层全连接层 64
        self.dnn2 = nn.Linear(64, out_features) #第二层全连接层 10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv3(x))
        # x = x.view(128,-1)
        x = x.view(-1,64*3*3)   # change，这样才能跟(64*3*3)匹配——self.dnn1 = nn.Linear(64*3*3, 64) #第一层全连接层 64
        x = F.relu(self.dnn1(x)) #relu**
        x = self.dnn2(x)
        return x

#************************************a2torchloadlocalminist*********************************************************
class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """

    def __init__(self, folder, transform=None):
        (train_set, train_labels) = load_data(folder)  # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)

def load_data(data_folder):
    train_data = pd.read_csv(data_folder)
    y_train = train_data.iloc[:,0]
    y_train = pd.Series.to_numpy(y_train)
    x_train = train_data.iloc[:,1:]
    x_train = x_train.values.reshape(-1, 28, 28).astype('float32')
    # x_train = x_train.values.reshape(-1, 28, 28, 1).astype('float32')
    return (x_train, y_train)

train_dataset = DealDataset('data/mnist_train.csv', transform=transforms.ToTensor())
test_dataset = DealDataset('data/mnist_test.csv', transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

net = Net(1, 10)

# 开始训练
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)

for epoch in range(5):
    running_loss, running_acc = 0.0, 0.0
    for i, data in enumerate(train_loader, 1):  # 可以改为0，是对i的给值(循环次数从0开始计数还是从1开始计数的问题)
        img, label = data
        out = net(img).float()
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * label.size(0)
        _, predicted = torch.max(out, 1)
        running_acc += (predicted==label).sum().item()
        print('Epoch [{}/5], Step [{}/{}], Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, i + 1, len(train_loader), loss.item(), (predicted==label).sum().item()/128))
        acc = (predicted==label).sum().item()/128
    #测试
    test_loss, test_acc = 0.0, 0.0
    for i, data in enumerate(test_loader):
        img, label = data

        out = net(img)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_loss += loss.item() * label.size(0)
        _, predicted = torch.max(out, 1)
        test_acc += (predicted == label).sum().item()

    print("Train {} epoch, Loss: {:.6f}, Acc: {:.6f}, Test_Loss: {:.6f}, Test_Acc: {:.6f}".format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset)),
        test_loss / (len(test_dataset)), test_acc / (len(test_dataset))))


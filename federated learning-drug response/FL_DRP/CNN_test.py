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

# 加载MNIST数据集

# #************************************a2torchloadlocalminist*********************************************************
class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """

    def __init__(self, folder, data_name, label_name, transform=None):
        (train_set, train_labels) = load_data(folder, data_name,
                                              label_name)  # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
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

def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return (x_train, y_train)

train_dataset = DealDataset('./data', "train-images-idx3-ubyte.gz",
                           "train-labels-idx1-ubyte.gz", transform=transforms.ToTensor())
test_dataset = DealDataset('./data', "t10k-images-idx3-ubyte.gz",
                           "t10k-labels-idx1-ubyte.gz", transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

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

net = Net(1, 10)

# 开始训练
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)

for epoch in range(5):
    running_loss, running_acc = 0.0, 0.0
    for i, data in enumerate(train_loader, 1):  # 可以改为0，是对i的给值(循环次数从0开始计数还是从1开始计数的问题)
        img, label = data
        out = net(img)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * label.size(0)
        _, predicted = torch.max(out, 1)
        running_acc += (predicted==label).sum().item()
        print('Epoch [{}/5], Step [{}/{}], Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, i + 1, len(train_loader), loss.item(), (predicted==label).sum().item()/128))
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










# 一维卷积神经网络结构
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=2)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(10, 20, 3, 2)
        self.max_pool2 = nn.MaxPool1d(3, 2)
        self.conv3 = nn.Conv1d(20, 40, 3, 2)

        self.liner1 = nn.Linear(40 * 67, 120)
        self.liner2 = nn.Linear(120, 84)
        self.liner3 = nn.Linear(84, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv3(x))

        x = x.view(-1, 40 * 67)
        x = F.relu(self.liner1(x))
        x = F.relu(self.liner2(x))
        x = self.liner3(x)

        return x

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
SEL_CEL = "Afatinib_1032"
DF = pd.read_csv("GDSC2_response.csv")
FilteredDF = DF[SEL_CEL]
Y = np.array(FilteredDF)

Feat_DF = pd.read_csv("GDSC2_filtered_featrue_var01.csv")
Cell_Features = Feat_DF

# Features
X = Cell_Features.values
X = X[:, 2:]
seed = 10
np.random.seed(seed)

# split training, validation and test sets based on each sample NSC ID
NSC_All = np.array(DF.index.values, dtype=int)
Train_Ind, Rest_Ind, Y_Train, Y_Rest = train_test_split(NSC_All, Y, test_size=0.2, random_state=seed)
Validation_Ind, Test_Ind, Y_Validation, Y_Test = train_test_split(Rest_Ind, Y_Rest, test_size=0.5,
                                                                  random_state=seed)
# Sort the NSCs
Train_Ind = np.sort(Train_Ind)
Validation_Ind = np.sort(Validation_Ind)
Test_Ind = np.sort(Test_Ind)

# Extracting the drug descriptors of each set based on their associated NSCs
X_Train_Raw = Cell_Features.loc[Train_Ind]
X_Validation_Raw = Cell_Features.loc[Validation_Ind]
X_Test_Raw = Cell_Features.loc[Test_Ind]

Y_Train = FilteredDF.loc[Train_Ind];
Y_Train = np.array(Y_Train)
Y_Train = Y_Train.astype(float)
Y_Validation = FilteredDF.loc[Validation_Ind];
Y_Validation = np.array(Y_Validation)
Y_Validation = Y_Validation.astype(float)
Y_Test = FilteredDF.loc[Test_Ind];
Y_Test = np.array(Y_Test)
Y_Test = Y_Test.astype(float)

X_Dummy = X_Train_Raw.values;
X_Train = X_Dummy[:, 2:]
X_Train = X_Train.astype(float)
X_Dummy = X_Validation_Raw.values;
X_Validation = X_Dummy[:, 2:]
X_Validation = X_Validation.astype(float)
X_Dummy = X_Test_Raw.values;
X_Test = X_Dummy[:, 2:]
X_Test = X_Test.astype(float)

print(X_Train.shape, Y_Train.shape, X_Test.shape, Y_Test.shape)

#将numpy数据集转化为tensor
import torch

X_Train = torch.tensor(X_Train)
Y_Train = torch.tensor(Y_Train)
X_Test = torch.tensor(X_Test)
Y_Test = torch.tensor(Y_Test)

#训练集数据类型转化为tensor.float32
X_Train =torch.tensor(X_Train, dtype = torch.float32)
X_Test =torch.tensor(X_Test, dtype = torch.float32)

#改变X_Train和X_Test的形状
X_Train = X_Train.reshape(X_Train.shape[0], 1, 1, X_Train.shape[1])
X_Test = X_Test.reshape(X_Test.shape[0], 1, 1, X_Test.shape[1])
print(X_Train.shape, X_Test.shape)

# 以上给出的标签形状是Y_Train.shape = (644,)，类型是float，但是训练器需要的是(644,1),dtype=tensor.long
Y_Train = Y_Train.reshape(Y_Train.shape[0], 1)
Y_Test = Y_Test.reshape(Y_Test.shape[0], 1)

Y_Train = torch.tensor(Y_Train, dtype=torch.float32)
Y_Test = torch.tensor(Y_Test, dtype=torch.float32)

# 开始训练，定义损失函数与优化器
import torch.optim as optim
import time
from tqdm import tqdm

criterion = nn.MSELoss()
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
start = time.time()
for epoch in tqdm(range(10)):
    running_loss = 0
    for i, input_data in enumerate(X_Train, 0):
        label = Y_Train[i]
        optimizer.zero_grad()
        outputs = net(input_data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %0.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('time = %2dm:%2ds' % ((time.time() - start) // 60, (time.time() - start) % 60))

def calculate_r_square(output, target):
    return 1 - torch.div(torch.sum((output - target).pow(2)),
                         torch.sum((target - target.mean()).pow(2)))

X_Test = X_Test.reshape(X_Test.shape[0], 1, X_Test.shape[3])
Y_Test = Y_Test.reshape(Y_Test.shape[0])

outputs = net(X_Test)
outputs[:,1]
print(calculate_r_square(outputs[:,1], Y_Test))
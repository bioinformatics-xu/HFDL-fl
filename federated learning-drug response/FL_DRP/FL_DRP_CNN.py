import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

def convert_return(feature_origin, y_origin, target_drug):
    # dataframe = pd.DataFrame(columns=[['y_tag'] + ['f' + range(feature_origin.shape[1]*(feature_origin.shape[1]+1))] + ])
    dataframe = pd.DataFrame(columns = range(feature_origin.shape[1]*(feature_origin.shape[1]+1) + 2))
    for i in feature_origin.index:
        # print(i)
        # feature_origin.shape[1]
        for j in range(10):
            gene_dummy_j = gene_dummy
            gene_dummy_j['gene_value'] = feature_origin.loc[i, :].tolist()
            gene_dummy_j = gene_dummy_j.reindex(np.random.permutation(gene_dummy_j.index))
            # gene_dummy_j = ((gene_dummy.T).reindex(np.random.permutation((gene_dummy.T).index))).T
            gene_dummy_j_row = gene_dummy_j.values.reshape(1, -1).tolist()
            gene_dummy_j_row_list = sum(gene_dummy_j_row,[])
            gene_dummy_j_row = [y_origin.loc[i,target_drug]] + gene_dummy_j_row_list + [i]
            # gene_dummy_j_row = [y_origin.loc[i, 'Camptothecin_1003']] + gene_dummy_j_row_list
            dataframe.loc[len(dataframe)] = gene_dummy_j_row
    return dataframe

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

def load_data(data_set):
    train_data = data_set
    y_train = train_data.iloc[:,0]
    y_train = pd.Series.to_numpy(y_train)
    x_train = train_data.iloc[:,1:]
    x_train = x_train.values.reshape(-1, 113, 114).astype('float32')
    # x_train = x_train.values.reshape(-1, 28, 28, 1).astype('float32')
    return (x_train, y_train)

class Net(nn.Module):
    def __init__(self, in_features, out_features):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 32, kernel_size = (3,4),stride=2) #32,56,56
        self.max_pool1 = nn.MaxPool2d(kernel_size=2) # 32, 28, 28
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 2, stride = 1) # 64, 27, 27
        self.max_pool2 = nn.MaxPool2d(kernel_size=2) # 64, 13, 13
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 2, stride = 1) # 64,12,12
        self.max_pool3 = nn.MaxPool2d(kernel_size=2)  # 64, 6, 6
        self.conv4 = nn.Conv2d(64, 64, kernel_size=2, stride=2)  # 64,3,3
        self.dnn1 = nn.Linear(64*3*3, 64) #第一层全连接层 64
        self.dnn2 = nn.Linear(64, out_features) #第二层全连接层 2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv3(x))
        x = self.max_pool3(x)
        x = F.relu(self.conv4(x))
        # x = x.view(128,-1)
        x = x.view(-1,64*3*3)   # change，这样才能跟(64*3*3)匹配——self.dnn1 = nn.Linear(64*3*3, 64) #第一层全连接层 64
        x = F.relu(self.dnn1(x)) #relu**
        x = self.dnn2(x)
        return x


cell_features = pd.read_csv('GDSC2_Expr_CGC_feature.csv')
X_total = cell_features.iloc[:,1:]
y_total = pd.read_csv("GDSC2_response_class.csv")
drug_names = y_total.columns.tolist()
del(drug_names[0])

for SEL_CEL in drug_names:
    y_tag = pd.DataFrame(y_total[SEL_CEL])
    rfmodel = RandomForestClassifier(criterion='gini',
                                     n_estimators=500,
                                     random_state=1,
                                     n_jobs=2,
                                     oob_score=True)
    # 拟合数据
    rfmodel.fit(X_total, np.array(y_tag[SEL_CEL]))

    # 提取特征重要性
    importance = pd.Series(rfmodel.feature_importances_, index=X_total.columns)
    importance = importance.sort_values(ascending=False)
    x_selected_columns = importance.index[0:int(round(X_total.shape[1] * 0.2))]
    x_origin = pd.DataFrame(X_total, columns=x_selected_columns)
    x_origin = x_origin.sort_index(axis=1)

    # x为数据集的feature，y为label.
    x_train, x_test, y_train, y_test = train_test_split(x_origin, y_tag, test_size=0.3, shuffle=True)

    gene_features = x_origin.columns
    gene_dataFrame = pd.DataFrame({"genes": gene_features})
    gene_dummy = pd.get_dummies(gene_dataFrame["genes"], prefix=None, dummy_na=False, drop_first=False)
    gene_dummy = gene_dummy.loc[:, gene_features]

    train_dataset_long = convert_return(x_train, y_train, SEL_CEL)
    train_sample = train_dataset_long.iloc[:, -1]
    train_dataset_long = train_dataset_long.iloc[:, :-1]

    test_dataset_long = convert_return(x_test, y_test, SEL_CEL)
    test_sample = train_dataset_long.iloc[:, -1]
    test_dataset_long = test_dataset_long.iloc[:, :-1]

    train_dataset = DealDataset(train_dataset_long, transform=transforms.ToTensor())
    test_dataset = DealDataset(test_dataset_long, transform=transforms.ToTensor())

    bacth_size_n = 128

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bacth_size_n, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bacth_size_n, shuffle=False)

    # 设置网络结构

    net = Net(1, 2)

    # 开始训练
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.02)

    for epoch in range(5):
        print('Epoch [{}/5]'.format(epoch + 1))
        with open('./result_FL_DRP_CNN.txt', 'a') as f:
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
                running_acc += (predicted == label).sum().item()
                print('Epoch [{}/5], Step [{}/{}], Loss: {:.6f}, Acc: {:.6f}'.format(
                    epoch + 1, i + 1, len(train_loader), loss.item(), (predicted == label).sum().item() / bacth_size_n))
                acc = (predicted == label).sum().item() / bacth_size_n
            # 测试
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
            Acc = running_acc / (len(train_dataset))
            test_loss_total = test_loss / (len(test_dataset))
            test_acc_total = test_acc / (len(test_dataset))
            f.write("Train {} epoch, Loss: {:.6f}, Acc: {:.6f}, Test_Loss: {:.6f}, Test_Acc: {:.6f}".format(
                epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset)),
                test_loss / (len(test_dataset)), test_acc / (len(test_dataset))) + '\n')
            print(SEL_CEL)
            print("Train {} epoch, Loss: {:.6f}, Acc: {:.6f}, Test_Loss: {:.6f}, Test_Acc: {:.6f}".format(
                epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset)),
                test_loss / (len(test_dataset)), test_acc / (len(test_dataset))))








# path2 = 'data'
# def create_csv(X,csv_head):
#     path = str(path2+"/"+"%s.csv" % (X)) #同目录下的新文件名
#     with open(path,'w',newline = '',encoding='utf-8_sig') as f:
#         csv_write = csv.writer(f)
#         csv_write.writerow(csv_head)#写入表头
#
# def write_csv(X,data_row):
#     path = str(path2+"/"+"%s.csv" % (X))#
#     with open(path,mode='a',newline = '',encoding='utf-8_sig') as f:
#         csv_write = csv.writer(f)
#         csv_write.writerow(data_row)

# create_csv('GDSC_random_train', (['y_tag'] + gene_features.values.tolist() + ['gene_value'] + ['sample']) )
# create_csv('GDSC_random_test', (['y_tag'] + gene_features.values.tolist() + ['gene_value'] + ['sample']) )



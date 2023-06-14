import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import crypten
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split


cell_features = pd.read_csv('GDSC2_Expr_CGC_feature.csv')
x_origin = cell_features.iloc[:, 1:]
gene_features = x_origin.columns
gene_dataFrame = pd.DataFrame({"genes":gene_features})
gene_dummy = pd.get_dummies(gene_dataFrame["genes"], prefix = None, dummy_na = False, drop_first = False)
gene_dummy = gene_dummy.loc[:,gene_features]

class Setting:
    """Parameters for training"""

    def __init__(self):
        self.epoch = 20
        self.lr = 0.05
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.batch_size = 128


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_size, 4000)
#         self.fc2 = nn.Linear(4000, 2000)
#         self.fc3 = nn.Linear(2000, 1000)
#         self.fc4 = nn.Linear(1000, 1000)
#         self.fc5 = nn.Linear(1000, 1)
#         self.dropout1 = nn.Dropout(p=0.25)
#         self.dropout2 = nn.Dropout(p=0.25)
#         self.dropout3 = nn.Dropout(p=0.25)
#         self.dropout4 = nn.Dropout(p=0.1)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(self.dropout1(x)))
#         x = F.relu(self.fc3(self.dropout2(x)))
#         x = F.relu(self.fc4(self.dropout3(x)))
#         x = nn.Sigmoid()(self.fc5(self.dropout4(x)))
#         return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(input_size, 2000)
        self.layer_2 = nn.Linear(2000, 1000)
        self.layer_out = nn.Linear(1000, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(2000)
        self.batchnorm2 = nn.BatchNorm1d(1000)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x

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

net = Net(input_size, 2)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    # correct_results_sum = np.sum(y_pred_tag == y_test)
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def divide_trainset_to_client(train_set, cli_num, BATCH_SIZE):
    length_list = []
    train_sets = []
    for _ in range(cli_num - 1):
        length_list.append(len(train_set) // cli_num)
    length_list.append(len(train_set) - (cli_num - 1)* (len(train_set) // cli_num))
    train_sets_pre = random_split(train_set, length_list)
    for i in train_sets_pre:
        train_sets.append(DataLoader(i, batch_size=BATCH_SIZE))
    return train_sets

def define_network(cli_num, lr_=0.05, momentum_=0.9, weight_decay_=0.0001):
    createVar = locals()
    optimizers = []
    models = []
    params = []
    for i in range(cli_num):
        k = str(i + 1)
        model_name = 'model_' + k
        opti_name = 'optimizer_' + k
        createVar[model_name] = Net().to(device)
        createVar[opti_name] = optim.SGD(
            locals()[model_name].parameters(),
            lr=lr_,momentum=momentum_,weight_decay=weight_decay_
            )
        models.append(locals()[model_name])
        params.append(list(locals()[model_name].parameters()))
        optimizers.append(locals()[opti_name])
    return models, optimizers, params


def fl_train(train_sets, fl_models, fl_optimizers, params):
    new_params = list()
    for k in range(len(train_sets)):
        for batch_idx, (data, target) in enumerate(train_sets[k]):
            fl_optimizers[k].zero_grad()
            data, target = data.to(device), target.to(device)
            output = fl_models[k](data)
            loss = criterion(output, target.unsqueeze(1))
            loss.backward()
            fl_optimizers[k].step()
    for param_i in range(len(params[0])):
        fl_params = list()
        for remote_index in range(cli_num):
            clone_param = params[remote_index][param_i].clone().cpu()
            fl_params.append(crypten.cryptensor(torch.tensor(clone_param)))
        sign = 0
        for i in fl_params:
            if sign == 0:
                fl_param = i
                sign = 1
            else:
                fl_param = fl_param + i

        new_param = (fl_param / cli_num).get_plain_text()
        new_params.append(new_param)

    with torch.no_grad():
        for model_para in params:
            for param in model_para:
                param *= 0

        for remote_index in range(cli_num):
            for param_index in range(len(params[remote_index])):
                new_params[param_index] = new_params[param_index].to(device)
                params[remote_index][param_index].set_(new_params[param_index])
    return fl_models


def train(dataloader, model, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target.unsqueeze(1))
        loss.backward()
        optimizer.step()
    return model


def test(test_x, test_y, model):
    model.eval()
    with torch.no_grad():
        output = model(test_x)
        acc = binary_acc(output.detach(), test_y).item()
        return acc

cell_features = pd.read_csv('GDSC2_Expr_CGC_feature.csv')
x_origin = cell_features.iloc[:, 1:]
drug_class = pd.read_csv('GDSC2_response_class.csv')

drug_names = drug_class.columns[1:]
drug_names = drug_names[0:3]
# y_origin = drug_class['Camptothecin_1003']


for drug in drug_names:
    y_origin = drug_class[drug]

    seed = 10
    np.random.seed(seed)
    x_train_origin, x_test_origin, y_train_origin, y_test_origin = train_test_split(x_origin, y_origin, test_size=0.3,
                                                                                    random_state=seed)

    # data_max = y_train_origin.max() if y_train_origin.max(
    # ) > y_test_origin.max() else y_test_origin.max()
    # data_min = y_train_origin.min() if y_train_origin.min(
    # ) < y_test_origin.min() else y_test_origin.min()
    # y_train_origin1 = y_train_origin.reshape((-1, 1))
    # y_test_origin1 = y_test_origin.reshape((-1, 1))

    train_x = torch.tensor(x_train_origin.to_numpy(), dtype=torch.float32)
    train_y = torch.tensor(y_train_origin.to_numpy(), dtype=torch.float32)
    test_x = torch.tensor(x_test_origin.to_numpy(), dtype=torch.float32)
    test_y = torch.tensor(y_test_origin.to_numpy(), dtype=torch.float32)
    #test_y = y_test_origin
    train_set = TensorDataset(train_x, train_y)

    crypten.init()
    args = Setting()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cli_num = 3
    epoch = args.epoch

    # criterion = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss()
    test_x, test_y = test_x.to(device), test_y.to(device)
    input_size = test_x.shape[1]
    train_sets = divide_trainset_to_client(
        train_set, cli_num, BATCH_SIZE=args.batch_size)

    models, optimizers, _ = define_network(
        cli_num, lr_=args.lr, momentum_=args.momentum, weight_decay_=args.weight_decay)
    fl_models, fl_optimizers, params = define_network(
        cli_num, lr_=args.lr, momentum_=args.momentum, weight_decay_=args.weight_decay)

    with open('./result.txt', 'a') as f:
        for n in range(cli_num):
            max_acc = 0
            for _ in range(epoch):
                model = train(train_sets[n], models[n], optimizers[n])
                acc = test(test_x, test_y.unsqueeze(1), model)
                if acc > max_acc:
                    max_acc = acc
                    torch.save(model.state_dict(),
                               './' + drug + '_model_' + str(n + 1) + '.pkl')
            #f.read()
            f.write(drug + '_client_' + str(n + 1) +
                    ' accuracy: ' + str(round(max_acc, 4)) + '\n')
        fl_max_acc = 0
        for _ in range(epoch):
            fl_models = fl_train(train_sets, fl_models, fl_optimizers, params)
            fl_acc = test(test_x, test_y.unsqueeze(1), fl_models[0])
            if fl_acc > fl_max_acc:
                fl_max_acc = fl_acc
                torch.save(fl_models[0].state_dict(), './' + drug + '_fl_model.pkl')
        #f.read()
        f.write(drug + '_FL max_acc: ' + str(round(fl_max_acc, 3)) + '\n')


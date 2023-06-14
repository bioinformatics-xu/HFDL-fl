# -*- coding: utf-8 -*
import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import csv
import logging
import os
import copy
from math import *
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from torchvision.datasets import MNIST

import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from datasets import DrugResponse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AZD7762', help='dataset used for training')
    # AZD7762 PLX4720 parbendazole
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    # two-set noniid-labeldir homo
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=30, help='number of local epochs')
    parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.25, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/commonWaterfall_client", help="Data directory")
    parser.add_argument('--gamma', type=int,  default=1, help="Focal_Loss gamma")
    # drugResponseCTRP
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cpu', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    args = parser.parse_args()
    return args



class FcNet(nn.Module):
    """
    Fully connected network for MNIST classification
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.0):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_p = dropout_p

        self.dims = [self.input_dim]
        self.dims.extend(hidden_dims)
        self.dims.append(self.output_dim)

        self.layers = nn.ModuleList([])

        for i in range(len(self.dims) - 1):
            ip_dim = self.dims[i]
            op_dim = self.dims[i + 1]
            self.layers.append(
                nn.Linear(ip_dim, op_dim, bias=True)
            )

        self.__init_net_weights__()

    def __init_net_weights__(self):

        for m in self.layers:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0.1)

    def forward(self, x):

        x = x.view(-1, self.input_dim)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Do not apply ReLU on the final layer
            if i < (len(self.layers) - 1):
                x = F.relu(x)

            if i < (len(self.layers) - 1):  # No dropout on output layer
                x = F.dropout(x, p=self.dropout_p, training=self.training)

        return x

def init_nets(dropout_p=0.25):

    input_size = 17180
    output_size = 3
    hidden_sizes = [1000, 500, 200]
    net = FcNet(input_size, hidden_sizes, output_size, dropout_p)

    model_meta_data = []
    layer_type = []
    for (k, v) in net.state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return net, model_meta_data, layer_type

class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=1):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # 是tensor数据格式的列表

    def forward(self, preds, labels):
        """
        preds:logist输出值
        labels:标签
        """
        preds = F.softmax(preds, dim=1)
        eps = 1e-7
        target = self.one_hot(preds.size(1), labels)
        ce = -1 * torch.log(preds + eps) * target
        floss = torch.pow((1 - preds), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)

    def one_hot(self, num, labels):
        one = torch.zeros((labels.size(0), num))
        one[range(labels.size(0)), labels] = 1
        return one

def train_net(net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu", train_test=None):

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc = compute_accuracy(net, test_dataloader, get_confusion_matrix=False, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    # criterion = Focal_Loss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)

    # target_list = train_dataloader[0].dataset.targets.tolist()
    # class_counts = Counter(target_list)
    # if len(class_counts) == 3:
    #     c0 = class_counts[0] / sum(target_list)
    #     c1 = class_counts[1] / sum(target_list)
    #     c2 = class_counts[2] / sum(target_list)
    #     weight_loss = torch.tensor(
    #             [(1 / c0) / (1 / c0 + 1 / c1 + 1 / c2), (1 / c1) / (1 / c0 + 1 / c1 + 1 / c2),
    #              (1 / c2) / (1 / c0 + 1 / c1 + 1 / c2)])
    #
    #     criterion = Focal_Loss(weight=weight_loss,gamma=args.gamma).to(device)

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())
                # print("第%d个epoch的第%d学习率：%f" % (epoch, batch_idx, optimizer.param_groups[0]['lr']))

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        scheduler.step(epoch_loss)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        test_acc_epoch, conf_matrix, macro_F1_epoch, weighted_F1_epoch = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                            device=device)

        if os.path.exists((args.datadir + '/results/results_client_loss' + '.csv')):
            list_result = [str(args.dataset), train_test, X_train_size, X_test_size, epoch, train_acc, test_acc_epoch, macro_F1_epoch, weighted_F1_epoch,conf_matrix]
            write_csv('results_client_loss', list_result)
        else:
            create_csv('results_client_loss',
                       ['drug_name', 'train_test','train_szie', 'test_size', "epoch", 'train_acc',
                        'final_test_acc', "macro_F1", "weighted_F1","confusion_matrix"])
            list_result = [str(args.dataset), train_test, X_train_size, X_test_size, epoch, train_acc, test_acc_epoch, macro_F1_epoch, weighted_F1_epoch,conf_matrix]
            write_csv('results_client_loss', list_result)


        print("test_acc_epoch:", test_acc_epoch)
        print("macro_F1_epoch",macro_F1_epoch)
        print("weighted_F1_epoch",weighted_F1_epoch)
        print(conf_matrix)

    #
    #
    # if (out.shape[1]==2):
    #     test_acc, conf_matrix, auc_value = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
    #                                                         device=device, auc=True)
    #     return train_acc, test_acc, auc_value
    # elif (out.shape[1]==3):
    #     test_acc, conf_matrix, macro_F1, weighted_F1 = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
    #                                                         device=device)
    #     return train_acc, test_acc, macro_F1, weighted_F1
    # else:
    #     return train_acc, test_acc
    #
    # logger.info('>> Training accuracy: %f' % train_acc)
    # logger.info('>> Test accuracy: %f' % test_acc)

def compute_accuracy(model, dataloader, get_confusion_matrix=False, moon_model=False, device="cpu", auc=False):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    prob_all = []
    label_all = []
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                if moon_model:
                    _, _, out = model(x)
                else:
                    out = model(x)
                _, pred_label = torch.max(out.data, 1)

                #softmax_out = F.softmax(out)
                # out_softmax = F.softmax(out, dim=1)
                softmax_out_max = torch.max(F.softmax(out,dim=1), 1)
                prob_all.extend(softmax_out_max.values.numpy())
                label_all.extend(target.data.numpy())
                # prediction = torch.max(F.softmax(out), 1)[1]

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)


    if was_training:
        model.train()

    if get_confusion_matrix:
        if (out.shape[1]==3):
            TP0 = conf_matrix[0, 0]
            TP1 = conf_matrix[1, 1]
            TP2 = conf_matrix[2, 2]
            FN0 = conf_matrix[1, 0] + conf_matrix[2, 0]
            FN1 = conf_matrix[0, 1] + conf_matrix[2, 1]
            FN2 = conf_matrix[0, 2] + conf_matrix[1, 2]
            FP0 = conf_matrix[0, 1] + conf_matrix[0, 2]
            FP1 = conf_matrix[1, 0] + conf_matrix[1, 2]
            FP2 = conf_matrix[2, 1] + conf_matrix[2, 0]
            Recallm = (TP0 + TP1 + TP2) / (TP0 + TP1 + TP2 + FN0 + FN1 + FN2)
            Precisionm = (TP0 + TP1 + TP2) / (TP0 + TP1 + TP2 + FP0 + FP1 + FP2)
            F1_score0 = (2 * TP0) / (2 * TP0 + FP0 + FN0)
            F1_score1 = (2 * TP1) / (2 * TP1 + FP1 + FN1)
            F1_score2 = (2 * TP2) / (2 * TP2 + FP2 + FN2)
            macro_F1 = (F1_score0 + F1_score1 + F1_score2) / 3

            class_num0 = conf_matrix[0, 0] + conf_matrix[1, 0] + conf_matrix[2, 0]
            class_num1 = conf_matrix[0, 1] + conf_matrix[1, 1] + conf_matrix[2, 1]
            class_num2 = conf_matrix[0, 2] + conf_matrix[1, 2] + conf_matrix[2, 2]

            weighted_F1 = (class_num0 * F1_score0 + class_num1 * F1_score1 + class_num2 * F1_score2) / (
                        class_num0 + class_num1 + class_num2)
            # weighted_Recall = (class_num0 * Recall0 + class_num1 * Recall1 + class_num2 * Recall2) / (
            #             class_num0 + class_num1 + class_num2)
            # weighted_Precision = (class_num0 * Precision0 + class_num1 * Precision1 + class_num2 * Precision2) / (
            #         class_num0 + class_num1 + class_num2)
            return correct / float(total), conf_matrix, macro_F1, weighted_F1
        elif (out.shape[1]==2):
            if auc:
                auc_value = roc_auc_score(label_all, prob_all)
                return correct / float(total), conf_matrix, auc_value
            else:
                return correct / float(total), conf_matrix
    else:
        if (out.shape[1]==2):
            if auc:
                auc_value = roc_auc_score(label_all, prob_all)
                return correct / float(total), auc_value
            else:
                return correct / float(total)
        else:
            return correct / float(total)

class DrugResponse(MNIST):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                 download=False, datadir=None, test_datai=None):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train
        self.dataidxs = dataidxs
        self.datadir = datadir
        self.test_datai = test_datai

        if self.train:
            if self.test_datai == "data0":
                self.data = np.load(self.datadir + "/X_train_1.npy")
                self.targets = np.load(self.datadir + "/y_train_1.npy")
            elif self.test_datai == "data1":
                self.data = np.load(self.datadir + "/X_train_2.npy")
                self.targets = np.load(self.datadir + "/y_train_2.npy")
            else:
                self.data = np.load(self.datadir + "/X_train.npy")
                self.targets = np.load(self.datadir + "/y_train.npy")
        else:
            if self.test_datai == "data0":
                self.data = np.load(self.datadir + "/X_test_1.npy")
                self.targets = np.load(self.datadir + "/y_test_1.npy")
            elif self.test_datai == "data1":
                self.data = np.load(self.datadir + "/X_test_2.npy")
                self.targets = np.load(self.datadir + "/y_test_2.npy")
            else:
                self.data = np.load(self.datadir + "/X_test.npy")
                self.targets = np.load(self.datadir + "/y_test.npy")

        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.targets = self.targets[self.dataidxs]



    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        return data, target

    def __len__(self):
        return len(self.data)

def create_csv(X, csv_head):
    path = str(args.datadir + "/results/" + "%s.csv" % (X))
    with open(path, 'w', newline='', encoding='utf-8_sig') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(csv_head)  # 写入表头

def write_csv(X, data_row):
    path = str(args.datadir + "/results/" + "%s.csv" % (X))
    with open(path, mode='a', newline='', encoding='utf-8_sig') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)

if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    train_bs = 64
    test_bs = 32
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)

    client_net, client_model_meta_data, client_layer_type = init_nets(args.dropout_p)

    client_net_para = client_net.state_dict()
    if args.is_same_initial:
        client_net.load_state_dict(client_net_para)

    device = torch.device(args.device)

    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    cell_features1 = pd.read_csv(args.datadir + '/GDSC2_Expr_feature_common.csv')
    x_origin1 = cell_features1.iloc[:, 1:]
    scaler = preprocessing.StandardScaler()
    x_features1 = scaler.fit_transform(x_origin1)
    x_origin1 = pd.DataFrame(x_features1, columns=x_origin1.columns)
    x_origin1 = x_origin1[x_origin1.columns.sort_values()]
    drug_class1 = pd.read_csv(args.datadir + '/GDSC2_response_class_common_waterfall.csv')
    y_origin1 = drug_class1[args.dataset]

    X_train1, X_test1, y_train1, y_test1 = train_test_split(x_origin1, y_origin1, test_size=0.2, random_state=66)
    X_train1 = np.array(X_train1, dtype='float32')
    X_test1 = np.array(X_test1, dtype='float32')
    y_train1 = np.array(y_train1, dtype='int64')
    y_test1 = np.array(y_test1, dtype='int64')

    cell_features2 = pd.read_csv(args.datadir + '/CTRP_Expr_feature_common.csv')
    x_origin2 = cell_features2.iloc[:, 1:]
    scaler = preprocessing.StandardScaler()
    x_features2 = scaler.fit_transform(x_origin2)
    x_origin2 = pd.DataFrame(x_features2, columns=x_origin2.columns)
    x_origin2 = x_origin2[x_origin2.columns.sort_values()]
    drug_class2 = pd.read_csv(args.datadir + '/CTRP_response_class_common_waterfall.csv')
    y_origin2 = drug_class2[args.dataset]

    X_train2, X_test2, y_train2, y_test2 = train_test_split(x_origin2, y_origin2, test_size=0.2, random_state=66)
    X_train2 = np.array(X_train2, dtype='float32')
    X_test2 = np.array(X_test2, dtype='float32')
    y_train2 = np.array(y_train2, dtype='int64')
    y_test2 = np.array(y_test2, dtype='int64')

    np.save(args.datadir + "/X_train_1.npy", X_train1)
    np.save(args.datadir + "/X_test_1.npy", X_test1)
    np.save(args.datadir + "/y_train_1.npy", y_train1)
    np.save(args.datadir + "/y_test_1.npy", y_test1)

    np.save(args.datadir + "/X_train_2.npy", X_train2)
    np.save(args.datadir + "/X_test_2.npy", X_test2)
    np.save(args.datadir + "/y_train_2.npy", y_train2)
    np.save(args.datadir + "/y_test_2.npy", y_test2)

    X_train_size = X_train1.shape[0]
    X_test_size = X_test1.shape[0]

    train_ds_GDSC = DrugResponse(args.datadir, datadir=args.datadir, train=True, transform=None, download=True, test_datai="data0")
    test_ds_GDSC = DrugResponse(args.datadir, datadir=args.datadir, train=False, transform=None, download=True, test_datai="data0")

    train_dl_GDSC = data.DataLoader(dataset=train_ds_GDSC, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl_GDSC = data.DataLoader(dataset=test_ds_GDSC, batch_size=test_bs, shuffle=False, drop_last=False)

    train_ds_CTRP = DrugResponse(args.datadir, datadir=args.datadir, train=True, transform=None, download=True, test_datai="data1")
    test_ds_CTRP = DrugResponse(args.datadir, datadir=args.datadir, train=False, transform=None, download=True, test_datai="data1")

    train_dl_CTRP = data.DataLoader(dataset=train_ds_CTRP, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl_CTRP = data.DataLoader(dataset=test_ds_CTRP, batch_size=test_bs, shuffle=False, drop_last=False)

    trainacc, testacc, macro_F1, weighted_F1 = train_net(client_net, train_dl_CTRP, test_dl_CTRP, args.epochs, args.lr,
                                                         args.optimizer, device=device, train_test="CTRP_CTRP")

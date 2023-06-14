import os
import logging
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

from datasets import DrugResponse
import copy

import torch.nn as nn
import torch.nn.functional as F

from scipy import interp

import random
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):

    if datadir == "./data/commonWaterfall":
        print("predict drug response common waterfall")

        cell_features1 = pd.read_csv(datadir + '/GDSC2_Expr_feature_common_revision1.csv')
        x_origin1 = cell_features1.iloc[:, 1:]

        x_origin1 = pd.DataFrame(x_origin1, columns=x_origin1.columns)
        x_origin1 = x_origin1[x_origin1.columns.sort_values()]
        drug_class1 = pd.read_csv(datadir + '/GDSC2_response_class_common_waterfall_revision1.csv')
        y_origin1 = drug_class1[dataset]
        X_train1, X_test1, y_train1, y_test1 = train_test_split(x_origin1, y_origin1, test_size=0.2, random_state=666)

        # CTRP data
        cell_features2 = pd.read_csv(datadir + '/CTRP_Expr_feature_common_revision1.csv')
        x_origin2 = cell_features2.iloc[:, 1:]
        x_origin2 = pd.DataFrame(x_origin2, columns=x_origin2.columns)
        x_origin2 = x_origin2[x_origin2.columns.sort_values()]

        drug_class2 = pd.read_csv(datadir + '/CTRP_response_class_common_waterfall_revision1.csv')
        y_origin2 = drug_class2[dataset]
        X_train2, X_test2, y_train2, y_test2 = train_test_split(x_origin2, y_origin2, test_size=0.2, random_state=666)

        X_train1 = np.array(X_train1, dtype='float32')
        scaler = preprocessing.StandardScaler()
        X_train1 = scaler.fit_transform(X_train1)

        X_test1 = np.array(X_test1, dtype='float32')
        scaler = preprocessing.StandardScaler()
        X_test1 = scaler.fit_transform(X_test1)

        y_train1 = np.array(y_train1, dtype='int64')
        y_test1 = np.array(y_test1, dtype='int64')

        X_train2 = np.array(X_train2, dtype='float32')
        scaler = preprocessing.StandardScaler()
        X_train2 = scaler.fit_transform(X_train2)

        X_test2 = np.array(X_test2, dtype='float32')
        scaler = preprocessing.StandardScaler()
        X_test2 = scaler.fit_transform(X_test2)

        y_train2 = np.array(y_train2, dtype='int64')
        y_test2 = np.array(y_test2, dtype='int64')

        np.save(datadir + "/X_train_1.npy", X_train1)
        np.save(datadir + "/X_test_1.npy", X_test1)
        np.save(datadir + "/y_train_1.npy", y_train1)
        np.save(datadir + "/y_test_1.npy", y_test1)

        np.save(datadir + "/X_train_2.npy", X_train2)
        np.save(datadir + "/X_test_2.npy", X_test2)
        np.save(datadir + "/y_train_2.npy", y_train2)
        np.save(datadir + "/y_test_2.npy", y_test2)

        X_train = np.concatenate((X_train1, X_train2), axis=0)
        X_test = np.concatenate((X_test1, X_test2), axis=0)
        y_train = np.concatenate((y_train1, y_train2), axis=0)
        y_test = np.concatenate((y_test1, y_test2), axis=0)

        np.save(datadir + "/X_train.npy", X_train)
        np.save(datadir + "/X_test.npy", X_test)
        np.save(datadir + "/y_train.npy", y_train)
        np.save(datadir + "/y_test.npy", y_test)
    else:
        print("predict drug response within study -- HFDL-Within")
        cell_features = pd.read_csv(datadir + '/Expr_feature.csv')
        x_origin = cell_features.iloc[:, 1:]
        x_origin = pd.DataFrame(x_origin, columns=x_origin.columns)
        drug_class = pd.read_csv(datadir + '/response_class.csv')
        y_origin = drug_class[dataset]
        X_train, X_test, y_train, y_test = train_test_split(x_origin, y_origin, test_size=0.2, random_state=666)

        X_train = np.array(X_train, dtype='float32')
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)

        X_test = np.array(X_test, dtype='float32')
        scaler = preprocessing.StandardScaler()
        X_test = scaler.fit_transform(X_test)

        y_train = np.array(y_train, dtype='int64')
        y_test = np.array(y_test, dtype='int64')

        if os.path.exists(datadir):
            pass
        else:
            mkdirs(datadir)

        np.save(datadir + "/X_train.npy", X_train)
        np.save(datadir + "/X_test.npy", X_test)
        np.save(datadir + "/y_train.npy", y_train)
        np.save(datadir + "/y_test.npy", y_test)

        n_train = y_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "two-set":
        net_1_num = np.array(range(0, y_train1.shape[0]))
        net_2_num = np.array(range(y_train1.shape[0], y_train1.shape[0] + y_train2.shape[0]))
        net_dataidx_map = {0: net_1_num, 1: net_2_num}

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)

def get_trainable_parameters(net):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    # logger.info("net.parameter.data:", list(net.parameters()))
    paramlist=list(trainable)
    N=0
    for params in paramlist:
        N+=params.numel()
        # logger.info("params.data:", params.data)
    X=torch.empty(N,dtype=torch.float64)
    X.fill_(0.0)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            X[offset:offset+numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset+=numel
    # logger.info("get trainable x:", X)
    return X

def put_trainable_parameters(net,X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset+numel].data.view_as(params.data))
        offset+=numel

def one_hot(self, num, labels):
    one = torch.zeros((labels.size(0), num))
    one[range(labels.size(0)), labels] = 1
    return one

def compute_accuracy(model, dataloader, get_confusion_matrix=False, moon_model=False, device="cpu", AUC=False):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])
    softmax_out_max_list = np.zeros((0, 3), dtype=np.float32)

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

                softmax_out_max = torch.max(F.softmax(out,dim=1), 1)
                prob_all.extend(softmax_out_max.values.numpy())
                label_all.extend(target.data.numpy())

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                    softmax_out_max_list = np.vstack([softmax_out_max_list, out.detach().numpy().astype(np.float32)])
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
                    softmax_out_max_list = np.vstack([softmax_out_max_list, out.cpu().detach().numpy().astype(np.float32)])

    # 计算每一类的ROC
    fpr_list = dict()
    tpr_list = dict()
    roc_auc_list = dict()

    y_labels_list = preprocessing.label_binarize(true_labels_list, classes=[0, 1, 2])

    for i in range(out.shape[1]):
        fpr_list[i], tpr_list[i], _ = roc_curve(y_labels_list[:, i], softmax_out_max_list[:, i])
        roc_auc_list[i] = auc(fpr_list[i], tpr_list[i])

    # Compute micro-average ROC curve and ROC area
    fpr_list["micro"], tpr_list["micro"], _ = roc_curve(y_labels_list.ravel(), softmax_out_max_list.ravel())
    roc_auc_list["micro"] = auc(fpr_list["micro"], tpr_list["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_list[i] for i in range(out.shape[1])]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(out.shape[1]):
        mean_tpr += interp(all_fpr, fpr_list[i], tpr_list[i])
    # Finally average it and compute AUC
    mean_tpr /= out.shape[1]
    fpr_list["macro"] = all_fpr
    tpr_list["macro"] = mean_tpr
    roc_auc_list["macro"] = auc(fpr_list["macro"], tpr_list["macro"])

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

            F1_score0 = (2 * TP0) / (2 * TP0 + FP0 + FN0)
            F1_score1 = (2 * TP1) / (2 * TP1 + FP1 + FN1)
            F1_score2 = (2 * TP2) / (2 * TP2 + FP2 + FN2)
            macro_F1 = (F1_score0 + F1_score1 + F1_score2) / 3

            class_num0 = conf_matrix[0, 0] + conf_matrix[1, 0] + conf_matrix[2, 0]
            class_num1 = conf_matrix[0, 1] + conf_matrix[1, 1] + conf_matrix[2, 1]
            class_num2 = conf_matrix[0, 2] + conf_matrix[1, 2] + conf_matrix[2, 2]

            weighted_F1 = (class_num0 * F1_score0 + class_num1 * F1_score1 + class_num2 * F1_score2) / (
                        class_num0 + class_num1 + class_num2)

            return correct / float(total), conf_matrix, macro_F1, weighted_F1, roc_auc_list
        elif (out.shape[1]==2):
            if AUC:
                auc_value = roc_auc_score(label_all, prob_all)
                return correct / float(total), conf_matrix, auc_value
            else:
                return correct / float(total), conf_matrix
    else:
        if (out.shape[1]==2):
            if AUC:
                auc_value = roc_auc_score(label_all, prob_all)
                return correct / float(total), auc_value
            else:
                return correct / float(total)
        else:
            return correct / float(total)

def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, net_id=None, total=0):
    if dataset != '':
        dl_obj = DrugResponse
        transform_train = None
        transform_test = None

        train_ds = dl_obj(datadir, datadir=datadir, dataidxs=dataidxs, train=True, transform=transform_train,
                          download=True)
        test_ds = dl_obj(datadir, datadir=datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds

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


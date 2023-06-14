# -*- coding: utf-8 -*
import json
import torch.optim as optim
import argparse
import csv
from math import *
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau

import datetime
import copy

from utils import *
# from resnetcifar import *
from datasets import DrugResponse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='PLX4720', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='two-set', help='the data partitioning strategy')
    parser.add_argument('--input_size', type=int, default=17180, help='the input feature size')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='fl algorithms: fedavg/scaffold/fednova')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=60, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.3, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/commonWaterfall", help="Data directory")
    parser.add_argument('--gamma', type=int,  default=3, help="Focal_Loss gamma")
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
    parser.add_argument('--out_name_g', type=str, default="results_global_raw", help="Sub file name of global results")
    parser.add_argument('--out_name_c', type=str, default="results_client_raw", help="Sub file name of client results")
    parser.add_argument('--out_name_l', type=str, default='results_local_raw', help="Sub file name of local results")
    parser.add_argument('--out_traindata_cls_count', type=str, default='traindata_cls_count_raw', help="file name of traindata cls count")
    args = parser.parse_args()
    return args

def init_nets(dropout_p, n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        if args.model == "mlp":
            if args.datadir == './data/drugResponseCTRP':
                input_size = 17180
                output_size = 3
                hidden_sizes = [1000, 500, 200]
            elif args.datadir == './data/drugResponseGDSC':
                input_size = 17180
                output_size = 3
                hidden_sizes = [1000, 500, 200]
            elif args.datadir == "./data/commonWaterfall":
                input_size = args.input_size
                output_size = 3
                hidden_sizes = [1000, 500, 200]

            net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type

class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # 是tensor数据格式的列表

    def forward(self, preds, labels):
        """
        preds:logist输出值
        labels:标签
        """
        preds = F.softmax(preds, dim=1)
        #print(preds)
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

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

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

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)

    target_list = train_dataloader[0].dataset.targets.tolist()
    class_counts = Counter(target_list)
    if len(class_counts) == 3:
        c0 = class_counts[0] / sum(target_list)
        c1 = class_counts[1] / sum(target_list)
        c2 = class_counts[2] / sum(target_list)
        weight_loss = torch.tensor(
                [(1 / c0) / (1 / c0 + 1 / c1 + 1 / c2), (1 / c1) / (1 / c0 + 1 / c1 + 1 / c2),
                 (1 / c2) / (1 / c0 + 1 / c1 + 1 / c2)])
        criterion = Focal_Loss(weight=weight_loss,gamma=args.gamma).to(device)

    elif len(class_counts) == 2:
        c0 = class_counts[0] / sum(target_list)
        c1 = class_counts[1] / sum(target_list)
        weight_loss = torch.tensor(
            [(1 / c0) / (1 / c0 + 1 / c1), (1 / c1) / (1 / c0 + 1 / c1)]
        )
        criterion = Focal_Loss(weight=weight_loss, gamma=args.gamma).to(device)

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

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        scheduler.step(epoch_loss)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    if (out.shape[1]==2):
        test_acc, conf_matrix, auc_value = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                            device=device, AUC=True)
        return train_acc, test_acc, auc_value
    elif (out.shape[1]==3):
        test_acc, conf_matrix, macro_F1, weighted_F1, AUC_ROC = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                            device=device)
        logger.info(args.dataset + '>> confusion matrix')
        logger.info(conf_matrix)

        return train_acc, test_acc, macro_F1, weighted_F1, AUC_ROC
    else:
        return train_acc, test_acc

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    logger.info(' ** Training complete **')

def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

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

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)

    target_list = train_dataloader[0].dataset.targets.tolist()
    class_counts = Counter(target_list)
    if len(class_counts) == 3:
        c0 = class_counts[0] / sum(target_list)
        c1 = class_counts[1] / sum(target_list)
        c2 = class_counts[2] / sum(target_list)
        weight_loss = torch.tensor(
                [(1 / c0) / (1 / c0 + 1 / c1 + 1 / c2), (1 / c1) / (1 / c0 + 1 / c1 + 1 / c2),
                 (1 / c2) / (1 / c0 + 1 / c1 + 1 / c2)])
        criterion = Focal_Loss(weight=weight_loss,gamma=args.gamma).to(device)

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

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        scheduler.step(epoch_loss)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)

    train_acc = compute_accuracy(net, train_dataloader, device=device)

    if (out.shape[1]==2):
        test_acc, conf_matrix, auc_value = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                            device=device, AUC=True)
        return train_acc, test_acc, c_delta_para, auc_value
    elif (out.shape[1]==3):
        test_acc, conf_matrix, macro_F1, weighted_F1, AUC_ROC = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                            device=device)
        return train_acc, test_acc, c_delta_para, macro_F1, weighted_F1, AUC_ROC
    else:
        test_acc = compute_accuracy(net, test_dataloader, get_confusion_matrix=False,
                                                            device=device, AUC=False)
        return train_acc, test_acc, c_delta_para

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    logger.info(' ** Training complete **')

def train_net_fednova(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc = compute_accuracy(net, test_dataloader, get_confusion_matrix=False, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    tau = 0
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)

    target_list = train_dataloader[0].dataset.targets.tolist()
    class_counts = Counter(target_list)
    if len(class_counts) == 3:
        c0 = class_counts[0] / sum(target_list)
        c1 = class_counts[1] / sum(target_list)
        c2 = class_counts[2] / sum(target_list)
        weight_loss = torch.tensor(
                [(1 / c0) / (1 / c0 + 1 / c1 + 1 / c2), (1 / c1) / (1 / c0 + 1 / c1 + 1 / c2),
                 (1 / c2) / (1 / c0 + 1 / c1 + 1 / c2)])
        criterion = Focal_Loss(weight=weight_loss,gamma=args.gamma).to(device)

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
                tau = tau + 1
                epoch_loss_collector.append(loss.item())
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        scheduler.step(epoch_loss)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        norm_grad[key] = torch.div(global_model_para[key]-net_para[key], a_i)
    train_acc = compute_accuracy(net, train_dataloader, device=device)

    if (out.shape[1]==2):
        test_acc, conf_matrix, auc_value = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                            device=device, AUC=True)
        return train_acc, test_acc, a_i, norm_grad, auc_value
    elif (out.shape[1]==3):
        test_acc, conf_matrix, macro_F1, weighted_F1, AUC_ROC = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                            device=device)
        return train_acc, test_acc, a_i, norm_grad, macro_F1, weighted_F1, AUC_ROC
    else:
        test_acc = compute_accuracy(net, test_dataloader, get_confusion_matrix=False,
                                                            device=device, AUC=False)
        return train_acc, test_acc, a_i, norm_grad

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    logger.info(' ** Training complete **')

def create_dir_and_file(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.isfile(file_path):
        with open(file_path, 'w'):
            pass  # do nothing, just create the file

def create_csv(X, csv_head):
    path = str(args.datadir + "/results/" + "%s.csv" % (X))
    create_dir_and_file(path)
    with open(path, 'w', newline='', encoding='utf-8_sig') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(csv_head)

def write_csv(X, data_row):
    path = str(args.datadir + "/results/" + "%s.csv" % (X))
    with open(path, mode='a', newline='', encoding='utf-8_sig') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)

def results_statistics_client(selected_0, selected_1):

    test_data0_ds = DrugResponse(args.datadir, datadir=args.datadir, train=False, transform=None,
                                 test_datai=("data" + str(selected_0)))
    test_data0_dl = data.DataLoader(dataset=test_data0_ds, batch_size=32, shuffle=False,
                                    drop_last=False)
    test_data1_ds = DrugResponse(args.datadir, datadir=args.datadir, train=False, transform=None,
                                 test_datai="data" + str(selected_1))
    test_data1_dl = data.DataLoader(dataset=test_data1_ds, batch_size=32, shuffle=False,
                                    drop_last=False)
    test_acc_client0_d0, conf_matrix_client0_d0, macro_F1_client0_d0, weighted_F1_client0_d0, AUC_ROC_client0_d0 = compute_accuracy(
        nets[selected_0], test_data0_dl,
        get_confusion_matrix=True,
        device=device,
        AUC=False)
    test_acc_client0_d1, conf_matrix_client0_d1, macro_F1_client0_d1, weighted_F1_client0_d1, AUC_ROC_client0_d1 = compute_accuracy(
        nets[selected_0], test_data1_dl,
        get_confusion_matrix=True,
        device=device,
        AUC=False)
    test_acc_client1_d0, conf_matrix_client1_d0, macro_F1_client1_d0, weighted_F1_client1_d0, AUC_ROC_client1_d0 = compute_accuracy(
        nets[selected_1], test_data0_dl,
        get_confusion_matrix=True,
        device=device,
        AUC=False)
    test_acc_client1_d1, conf_matrix_client1_d1, macro_F1_client1_d1, weighted_F1_client1_d1, AUC_ROC_client1_d1 = compute_accuracy(
        nets[selected_1], test_data1_dl,
        get_confusion_matrix=True,
        device=device,
        AUC=False)
    test_acc_global_d0, conf_matrix_global_d0, macro_F1_global_d0, weighted_F1_global_d0, AUC_ROC_global_d0 = compute_accuracy(
        global_model, test_data0_dl,
        get_confusion_matrix=True,
        device=device,
        AUC=False)
    test_acc_global_d1, conf_matrix_global_d1, macro_F1_global_d1, weighted_F1_global_d1, AUC_ROC_global_d1 = compute_accuracy(
        global_model, test_data1_dl,
        get_confusion_matrix=True,
        device=device,
        AUC=False)

    if os.path.exists(args.datadir + '/results/' + args.out_name_c + '.csv'):
        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round,
                       'client' + str(selected_0) + '_d' + str(selected_0),
                       test_acc_client0_d0, macro_F1_client0_d0, weighted_F1_client0_d0, conf_matrix_client0_d0, AUC_ROC_client0_d0]
        write_csv(args.out_name_c, list_result)
        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round,
                       'client' + str(selected_0) + '_d' + str(selected_1),
                       test_acc_client0_d1, macro_F1_client0_d1, weighted_F1_client0_d1, conf_matrix_client0_d1, AUC_ROC_client0_d1]

        write_csv(args.out_name_c, list_result)
        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round,
                       'client' + str(selected_1) + '_d' + str(selected_0),
                       test_acc_client1_d0, macro_F1_client1_d0, weighted_F1_client1_d0, conf_matrix_client1_d0, AUC_ROC_client1_d0]

        write_csv(args.out_name_c, list_result)
        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round,
                       'client' + str(selected_1) + '_d' + str(selected_1),
                       test_acc_client1_d1, macro_F1_client1_d1, weighted_F1_client1_d1, conf_matrix_client1_d1, AUC_ROC_client1_d1]
        write_csv(args.out_name_c, list_result)
        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round,
                       'global_d' + str(selected_0),
                       test_acc_global_d0, macro_F1_global_d0, weighted_F1_global_d0, conf_matrix_global_d0, AUC_ROC_global_d0]
        write_csv(args.out_name_c, list_result)
        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round,
                       'global_d' + str(selected_1),
                       test_acc_global_d1, macro_F1_global_d1, weighted_F1_global_d1, conf_matrix_global_d1, AUC_ROC_global_d1]
        write_csv(args.out_name_c, list_result)

    else:
        create_csv(args.out_name_c,
                   ['drug_name', 'algorithm', 'gamma', 'partition', 'n_parties', 'round', 'type',
                    'test_acc', "macro_F1", "weighted_F1", "confusion_matrix", "AUC_ROC"])
        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round,
                       'client' + str(selected_0) + '_d' + str(selected_0),
                       test_acc_client0_d0, macro_F1_client0_d0, weighted_F1_client0_d0, conf_matrix_client0_d0, AUC_ROC_client0_d0]
        write_csv(args.out_name_c, list_result)
        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round,
                       'client' + str(selected_0) + '_d' + str(selected_1),
                       test_acc_client0_d1, macro_F1_client0_d1, weighted_F1_client0_d1, conf_matrix_client0_d1, AUC_ROC_client0_d1]

        write_csv(args.out_name_c, list_result)
        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round,
                       'client' + str(selected_1) + '_d' + str(selected_0),
                       test_acc_client1_d0, macro_F1_client1_d0, weighted_F1_client1_d0, conf_matrix_client1_d0, AUC_ROC_client1_d0]

        write_csv(args.out_name_c, list_result)
        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round,
                       'client' + str(selected_1) + '_d' + str(selected_1),
                       test_acc_client1_d1, macro_F1_client1_d1, weighted_F1_client1_d1, conf_matrix_client1_d1, AUC_ROC_client1_d1]
        write_csv(args.out_name_c, list_result)
        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round,
                       'global_d' + str(selected_0),
                       test_acc_global_d0, macro_F1_global_d0, weighted_F1_global_d0, conf_matrix_global_d0, AUC_ROC_global_d0]
        write_csv(args.out_name_c, list_result)
        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round,
                       'global_d' + str(selected_1),
                       test_acc_global_d1, macro_F1_global_d1, weighted_F1_global_d1, conf_matrix_global_d1, AUC_ROC_global_d1]
        write_csv(args.out_name_c, list_result)

def local_train_net(nets, selected, args, net_dataidx_map, test_dl = None, device="cpu", round = None):
    avg_acc = 0.0
    avg_macro_F1 = 0.0
    avg_weighted_F1 = 0.0
    avg_auc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)

        n_epoch = args.epochs

        if net.output_dim==2:
            trainacc, testacc, testauc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr,
                                                   args.optimizer, device=device)
            avg_auc += testauc
            if os.path.exists(args.datadir + '/results/' + args.out_name_l + '.csv'):
                list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, net_id, len(test_dl),
                               testacc, testauc]
                write_csv(args.out_name_l, list_result)
            else:
                create_csv(args.out_name_l,
                           ['drug_name', 'algorithm', 'gamma','partition', 'n_parties', 'round', 'net_id', 'test_num',
                            'final_test_acc', 'final_test_auc'])
                list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, net_id, len(test_dl),
                               testacc, testauc]
                write_csv(args.out_name_l, list_result)

        elif net.output_dim==3:
            trainacc, testacc, macro_F1, weighted_F1, AUC_ROC = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr,
                                                   args.optimizer, device=device)

            if os.path.exists(args.datadir + '/results/' + args.out_name_l + '.csv'):
                list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, net_id, len(test_dl),
                               testacc, macro_F1, weighted_F1, AUC_ROC]
                write_csv(args.out_name_l, list_result)

            else:
                create_csv(args.out_name_l,
                           ['drug_name', 'algorithm', 'gamma', 'partition', 'n_parties', 'round', 'net_id', 'test_num',
                            'final_test_acc', "macro_F1", "weighted_F1", 'AUC_ROC'])
                list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, net_id, len(test_dl),
                               testacc, macro_F1, weighted_F1, AUC_ROC]
                write_csv(args.out_name_l, list_result)
            avg_macro_F1 += macro_F1
            avg_weighted_F1 += weighted_F1
        else:
            trainacc, testacc= train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr,
                                                   args.optimizer, device=device)

        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)

    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    if net.output_dim==2:
        avg_auc /= len(selected)
        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, "avg test acc", len(test_dl),
                       avg_acc]
        write_csv(args.out_name_l, list_result)
    elif net.output_dim==3:
        avg_weighted_F1 /= len(selected)
        avg_macro_F1 /= len(selected)
        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, "avg test acc", len(test_dl),
                       avg_acc, avg_macro_F1, avg_weighted_F1]
        write_csv(args.out_name_l, list_result)

    nets_list = list(nets.values())
    return nets_list

def local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = None, device="cpu", round=None):
    avg_acc = 0.0
    avg_macro_F1 = 0.0
    avg_weighted_F1 = 0.0
    avg_auc = 0.0

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)
        c_nets[net_id].to(device)
        noise_level = args.noise

        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        if net.output_dim == 2:
            trainacc, testacc, c_delta_para, testauc = train_net_scaffold(net_id, net, global_model, c_nets[net_id],
                                                                          c_global, train_dl_local, test_dl, n_epoch,
                                                                          args.lr, args.optimizer, device=device)

            if os.path.exists(args.datadir + '/results/'+ args.out_name_l + '.csv'):
                list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, net_id, len(test_dl),
                               testacc, testauc]
                write_csv(args.out_name_l, list_result)
            else:
                create_csv(args.out_name_l,
                           ['drug_name', 'algorithm', 'gamma', 'partition', 'n_parties', 'round', 'net_id', 'test_num',
                            'final_test_acc', 'final_test_auc'])
                list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, net_id, len(test_dl),
                               testacc, testauc]
                write_csv(args.out_name_l, list_result)
            avg_auc += testauc
        elif net.output_dim == 3:
            trainacc, testacc, c_delta_para, macro_F1, weighted_F1, AUC_ROC = train_net_scaffold(net_id, net, global_model, c_nets[net_id],
                                                                          c_global, train_dl_local, test_dl, n_epoch,
                                                                          args.lr, args.optimizer, device=device)

            if os.path.exists(args.datadir + '/results/' + args.out_name_l + '.csv'):
                list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, net_id, len(test_dl),
                               testacc, macro_F1, weighted_F1, AUC_ROC]
                write_csv(args.out_name_l, list_result)
            else:
                create_csv(args.out_name_l,
                           ['drug_name', 'algorithm', 'gamma','partition', 'n_parties', 'round', 'net_id', 'test_num',
                            'final_test_acc', "macro_F1", "weighted_F1", "AUC_ROC"])
                list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, net_id, len(test_dl),
                               testacc, macro_F1, weighted_F1, AUC_ROC]
                write_csv(args.out_name_l, list_result)

            avg_macro_F1 += macro_F1
            avg_weighted_F1 += weighted_F1

        else:
            trainacc, testacc, c_delta_para = train_net_scaffold(net_id, net, global_model,
                                                                                        c_nets[net_id],
                                                                                        c_global, train_dl_local,
                                                                                        test_dl, n_epoch,
                                                                                        args.lr, args.optimizer,
                                                                                        device=device)

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]

        avg_acc += testacc
        logger.info("net %d final test acc %f" % (net_id, testacc))

    for key in total_delta:
        total_delta[key] /= args.n_parties
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    if net.output_dim == 2:
        avg_auc /= len(selected)

        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, "avg test acc", len(test_dl),
                       avg_acc, avg_auc]
        write_csv(args.out_name_l, list_result)
    elif net.output_dim == 3:
        avg_weighted_F1 /= len(selected)
        avg_macro_F1 /= len(selected)

        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, "avg test acc", len(test_dl),
                       avg_acc, avg_macro_F1, avg_weighted_F1]
        write_csv(args.out_name_l, list_result)

    nets_list = list(nets.values())
    return nets_list

def local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl = None, device="cpu", round=None):
    avg_acc = 0.0
    avg_macro_F1 = 0.0
    avg_weighted_F1 = 0.0
    avg_auc = 0.0

    a_list = []
    d_list = []
    n_list = []
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        if net.output_dim == 2:
            trainacc, testacc, a_i, d_i, testauc = train_net_fednova(net_id, net, global_model, train_dl_local, test_dl,
                                                                     n_epoch, args.lr, args.optimizer, device=device)
            avg_auc +=testauc
            if os.path.exists(args.datadir + '/results/' + args.out_name_l + '.csv'):
                list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, net_id, len(test_dl),
                               testacc, testauc]
                write_csv(args.out_name_l, list_result)
            else:
                create_csv(args.out_name_l,
                           ['drug_name', 'algorithm', 'gamma','partition', 'n_parties', 'round', 'net_id', 'test_num',
                            'final_test_acc', 'final_test_auc'])
                list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, net_id, len(test_dl),
                               testacc, testauc]
                write_csv(args.out_name_l, list_result)

        elif net.output_dim == 3:
            trainacc, testacc, a_i, d_i, macro_F1, weighted_F1, AUC_ROC  = train_net_fednova(net_id, net, global_model, train_dl_local, test_dl,
                                                                     n_epoch, args.lr, args.optimizer, device=device)

            if os.path.exists(args.datadir + '/results/' + args.out_name_l + '.csv'):
                list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, net_id, len(test_dl),
                               testacc, macro_F1, weighted_F1, AUC_ROC]
                write_csv(args.out_name_l, list_result)
            else:
                create_csv(args.out_name_l,
                           ['drug_name', 'algorithm', 'gamma','partition', 'n_parties', 'round', 'net_id', 'test_num',
                            'final_test_acc', "macro_F1", "weighted_F1", "AUC_ROC"])
                list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, net_id, len(test_dl),
                               testacc, macro_F1, weighted_F1, AUC_ROC]
                write_csv(args.out_name_l, list_result)
            avg_macro_F1 += macro_F1
            avg_weighted_F1 += weighted_F1
        else:
            trainacc, testacc, a_i, d_i  = train_net_fednova(net_id, net, global_model, train_dl_local, test_dl,
                                                                     n_epoch, args.lr, args.optimizer, device=device)

        avg_acc += testacc
        a_list.append(a_i)
        d_list.append(d_i)
        n_i = len(train_dl_local)
        n_list.append(n_i)

        logger.info("net %d final test acc %f" % (net_id, testacc))

    avg_acc /= len(selected)

    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    if net.output_dim == 2:
        avg_auc /= len(selected)
        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, "avg test acc", len(test_dl),
                       avg_acc]
        write_csv(args.out_name_l, list_result)
    elif net.output_dim == 3:
        avg_weighted_F1 /= len(selected)
        avg_macro_F1 /= len(selected)
        list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, "avg test acc", len(test_dl),
                       avg_acc, avg_macro_F1, avg_weighted_F1]
        write_csv(args.out_name_l, list_result)

    nets_list = list(nets.values())
    return nets_list, a_list, d_list, n_list

def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map

if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info(args.datadir + " drug " + args.dataset)
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_classes = len(np.unique(y_train))

    args.input_size = X_train.shape[1]

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                        args.datadir,
                                                                                        args.batch_size,
                                                                                        32)

    print("len train_dl_global:", len(train_ds_global))

    train_all_in_list = []
    test_all_in_list = []
    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, party_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)

    if args.alg == 'fedavg':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net(nets, selected, args, net_dataidx_map, test_dl = test_dl_global, device=device, round=round)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            if n_classes==2:
                test_acc, conf_matrix, test_auc = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device, AUC=True)
                logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)
                logger.info('>> Global Model Test auc: %f' % test_auc)

                if os.path.exists(args.datadir + '/results/' + args.out_name_g + '.csv'):
                    list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, train_acc,
                                   test_acc, test_auc, conf_matrix]
                    write_csv(args.out_name_g, list_result)
                else:
                    create_csv(args.out_name_g,
                               ['drug_name', 'algorithm', 'gamma','partition', 'n_parties', 'round', 'Global_train_acc',
                                'Global_test_acc', 'Global_test_auc','confusion_matrix'])
                    list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, train_acc,
                                   test_acc, test_auc, conf_matrix]
                    write_csv(args.out_name_g, list_result)

            elif n_classes==3:
                test_acc, conf_matrix, macro_F1, weighted_F1, AUC_ROC  = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device, AUC=False)
                logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)

                if os.path.exists(args.datadir + '/results/' + args.out_name_g + '.csv'):
                    list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, train_acc,
                                   test_acc, macro_F1, weighted_F1, conf_matrix, AUC_ROC]
                    write_csv(args.out_name_g, list_result)
                else:
                    create_csv(args.out_name_g,
                               ['drug_name', 'algorithm', 'gamma', 'partition', 'n_parties', 'round', 'Global_train_acc',
                                'Global_test_acc', "macro_F1", "weighted_F1","confusion_matrix", "AUC_ROC"])
                    list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, train_acc,
                                   test_acc, macro_F1, weighted_F1, conf_matrix, AUC_ROC]
                    write_csv(args.out_name_g, list_result)
                if args.datadir == "./data/commonWaterfall":
                    results_statistics_client(selected[0], selected[1])

    elif args.alg == 'scaffold':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(0, 1, args)
        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.dropout_p, args.n_parties, args)
        c_globals, _, _ = init_nets(0, 1, args)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = test_dl_global, device=device, round=round)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)

            if n_classes==2:
                test_acc, conf_matrix, test_auc = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device, AUC=True)
                logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)
                logger.info('>> Global Model Test auc: %f' % test_auc)

                if os.path.exists(args.datadir + '/results/' + args.out_name_g + '.csv'):
                    list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, train_acc,
                                   test_acc, test_auc,conf_matrix]
                    write_csv(args.out_name_g, list_result)
                else:
                    create_csv(args.out_name_g,
                               ['drug_name', 'algorithm', 'gamma', 'partition', 'n_parties', 'round', 'Global_train_acc',
                                'Global_test_acc', 'Global_test_auc',"confusion_matrix"])
                    list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, train_acc,
                                   test_acc, test_auc,conf_matrix]
                    write_csv(args.out_name_g, list_result)

            elif n_classes==3:
                test_acc, conf_matrix, macro_F1, weighted_F1, AUC_ROC = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device, AUC=False)
                logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)

                if os.path.exists(args.datadir + '/results/' + args.out_name_g + '.csv'):
                    list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, train_acc,
                                   test_acc, macro_F1, weighted_F1, conf_matrix, AUC_ROC]
                    write_csv(args.out_name_g, list_result)
                else:
                    create_csv(args.out_name_g,
                               ['drug_name', 'algorithm', 'gamma', 'partition', 'n_parties', 'round', 'Global_train_acc',
                                'Global_test_acc', "macro_F1", "weighted_F1","confusion_matrix", "AUC_ROC"])
                    list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, train_acc,
                                   test_acc, macro_F1, weighted_F1, conf_matrix, AUC_ROC]
                    write_csv(args.out_name_g, list_result)
                if args.datadir == "./data/commonWaterfall":
                    results_statistics_client(selected[0], selected[1])

    elif args.alg == 'fednova':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(0, 1, args)
        global_model = global_models[0]

        d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.n_parties)]
        d_total_round = copy.deepcopy(global_model.state_dict())
        for i in range(args.n_parties):
            for key in d_list[i]:
                d_list[i][key] = 0
        for key in d_total_round:
            d_total_round[key] = 0

        data_sum = 0
        for i in range(args.n_parties):
            data_sum += len(traindata_cls_counts[i])
        portion = []
        for i in range(args.n_parties):
            portion.append(len(traindata_cls_counts[i]) / data_sum)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            _, a_list, d_list, n_list = local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl = test_dl_global, device=device, round=round)
            total_n = sum(n_list)
            #print("total_n:", total_n)
            d_total_round = copy.deepcopy(global_model.state_dict())
            for key in d_total_round:
                d_total_round[key] = 0.0

            for i in range(len(selected)):
                d_para = d_list[i]
                for key in d_para:
                    d_total_round[key] += d_para[key] * n_list[i] / total_n

            # update global model
            coeff = 0.0
            for i in range(len(selected)):
                coeff = coeff + a_list[i] * n_list[i]/total_n

            updated_model = global_model.state_dict()
            for key in updated_model:
                #print(updated_model[key])
                if updated_model[key].type() == 'torch.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    updated_model[key] -= coeff * d_total_round[key]
            global_model.load_state_dict(updated_model)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)

            if n_classes==2:
                test_acc, conf_matrix, test_auc = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device, AUC=True)
                logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)
                logger.info('>> Global Model Test auc: %f' % test_auc)

                if os.path.exists(args.datadir + '/results/' + args.out_name_g + '.csv'):
                    list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, train_acc,
                                   test_acc, test_auc,conf_matrix]
                    write_csv(args.out_name_g, list_result)
                else:
                    create_csv(args.out_name_g,
                               ['drug_name', 'algorithm', 'gamma','partition', 'n_parties', 'round', 'Global_train_acc',
                                'Global_test_acc', 'Global_test_auc','confusion_matrix'])
                    list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, train_acc,
                                   test_acc, test_auc,conf_matrix]
                    write_csv(args.out_name_g, list_result)

            elif n_classes==3:
                test_acc, conf_matrix, macro_F1, weighted_F1, AUC_ROC = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device, AUC=False)
                logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)

                if os.path.exists(args.datadir + '/results/' + args.out_name_g + '.csv'):
                    list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, train_acc,
                                   test_acc, macro_F1, weighted_F1, conf_matrix, AUC_ROC]
                    write_csv(args.out_name_g, list_result)
                else:
                    create_csv(args.out_name_g,
                               ['drug_name', 'algorithm', 'gamma','partition', 'n_parties', 'round', 'Global_train_acc',
                                'Global_test_acc', "macro_F1", "weighted_F1","confusion_matrix", "AUC_ROC"])
                    list_result = [str(args.dataset), args.alg, args.gamma, args.partition, args.n_parties, round, train_acc,
                                   test_acc, macro_F1, weighted_F1, conf_matrix, AUC_ROC]
                    write_csv(args.out_name_g, list_result)
                if args.datadir == "./data/commonWaterfall":
                    results_statistics_client(selected[0], selected[1])

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.dropout_p, args.n_parties, args)
        arr = np.arange(args.n_parties)
        local_train_net(nets, arr, args, net_dataidx_map, test_dl = test_dl_global, device=device, round=round)

    elif args.alg == 'all_in':
        nets, local_model_meta_data, layer_type = init_nets(args.dropout_p, 1, args)
        n_epoch = args.epochs

        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl_global, n_epoch, args.lr, args.optimizer, device=device)

        logger.info("All in test acc: %f" % testacc)

    if os.path.exists(args.datadir + '/results/' + args.out_traindata_cls_count + '.csv'):
        list_result = [str(args.dataset), args.alg, args.partition, args.n_parties, traindata_cls_counts]
        write_csv(args.out_traindata_cls_count, list_result)
    else:
        create_csv(args.out_traindata_cls_count,
                   ['drug_name', 'algorithm', 'partition', 'n_parties', 'traindata_cls_counts'])
        list_result = [str(args.dataset), args.alg, args.partition, args.n_parties, traindata_cls_counts]
        write_csv(args.out_traindata_cls_count, list_result)
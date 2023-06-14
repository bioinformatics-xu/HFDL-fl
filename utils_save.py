import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc
from sklearn.metrics import classification_report
import random
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from torch.utils.data import DataLoader
import copy

from datasets import DrugResponse
from math import sqrt, floor

import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
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

# do mannwhitneyu test for dataframe X1 and X2 with same columns
def mannwhitneyu_test(X1, X2):
    significant_cols = []
    for col_name in X1.columns:
        col_x1 = X1[col_name]
        col_x2 = X2[col_name]

        if len(set(col_x1)) == 1 or len(set(col_x2)) == 1:
            print(col_name)
            print('All numbers are identical in at least one column.')
        else:
            stat, p = mannwhitneyu(col_x1, col_x2)
            if p < 0.01:
                significant_cols.append(col_name)
    return significant_cols


from sklearn.mixture import GaussianMixture
def Gaussian_sample(X_data, y_data, n_comp, sample_num):
    # 构建高斯混合模型
    counts = y_data.value_counts().sort_index().tolist()
    total = sum(counts)
    class_weights = [total / count for count in counts]
    class_weights /= np.sum(class_weights)

    # 将DataFrame和Series转化为NumPy数组
    X_data_np = X_data.values
    y_data_np = y_data.values

    # 使用高斯混合模型拟合原始数据
    gmm = GaussianMixture(n_components=n_comp)
    gmm.fit(X_data_np, y_data_np)
    #gmm.weights_ = class_weights

    np.random.seed(666)  # 设置随机种子
    X_data_new, y_data_new = gmm.sample(sample_num)

    # 将生成的数据添加到原始数据集中
    X_data = np.vstack((X_data, X_data_new))
    y_data = np.hstack((y_data, y_data_new))
    # # 打乱数据集
    indices = np.random.permutation(X_data.shape[0])
    X_data = X_data[indices]
    y_data = y_data[indices]
    print("Gaussian_sample Done")

    return (X_data, y_data)

from sklearn.ensemble import RandomForestClassifier

def RF_featureSelection(X_data, y_data, threshold = 'mean'):
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', criterion = 'gini', n_estimators = 500)
    selector = SelectFromModel(rf, prefit=False, threshold=threshold)

    X_data_np = X_data.values
    y_data_np = y_data.values

    selector.fit(X_data_np, y_data_np)

    print("Feature Thresholds: ", selector.threshold_)

    # 获取选择的特征名称
    selected_features = X_data.columns[selector.get_support()]
    print("Feature number is: ", len(selected_features))
    # 输出选择后的 X 数据
    # selected_X = selector.transform(X)
    return selected_features

from sklearn.ensemble import GradientBoostingClassifier
def GBT_featureSelection(X_data, y_data, threshold = 'mean'):

    X_data_np = X_data.values
    y_data_np = y_data.values

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_data_np, y_data_np)
    selector = SelectFromModel(model, prefit=False, threshold=threshold)

    selected_features = X_data.columns[selector.get_support()]
    print("Feature number is: ", len(selected_features))
    return selected_features

from sklearn.feature_selection import mutual_info_classif

def mutual_info_feature_selection(X_data, y_data, feature_num):
    X_data_np = X_data.values
    y_data_np = y_data.values

    # 假设 X 和 y 分别表示特征和目标变量
    # 调用 mutual_info_classif 函数计算每个特征的信息增益
    mi = mutual_info_classif(X_data_np, y_data_np)
    top_k_idx = mi.argsort()[::-1][:feature_num]

    selected_features = X_data.columns[top_k_idx]

    print("Feature number is: ", len(selected_features))

    return selected_features

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
def ExtraTrees_feature_selection(X_data, y_data, threshold = 'mean'):
    X_data_np = X_data.values
    y_data_np = y_data.values

    # 创建 ExtraTreesClassifier 对象，并设置 n_jobs=4 进行并行计算
    clf = ExtraTreesClassifier(n_estimators=200, n_jobs=4)

    # 创建 SelectFromModel 对象，并调用 fit 方法进行特征选择
    selector = SelectFromModel(clf, threshold = threshold).fit(X_data_np, y_data_np)

    selected_features = X_data.columns[selector.get_support(indices=True)]
    print("Feature number is: ", len(selected_features))
    return(selected_features)

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
def RFECV_featureSelection(X_data, y_data):
    X_data_np = X_data.values
    y_data_np = y_data.values

    # 创建逻辑回归模型
    estimator = LogisticRegression()

    # 创建 RFECV 对象，设置交叉验证次数为 5
    selector = RFECV(estimator, step=50, cv=5)

    # 调用 fit 方法进行特征选择
    selector.fit(X_data_np, y_data_np)
    selected_features = X_data.columns[selector.support_]
    print("Feature number is: ", len(selected_features))
    return selected_features

    # 输出最佳特征子集及其对应的评分
    print(selector.support_)
    print(selector.ranking_)

from sklearn.feature_selection import RFE

def RFE_featureSelection(X_data, y_data, feature_num=1000):
    X_data_np = X_data.values
    y_data_np = y_data.values

    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    selector = RFE(estimator=rf, n_features_to_select=feature_num, step=10)
    selector = selector.fit(X_data_np, y_data_np)

    selected_features = X_data.columns[selector.support_]
    print("Feature number is: ", len(selected_features))
    return selected_features

def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    #np.random.seed(2020)
    #torch.manual_seed(2020)

    if datadir == "./data/common":
        print("predict drug response common")
        cell_features1 = pd.read_csv(datadir + '/GDSC_Expr_feature_common.csv')
        x_origin1 = cell_features1.iloc[:, 1:]

        x_origin1 = pd.DataFrame(x_origin1, columns=x_origin1.columns)
        x_origin1 = x_origin1[x_origin1.columns.sort_values()]
        drug_class1 = pd.read_csv(datadir + '/GDSC_response_class_common.csv')
        y_origin1 = drug_class1[dataset]

        X_train1, X_test1, y_train1, y_test1 = train_test_split(x_origin1, y_origin1, test_size=0.3, random_state=666)
        X_train1 = np.array(X_train1, dtype='float32')
        X_test1 = np.array(X_test1, dtype='float32')
        y_train1 = np.array(y_train1, dtype='int64')
        y_test1 = np.array(y_test1, dtype='int64')

        cell_features2 = pd.read_csv(datadir + '/CTRP_Expr_feature_common.csv')
        x_origin2 = cell_features2.iloc[:, 1:]

        x_origin2 = pd.DataFrame(x_origin2, columns=x_origin2.columns)
        x_origin2 = x_origin2[x_origin2.columns.sort_values()]
        drug_class2 = pd.read_csv(datadir + '/CTRP_response_class_common.csv')
        y_origin2 = drug_class2[dataset]

        X_train2, X_test2, y_train2, y_test2 = train_test_split(x_origin2, y_origin2, test_size=0.3, random_state=666)
        X_train2 = np.array(X_train2, dtype='float32')
        X_test2 = np.array(X_test2, dtype='float32')
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

        n_train = y_train1.shape[0] + y_train2.shape[0]
    elif datadir == "./data/commonWaterfall":
        print("predict drug response common waterfall")

        # take the selected features

        # feature_dir = ("./FedFS-DRP/src/fedFS_" + dataset + "_revision2/")
        # feature_selected_data = pd.read_csv(
        #     feature_dir + 'fed_running_fs-nodes(1.0)_data(1.0)-alpha_(1.0).csv', delimiter='\t')
        # feature_selected = feature_selected_data.loc[feature_selected_data.shape[0] - 1, :]
        # feature_selected = feature_selected.apply(lambda x: [int(num_str) for num_str in x.split(',')])
        # features_selected = list(map(int, feature_selected[0][:]))

        cell_features1 = pd.read_csv(datadir + '/GDSC2_Expr_feature_common_revision1.csv')
        x_origin1 = cell_features1.iloc[:, 1:]
        # scaler = preprocessing.StandardScaler()
        # x_features1 = scaler.fit_transform(x_origin1)
        # x_origin1 = pd.DataFrame(x_features1, columns=x_origin1.columns)
        x_origin1 = pd.DataFrame(x_origin1, columns=x_origin1.columns)
        x_origin1 = x_origin1[x_origin1.columns.sort_values()]
        drug_class1 = pd.read_csv(datadir + '/GDSC2_response_class_common_waterfall_revision1.csv')
        y_origin1 = drug_class1[dataset]

        X_train1, X_test1, y_train1, y_test1 = train_test_split(x_origin1, y_origin1, test_size=0.2, random_state=666)

        # X_train1_sample0 = X_train1.loc[y_train1 == 0]
        # X_train1_sample2 = X_train1.loc[y_train1 == 2]
        #
        # signaficance_genes_GDSC = set(mannwhitneyu_test(X_train1_sample0,X_train1_sample2))

        # CTRP data
        cell_features2 = pd.read_csv(datadir + '/CTRP_Expr_feature_common_revision1.csv')
        x_origin2 = cell_features2.iloc[:, 1:]
        # scaler = preprocessing.StandardScaler()
        # x_features2 = scaler.fit_transform(x_origin2)
        # x_origin2 = pd.DataFrame(x_features2, columns=x_origin2.columns)
        x_origin2 = pd.DataFrame(x_origin2, columns=x_origin2.columns)
        x_origin2 = x_origin2[x_origin2.columns.sort_values()]

        drug_class2 = pd.read_csv(datadir + '/CTRP_response_class_common_waterfall_revision1.csv')
        y_origin2 = drug_class2[dataset]

        X_train2, X_test2, y_train2, y_test2 = train_test_split(x_origin2, y_origin2, test_size=0.2, random_state=666)

        # X_train2_sample0 = X_train2.loc[y_train2 == 0]
        # X_train2_sample2 = X_train2.loc[y_train2 == 2]
        # signaficance_genes_CTRP = set(mannwhitneyu_test(X_train2_sample0, X_train2_sample2))
        #
        # merged_sig_genes = sorted(signaficance_genes_GDSC.intersection(signaficance_genes_CTRP))
        #
        # X_train1 = X_train1.loc[:, merged_sig_genes]
        # X_train2 = X_train2.loc[:, merged_sig_genes]

        # mutual_features_GDSC = mutual_info_feature_selection(X_train1, y_train1, feature_num = floor(X_train1.shape[1]*0.3))
        # mutual_features_CTRP = mutual_info_feature_selection(X_train2, y_train2, feature_num = floor(X_train2.shape[1]*0.3))
        # merged_sig_genes = sorted(mutual_features_GDSC.intersection(mutual_features_CTRP))

        # RF_features_GDSC = RF_featureSelection(X_train1, y_train1, threshold='mean')
        # RF_features_CTRP = RF_featureSelection(X_train2, y_train2, threshold='mean')
        # merged_sig_genes = sorted(RF_features_GDSC.intersection(RF_features_CTRP))

        # GBT_features_GDSC = GBT_featureSelection(X_train1, y_train1, threshold='0.5*mean')
        # GBT_features_CTRP = GBT_featureSelection(X_train2, y_train2, threshold='0.5*mean')
        # merged_sig_genes = sorted(GBT_features_GDSC.intersection(GBT_features_CTRP))

        # RFE_features_GDSC = RFE_featureSelection(X_train1, y_train1, feature_num=floor(X_train1.shape[1]*0.5))
        # RFE_features_CTRP = RFE_featureSelection(X_train2, y_train2, feature_num=floor(X_train2.shape[1]*0.5))
        # merged_sig_genes = sorted(RFE_features_GDSC.intersection(RFE_features_CTRP))
        #
        # print("Merged feature number is: ", len(merged_sig_genes))
        #
        # X_train1 = X_train1.loc[:,merged_sig_genes]
        # X_test1 = X_test1.loc[:,merged_sig_genes]
        #
        # X_train2 = X_train2.loc[:,merged_sig_genes]
        # X_test2 = X_test2.loc[:,merged_sig_genes]

        # X_train1, y_train1 = Gaussian_sample(X_train1, y_train1, n_comp=len(set(y_train1)), sample_num=floor(X_train1.shape[0]))
        # X_train2, y_train2 = Gaussian_sample(X_train2, y_train2, n_comp=len(set(y_train2)), sample_num=floor(X_train2.shape[0]))

        # counts1 = np.unique(y_train1, return_counts=True)[1]
        # counts2 = np.unique(y_train2, return_counts=True)[1]

        # 使用SMOTE算法进行采样
        # from imblearn.over_sampling import SMOTE
        # smote = SMOTE(sampling_strategy='auto', k_neighbors=50)
        # X_train1, y_train1 = smote.fit_resample(X_train1, y_train1)
        #
        # X_train2, y_train2 = smote.fit_resample(X_train2, y_train2)
        # #y_smote2_counts = y_smote2.value_counts()


        # X_train1 = X_train1.iloc[:,features_selected]
        # X_test1 = X_test1.iloc[:, features_selected]

        # pca = PCA(n_components=500)
        # X_train1 = pca.fit_transform(X_train1)
        # X_test1 = pca.transform(X_test1)

        X_train1 = np.array(X_train1, dtype='float32')
        scaler = preprocessing.StandardScaler()
        # scaler = preprocessing.MinMaxScaler()
        X_train1 = scaler.fit_transform(X_train1)

        # # 数据预处理
        # n_features = X_train1.shape[1]
        # n_samples = X_train1.shape[0]
        #
        # alpha = 0.001
        # I = np.eye(n_features) * alpha  # 创建对角矩阵并乘以正则化系数
        #
        # X_train1 = np.pad(X_train1, ((0, n_features - n_samples), (0, n_features - n_features)), mode='constant')
        # X_train1 = X_train1 + I
        # X_train1 = np.array(X_train1, dtype='float32')

        X_test1 = np.array(X_test1, dtype='float32')
        scaler = preprocessing.StandardScaler()
        # scaler = preprocessing.MinMaxScaler()
        X_test1 = scaler.fit_transform(X_test1)
        y_train1 = np.array(y_train1, dtype='int64')
        y_test1 = np.array(y_test1, dtype='int64')

        # X_train2 = X_train2.iloc[:,features_selected]
        # X_test2 = X_test2.iloc[:, features_selected]

        # pca = PCA(n_components=500)
        # X_train2 = pca.fit_transform(X_train2)
        # X_test2 = pca.transform(X_test2)

        X_train2 = np.array(X_train2, dtype='float32')
        scaler = preprocessing.StandardScaler()
        # scaler = preprocessing.MinMaxScaler()
        X_train2 = scaler.fit_transform(X_train2)

        # # 数据预处理
        # n_features = X_train1.shape[1]
        # n_samples = X_train1.shape[0]
        #
        # alpha = 0.001
        # I = np.eye(n_features) * alpha  # 创建对角矩阵并乘以正则化系数
        #
        # X_train1 = np.pad(X_train1, ((0, n_features - n_samples), (0, n_features - n_features)), mode='constant')
        # X_train1 = X_train1 + I
        # X_train2 = np.array(X_train2, dtype='float32')

        X_test2 = np.array(X_test2, dtype='float32')
        scaler = preprocessing.StandardScaler()
        # scaler = preprocessing.MinMaxScaler()
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

    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        # K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100
        elif dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
        elif dataset in ('mnist', 'cifar10', 'femnist', 'svhn', 'generated'):
            K = 10
        else:
            K = 3

        N = y_train.shape[0]
        # np.random.seed(2020)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        else:
            K = 10
        if dataset == "cifar100":
            K = 100
        elif dataset == "tinyimagenet":
            K = 200
        if num == 10:
            net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, n_parties)
                for j in range(n_parties):
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[j])
        else:
            times = [0 for i in range(K)]
            contain = []
            for i in range(n_parties):
                current = [i % K]
                times[i % K] += 1
                j = 1
                while (j < num):
                    ind = random.randint(0, K - 1)
                    if (ind not in current):
                        j = j + 1
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)
            net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, times[i])
                ids = 0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                        ids += 1

    elif partition == "iid-diff-quantity":
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions / proportions.sum()
            min_size = np.min(proportions * len(idxs))
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs, proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "mixed":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100

        N = y_train.shape[0]
        net_dataidx_map = {}

        times = [1 for i in range(10)]
        contain = []
        for i in range(n_parties):
            current = [i % K]
            j = 1
            while (j < 2):
                ind = random.randint(0, K - 1)
                if (ind not in current and times[ind] < 2):
                    j = j + 1
                    current.append(ind)
                    times[ind] += 1
            contain.append(current)
        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions / proportions.sum()
            min_size = np.min(proportions * n_train)

        for i in range(K):
            idx_k = np.where(y_train == i)[0]
            np.random.shuffle(idx_k)

            proportions_k = np.random.dirichlet(np.repeat(beta, 2))
            # proportions_k = np.ndarray(0,dtype=np.float64)
            # for j in range(n_parties):
            #    if i in contain[j]:
            #        proportions_k=np.append(proportions_k ,proportions[j])

            proportions_k = (np.cumsum(proportions_k) * len(idx_k)).astype(int)[:-1]

            split = np.split(idx_k, proportions_k)
            ids = 0
            for j in range(n_parties):
                if i in contain[j]:
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                    ids += 1

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

                #softmax_out = F.softmax(out)
                # out_softmax = F.softmax(out, dim=1)
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

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:,row*size+i,col*size+j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

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

def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, x, mu, var):

        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll

def noise_sample(choice, n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
    """
    Sample random noise vector for training.

    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Dimension of iicompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """

    z = torch.randn(batch_size, n_z, 1, 1, device=device)
    idx = np.zeros((n_dis_c, batch_size))
    if(n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)

        c_tmp = np.array(choice)

        for i in range(n_dis_c):
            idx[i] = np.random.randint(len(choice), size=batch_size)
            for j in range(batch_size):
                idx[i][j] = c_tmp[int(idx[i][j])]

            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if(n_con_c != 0):
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

    noise = z
    if(n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if(n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx


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


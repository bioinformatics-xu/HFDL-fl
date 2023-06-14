import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

result_client = pd.read_csv("./data/commonWaterfall_client/results/results_client_loss.csv")
client_statistic = pd.DataFrame(data=None, columns = ['drug_name','train_test', 'epoch','final_test_acc','macro_F1','weighted_F1','confusion_matrix'], dtype=object)
drug_list = np.unique(result_client['drug_name']).tolist()
best_epoch_df = pd.DataFrame(data=None, columns = ['drug_name', 'train_test','epoch'], dtype=object)
best_criterion = "macro_F1"

train_test_list = np.unique(result_client['train_test']).tolist()

for tt in train_test_list:
    result_global_tt = result_client[result_client['train_test'] == tt]
    for drug in drug_list:
        result_tt_drug = result_global_tt[result_global_tt['drug_name'].str.contains(drug)]
        best_epoch = result_tt_drug['epoch'][result_tt_drug[best_criterion].idxmax()]
        result_tt_drug.index = range(len(result_tt_drug))
        client_statistic = client_statistic.append(result_tt_drug.loc[
                                                       best_epoch, ['drug_name', 'train_test', 'epoch',
                                                                    'final_test_acc', 'macro_F1',
                                                                    'weighted_F1',
                                                                    'confusion_matrix']])
        best_epoch_df = best_epoch_df.append(result_tt_drug.loc[
                                                 best_epoch, ['drug_name', 'train_test', 'epoch']])


client_statistic_filter = client_statistic.loc[(client_statistic['drug_name'].isin(['AZD7762','Fluorouracil','Linsitinib','Olaparib','PLX4720']))]
client_statistic_filter.to_csv("./data/commonWaterfall_client/results/statistic_output/single_client_statistic.csv", index=False)

global_statistic = pd.read_csv("./data/commonWaterfall/results/statistic_output/global_statistic.csv")
HFL_client = pd.read_csv("./data/commonWaterfall/results/results_clientdata.csv")

global_client_test = pd.DataFrame(data=None, columns = ['drug_name','algorithm','gamma','round','type','test_acc','weighted_F1', 'macro_F1','confusion_matrix'], dtype=object)

drug_list = np.unique(global_statistic['drug_name']).tolist()
for drug in drug_list:
    global_statistic_drug = global_statistic[global_statistic['drug_name'].str.contains(drug)]
    best_index = global_statistic_drug[best_criterion].idxmax()

    HFL_client_drug = HFL_client[(HFL_client['drug_name'].str.contains(drug)) & (HFL_client['algorithm'].str.contains(global_statistic_drug.loc[best_index,'algorithm'])) & (HFL_client['gamma']==global_statistic_drug.loc[best_index,'gamma']) & (HFL_client['round']==global_statistic_drug.loc[best_index,'round'])]
    global_results = HFL_client_drug.loc[
        (HFL_client_drug['type'].str.contains('global_d0') | HFL_client_drug['type'].str.contains('global_d1')), ['drug_name', 'algorithm', 'gamma', 'round', 'type',
                                                                    'test_acc', 'weighted_F1', 'macro_F1','confusion_matrix']]
    global_client_test = global_client_test.append(global_results)

global_client_test.to_csv("./data/commonWaterfall/results/statistic_output/sup_global_client_test.csv", index=False)
client_statistic_filter.to_csv("./data/commonWaterfall/results/statistic_output/sup_client_test.csv", index=False)


def plot_global_client(test_dataset, criterion_plot_client):
    names = drug_list
    x = range(len(names))
    if criterion_plot_client=="final_test_acc":
        criterion_plot_global = "test_acc"
        y_lable = "ACC"
    else:
        criterion_plot_global = criterion_plot_client
        y_lable = criterion_plot_client

    y_1 = client_statistic_filter.loc[
        client_statistic_filter['train_test'].str.contains(test_dataset+'_'+test_dataset), criterion_plot_client]
    # y_2 = client_statistic_filter.loc[
    #     client_statistic_filter['train_test'].str.contains('CTRP_'+test_dataset), criterion_plot_client]
    if test_dataset=="GDSC":
        y_3 = global_client_test.loc[global_client_test['type'].str.contains('global_d0'), criterion_plot_global]
    elif test_dataset=="CTRP":
        y_3 = global_client_test.loc[global_client_test['type'].str.contains('global_d1'), criterion_plot_global]

    plt.plot(x, y_1, color='green', marker='o', linestyle='-', label= test_dataset+'_train_'+test_dataset+'_test')
    #plt.plot(x, y_2, color='blueviolet', marker='o', linestyle='-', label='CTRP_train_'+test_dataset+'_test')
    plt.plot(x, y_3, color='orangered', marker='o', linestyle='-', label='Global_train_'+test_dataset+'_test')
    plt.legend()  # 显示图例
    plt.xticks(x, names, rotation=0)
    plt.xlabel(" ")  # X轴标签
    plt.ylabel(y_lable)  # Y轴标签
    #plt.show()

def plot_global_client_2(test_dataset, criterion_plot_client):
    names = drug_list
    x = range(len(names))
    if criterion_plot_client=="final_test_acc":
        criterion_plot_global = "test_acc"
        y_lable = "ACC"
    else:
        criterion_plot_global = criterion_plot_client
        y_lable = criterion_plot_client

    y_1 = client_statistic_filter.loc[
        client_statistic_filter['train_test'].str.contains('GDSC'+'_'+test_dataset), criterion_plot_client]
    y_2 = client_statistic_filter.loc[
        client_statistic_filter['train_test'].str.contains('CTRP_'+test_dataset), criterion_plot_client]
    if test_dataset=="GDSC":
        y_3 = global_client_test.loc[global_client_test['type'].str.contains('global_d0'), criterion_plot_global]
    elif test_dataset=="CTRP":
        y_3 = global_client_test.loc[global_client_test['type'].str.contains('global_d1'), criterion_plot_global]

    plt.plot(x, y_1, color='green', marker='o', linestyle='-', label= "GDSC" +'_train_'+test_dataset+'_test')
    plt.plot(x, y_2, color='blueviolet', marker='o', linestyle='-', label='CTRP_train_'+test_dataset+'_test')
    plt.plot(x, y_3, color='orangered', marker='o', linestyle='-', label='Global_train_'+test_dataset+'_test')
    plt.legend()  # 显示图例
    plt.xticks(x, names, rotation=0)
    plt.xlabel(" ")  # X轴标签
    plt.ylabel(y_lable)  # Y轴标签
    #plt.show()


plt.figure(figsize=(12.0,8.0))
plt.subplot(321)
ax1 = plot_global_client("GDSC", "final_test_acc")
plt.subplot(322)
ax4 = plot_global_client("CTRP", "final_test_acc")
plt.subplot(323)
ax2 = plot_global_client("GDSC", "weighted_F1")
plt.subplot(324)
ax4 = plot_global_client("CTRP", "weighted_F1")
plt.subplot(325)
ax3 = plot_global_client("GDSC", "macro_F1")
plt.subplot(326)
ax6 = plot_global_client("CTRP", "macro_F1")
plt.tight_layout()
plt.savefig('./figures_FL_DRP/global_client_test_GDSC_CTRP.pdf', bbox_inches='tight') # 保存成PDF放大后不失真（默认保存在了当前文件夹下）
#plt.show()




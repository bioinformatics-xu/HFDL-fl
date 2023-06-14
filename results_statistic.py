import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

result_global = pd.read_csv("./data/commonWaterfall/results/results_global_revision1.csv")
row_n = len(np.unique(result_global['gamma']))*len(np.unique(result_global['algorithm']))*len(np.unique(result_global['drug_name']))
global_statistic = pd.DataFrame(data=None, columns = ['gamma','algorithm','round','drug_name','Global_test_acc','macro_F1','weighted_F1','confusion_matrix'], dtype=object)
gm_list = np.unique(result_global['gamma']).tolist()
gm_list.sort()
alg_list = np.unique(result_global['algorithm']).tolist()
drug_list = np.unique(result_global['drug_name']).tolist()
best_round_df = pd.DataFrame(data=None, columns = ['gamma', 'algorithm', 'drug_name', 'round'], dtype=object)
best_criterion = "Global_test_acc"

for gm in gm_list:
    result_global_gamma = result_global[result_global['gamma'] == gm]
    for drug in drug_list:
        for alg in alg_list:
            result_drug_alg = result_global_gamma[(result_global_gamma['drug_name'].str.contains(drug)) & (
                result_global_gamma['algorithm'].str.contains(alg))]
            result_drug_alg.index = range(len(result_drug_alg))
            # 取最优的round对应的结果
            best_round = result_drug_alg['round'][result_drug_alg[best_criterion].idxmax()]
            global_statistic = global_statistic.append(result_drug_alg.loc[
                                        best_round, ['gamma', 'algorithm','round', 'drug_name', 'Global_test_acc', 'macro_F1',
                                                     'weighted_F1',
                                                     'confusion_matrix']])
            best_round_df = best_round_df.append(result_drug_alg.loc[
                                     best_round, ['gamma', 'algorithm', 'drug_name', 'round']])


global_statistic_filter = global_statistic.loc[(global_statistic['drug_name'].isin(['AZD7762','Fluorouracil','Linsitinib','Olaparib','PLX4720']))]
gamma_summary = pd.DataFrame(data=None, columns=['gamma','algorithm','Global_test_acc','macro_F1','weighted_F1'],dtype=object)

for gm in gm_list:
    results_gm = global_statistic_filter[global_statistic_filter['gamma']==gm]
    results_gm = results_gm.drop(['drug_name','confusion_matrix'], axis=1)
    avg_acc_gm = results_gm.groupby(['gamma', 'algorithm']).mean().reset_index()
    gamma_summary = gamma_summary.append(avg_acc_gm)

gamma_summary = gamma_summary.loc[(gamma_summary['algorithm'].isin(['fedavg','fednova','scaffold']))]

global_statistic_filter = global_statistic_filter.loc[(global_statistic_filter['algorithm'].isin(['fedavg','fednova','scaffold']))]
global_statistic_filter.to_csv("./data/commonWaterfall/results/statistic_output/global_statistic_revision1.csv", index=False)



# plot the boxplot of the criterion for algorithms
plt.figure(figsize=(9.0,9.0))
plt.subplot(311)
ax1 = sns.boxplot(x="gamma", y="Global_test_acc", hue="algorithm", data=global_statistic_filter, showfliers=True, palette="Set1", width=0.5)
ax1.set(ylim=(0.5, 0.8))
plt.subplot(312)
ax2 = sns.boxplot(x="gamma", y="weighted_F1", hue="algorithm", data=global_statistic_filter, showfliers=True, palette="Set1", width=0.5)
ax2.set(ylim=(0.5, 0.8))
plt.subplot(313)
ax2 = sns.boxplot(x="gamma", y="macro_F1", hue="algorithm", data=global_statistic_filter, showfliers=True, palette="Set1", width=0.5)
ax2.set(ylim=(0.3, 0.6))
plt.tight_layout()
plt.savefig('./figures_FL_DRP/global_drug_boxplot_revision1.pdf', bbox_inches='tight') # 保存成PDF放大后不失真（默认保存在了当前文件夹下）
#plt.show()

# plot the mean criterion of algorithm
def plot_bar_criterion(gamma_summary,criterion):
    x = np.arange(0, len(np.unique(gamma_summary['gamma'])) * 4, 4)
    width = 1
    x1 = x - 3 * width / 4
    x2 = x - width / 4
    x3 = x + width / 4
    #x4 = x + 3 * width / 4
    y1 = np.array(gamma_summary[(gamma_summary['algorithm'].str.contains('fednova'))][criterion].tolist())
    y2 = np.array(gamma_summary[(gamma_summary['algorithm'].str.contains('fedavg'))][criterion].tolist())
    y3 = np.array(gamma_summary[(gamma_summary['algorithm'].str.contains('scaffold'))][criterion].tolist())
    y4 = np.array(gamma_summary[(gamma_summary['algorithm'].str.contains('fedprox'))][criterion].tolist())

    # plt.bar(x1,y1,width=width/2,label='fednova',color='#EB5353',zorder=4)
    # plt.bar(x2,y2,width=width/2,label='fedavg',color='#34B3F1',zorder=4)
    # plt.bar(x3,y3,width=width/2,label='scaffold',color='#FBCB0A',zorder=4)
    # plt.bar(x4,y4,width=width/2,label='fedprox',color='#5FD068',zorder=4)

    plt.ylim(0, 1)
    plt.bar(x1, y1, width=width / 2, label='fednova', zorder=4)
    plt.bar(x2, y2, width=width / 2, label='fedavg', zorder=4)
    plt.bar(x3, y3, width=width / 2, label='scaffold', zorder=4)
    #plt.bar(x4, y4, width=width / 2, label='fedprox', zorder=4)

    # 添加x,y轴名称、图例和网格线
    # plt.xlabel('gamma',fontsize=11)
    plt.ylabel(criterion, fontsize=11)
    plt.legend(loc=0, ncol=1)
    plt.grid(ls='--', alpha=0.5)

    # 功能2
    for i, j in zip(x1, y1):
        plt.text(i, j + 0.01, "%.4f" % j, ha="center", va="bottom", fontsize=8)
    for i, j in zip(x2, y2):
        plt.text(i, j + 0.01, "%.4f" % j, ha="center", va="bottom", fontsize=8)
    for i, j in zip(x3, y3):
        plt.text(i, j + 0.01, "%.4f" % j, ha="center", va="bottom", fontsize=8)
    # for i, j in zip(x4, y4):
    #     plt.text(i, j + 0.01, "%.4f" % j, ha="center", va="bottom", fontsize=7)

    # 修改x刻度标签为对应日期
    xticks_gamma = ['gamma=' + str(np.unique(gamma_summary['gamma'])[0]),
                    'gamma=' + str(np.unique(gamma_summary['gamma'])[1]),
                    'gamma=' + str(np.unique(gamma_summary['gamma'])[2])]
    plt.xticks(x, xticks_gamma, fontsize=10)
    plt.tick_params(axis='x', length=0)

plt.figure(figsize=(9.0,9.0))
plt.subplot(311)
plot_bar_criterion(gamma_summary,'Global_test_acc')
plt.subplot(312)
plot_bar_criterion(gamma_summary,'weighted_F1')
plt.subplot(313)
plot_bar_criterion(gamma_summary,'macro_F1')
plt.tight_layout()
plt.savefig('./figures_FL_DRP/global_drug_mean_bar_revision1.pdf', bbox_inches='tight') # 保存成PDF放大后不失真（默认保存在了当前文件夹下）
#plt.show()
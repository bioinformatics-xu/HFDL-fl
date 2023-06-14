import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# dataset = "CTRP"
dataset = "GDSC"
result_global = pd.read_csv("./data/drugResponse"+dataset+"/results/results_global.csv")
global_statistic = pd.DataFrame(data=None, columns = ['gamma','algorithm','round','drug_name','Global_test_acc','macro_F1','weighted_F1','confusion_matrix'], dtype=object)
gm_list = np.unique(result_global['gamma']).tolist()
gm_list.sort()
alg_list = np.unique(result_global['algorithm']).tolist()
drug_list = np.unique(result_global['drug_name']).tolist()
best_round_df = pd.DataFrame(data=None, columns = ['gamma', 'algorithm', 'drug_name', 'round'], dtype=object)
best_criterion = "macro_F1"



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
global_statistic_filter.to_csv("./data/drugResponse"+dataset+"/results/within_study_"+dataset+".csv", index=False)

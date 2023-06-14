import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

CTRP_response = pd.read_csv("./data/commonWaterfall/CTRP_response_revision1.csv",index_col=0)
CTRP_response_class = pd.read_csv("./data/commonWaterfall/CTRP_response_class_common_waterfall_revision1.csv",index_col=0)
GDSC_response = pd.read_csv("./data/commonWaterfall/GDSC2_response_revision1.csv",index_col=0)
GDSC_response_class = pd.read_csv("./data/commonWaterfall/GDSC2_response_class_common_waterfall_revision1.csv",index_col=0)

CTRP_response_common = CTRP_response.loc[:,['AZD7762', 'PLX-4720', 'olaparib', 'dasatinib', 'docetaxel:tanespimycin (2:1 mol/mol)','linsitinib','fluorouracil']]
CTRP_response_common.columns = ['AZD7762', 'PLX4720', 'Olaparib','Dasatinib', 'Docetaxel', 'Linsitinib','Fluorouracil']

GDSC_response_common = GDSC_response.loc[:,['AZD7762_1022', 'PLX-4720_1036', 'Olaparib_1017', 'Dasatinib_1079', 'Docetaxel_1007', 'Linsitinib_1510', '5-Fluorouracil_1073']]
GDSC_response_common.columns = ['AZD7762', 'PLX4720', 'Olaparib','Dasatinib', 'Docetaxel', 'Linsitinib','Fluorouracil']
drug_list = GDSC_response_common.columns

#plot the response distribution in GDSC and CTRP
pp = PdfPages('./figures_FL_DRP/response_distribution_revision1.pdf')
fig,axes=plt.subplots(2,7)

for gi in range(len(drug_list)):
    d = np.array(GDSC_response_common[drug_list[gi]].tolist())
    sns.set_style('darkgrid')
    sns.distplot(d, norm_hist=False, kde=False, color='#00BFFF', bins=20, axlabel=drug_list[gi],ax=axes[0, gi])
    # bins=20,
    if gi==0:
        axes[0, gi].set_ylabel('GDSC',fontsize=12)

for gi in range(len(drug_list)):
    d = np.array(CTRP_response_common[drug_list[gi]].tolist())
    sns.set_style('darkgrid')
    sns.distplot(d,norm_hist=False,kde=False,color='#00BFFF',bins=15, axlabel=drug_list[gi],ax=axes[1,gi])
    #bins=13,
    if gi==0:
        axes[1, gi].set_ylabel('CTRP',fontsize=12)
fig.set_size_inches(16, 7)
plt.tight_layout()
pp.savefig()
pp.close()


# plot the number of samples in each classification
class_statistic_GDSC = pd.DataFrame(data=None,columns=drug_list)
for di in range(len(drug_list)):
    class_target_GDSC = GDSC_response_class[drug_list[di]]
    class_target_GDSC = class_target_GDSC.value_counts()
    class_target_GDSC = class_target_GDSC.sort_index()
    class_statistic_GDSC[drug_list[di]] =  class_target_GDSC.tolist()

class_statistic_CTRP = pd.DataFrame(data=None,columns=drug_list)
for di in range(len(drug_list)):
    class_target_CTRP = CTRP_response_class[drug_list[di]]
    class_target_CTRP = class_target_CTRP.value_counts()
    class_target_CTRP = class_target_CTRP.sort_index()
    class_statistic_CTRP[drug_list[di]] =  class_target_CTRP.tolist()

def plot_sample_number(class_statistic, y_label):
    x = np.arange(0, class_statistic.shape[1] * 4, 4)
    width = 2
    x1 = x - width / 3
    #x2 = x
    x2 = x + width / 3
    y1 = np.array(class_statistic.iloc[0].tolist())
    y2 = np.array(class_statistic.iloc[1].tolist())
    #y3 = np.array(class_statistic.iloc[2].tolist())

    # plt.ylim(0, 1)
    plt.bar(x1, y1, width=width / 1.5, label='resistant')
    #plt.bar(x2, y2, width=width / 2, label='intermediate')
    plt.bar(x2, y2, width=width / 1.5, label='sensitive')

    # 添加x,y轴名称、图例和网格线
    # plt.xlabel('gamma',fontsize=11)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(loc=0, ncol=1, bbox_to_anchor=(1.0, 1.0))
    plt.grid(ls='--', alpha=0.5)

    # 功能2
    for i, j in zip(x1, y1):
        plt.text(i, j + 0.01, "%.0f" % j, ha="center", va="bottom", fontsize=8)
    for i, j in zip(x2, y2):
        plt.text(i, j + 0.01, "%.0f" % j, ha="center", va="bottom", fontsize=8)
    # for i, j in zip(x3, y3):
    #     plt.text(i, j + 0.01, "%.0f" % j, ha="center", va="bottom", fontsize=8)

    # 修改x刻度标签为对应日期
    plt.xticks(x, drug_list, fontsize=10)
    plt.tick_params(axis='x', length=0)

plt.figure(figsize=(10.0,7.0))
plt.subplot(211)
plot_sample_number(class_statistic_GDSC, "GDSC")
plt.subplot(212)
plot_sample_number(class_statistic_CTRP, "CTRP")
# plt.subplot(133)
plt.tight_layout()
plt.savefig('./figures_FL_DRP/sample_number_revision1.pdf', bbox_inches='tight') # 保存成PDF放大后不失真（默认保存在了当前文件夹下）
plt.show()

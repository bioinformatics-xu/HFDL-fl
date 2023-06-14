import pandas as pd
import numpy as np
Feat_DF = pd.read_csv("GDSC2_Expr_feature.csv")
CGC_genes = pd.read_csv("GDSC2_Expr_CGC_feature.csv")

# take the feature name list
feature_names_list = Feat_DF.columns.tolist()
feature_names_list.remove('Unnamed: 0')

# take the CGC gene list
CGC_list = CGC_genes.columns.tolist()
CGC_list.remove('Unnamed: 0')

CGC_feat = Feat_DF[CGC_list]

feat_all = Feat_DF[feature_names_list]
featCor = feat_all.corr()

CGC_feat_cor = featCor[CGC_list]
CGC_feat_cor.drop(CGC_list, axis = 0, inplace=True)
#CGC_feat_cor.loc['CASP10',:][(CGC_feat_cor.loc['CASP10',:]>0.5)&(CGC_feat_cor.loc['CASP10',:]<1)]

indices = np.where((CGC_feat_cor>0.5).any(axis=1))

Feat_DF_filtered = Feat_DF[set(feature_names_list)-set(CGC_feat_cor.index[indices].tolist())]

#from sklearn.feature_selection import VarianceThreshold
#X = VarianceThreshold(threshold=0.1).fit_transform(Feat_DF_filtered)

feat_var = np.var(Feat_DF_filtered)
Feat_DF_filtered = Feat_DF_filtered[list(feat_var[feat_var.values >= 0.5].index)]

Feat_DF_filtered.index = Feat_DF['Unnamed: 0']

# 原始的index保存到csv文件
Feat_DF_filtered.to_csv("GDSC2_filtered_featrue.csv")
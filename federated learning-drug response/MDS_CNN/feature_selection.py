from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit
import numpy as np
import pandas as pd

Feat_DF = pd.read_csv(
    "GDSC2_Expr_CGC_feature.csv")  # Load the drug descriptors of the drugs applied on the selected cell line
gene_name_list = list(Feat_DF.columns)
del(gene_name_list[0])
# Features
X = Feat_DF.values
X = X[:, 2:]

FilteredDF = pd.read_csv("GDSC2_response_class.csv")
drug_names = FilteredDF.columns.tolist()
del(drug_names[0])

Results_Dic = {}

for SEL_CEL in drug_names:

    Y = np.array(FilteredDF[SEL_CEL])
    rf = RandomForestClassifier(criterion='gini',
                                n_estimators=25,
                                random_state=1,
                                n_jobs=2)

    # rf = RandomForestRegressor(n_estimators=20, max_depth=4)
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    print(len(X))
    # cv=3
    for i in range(X.shape[1]):
        score = cross_val_score(rf, X[:, i:i + 1], Y,
                                cv=cv)
        #  scoring="r2"
        scores.append((round(np.mean(score), 3), gene_name_list[i]))
    print(sorted(scores, reverse=True))
    Results_Dic[SEL_CEL] = scores










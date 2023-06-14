import pandas as pd
import os
from math import floor

# rm ./data/commonWaterfall/results/results_global.csv
# cat ./data/commonWaterfall/results/results_global.csv
# rm ./data/commonWaterfall/results/traindata_cls_count.csv
# nohup python3 run_DRP.py &
# conda activate NIID
# pip3 install imblearn -i https://mirrors.aliyun.com/pypi/simple/
# ps -ef | grep 'experiments.py'

# datadir = "./data/commonWaterfall"
# y_total = pd.read_csv(datadir + "/CTRP_response_class_common_waterfall.csv")
#
# drug_names = y_total.columns[1:8]
# print(drug_names)

# for gm in [3,2,1]:
# 	for alg in ['fedavg', 'fednova', 'scaffold']:
# 		for drug_name in drug_names:
# 			os.system(
# 				'python' + ' experiments.py' + ' --model=mlp' + ' --partition=two-set' + ' --comm_round=60' + ' --out_name_g=results_global_revision1' + ' --out_name_c=results_client_revision1' + ' --out_name_l=results_local_revision1' + ' --out_traindata_cls_count=traindata_cls_count_revision1' + ' --gamma='+ str(gm) + ' --dataset=' + drug_name + ' --datadir=' + datadir + ' --alg=' + alg + ' --epochs=10' + ' --batch-size=64' + ' --n_parties=2')
# 			print('gamma='+ ' ' + str(gm) + ' ' +alg + ' ' + drug_name)


datadir = "./data/drugResponseGDSC"
y_total = pd.read_csv(datadir + "/response_class.csv")

drug_names = y_total.columns[2:6]
#print(drug_names)

for gm in [1]:
	#'fedavg', 'fednova', 'fedprox', 'scaffold'
	for alg in ['scaffold']:
		for drug_name in drug_names:
			os.system(
				'python' + ' experiments.py' + ' --model=mlp' + ' --partition=homo' + ' --comm_round=60' + ' --out_name_g=results_within_global_revision1' + ' --out_name_l=results_within_local_revision1' + ' --out_traindata_cls_count=traindata_cls_count_within_revision1' + ' --gamma='+ str(gm) + ' --dataset=' + drug_name + ' --datadir=' + datadir + ' --alg=' + alg + ' --epochs=10' + ' --batch-size=64' + ' --n_parties=2')
			print('gamma='+ ' ' + str(gm) + ' ' +alg + ' ' + drug_name)




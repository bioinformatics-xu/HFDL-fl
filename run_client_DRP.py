import pandas as pd
import os

# rm ./data/commonWaterfall/results/results_global.csv
# cat ./data/commonWaterfall/results/results_global.csv
# rm ./data/commonWaterfall/results/traindata_cls_count.csv
# nohup python3 run_DRP.py &
# conda activate NIID

# datadir = "./data/commonWaterfall"
# y_total = pd.read_csv(datadir + "/CTRP_response_class_common_waterfall.csv")
#
# drug_names = y_total.columns[1:8]
# print(drug_names)
#
# for gm in [2]:
# 	#'fedavg', 'fednova', 'fedprox', 'scaffold'
# 	for alg in ['scaffold']:
# 		for drug_name in drug_names:
# 			os.system(
# 				'python' + ' experiments.py' + ' --model=mlp' + ' --partition=two-set' + ' --comm_round=60' + ' --gamma='+ str(gm) + ' --dataset=' + drug_name + ' --datadir=' + datadir + ' --alg=' + alg + ' --epochs=10' + ' --batch-size=64' + ' --n_parties=2')
# 			print('gamma='+ ' ' + str(gm) + ' ' +alg + ' ' + drug_name)

datadir = "./data/commonWaterfall_client"
y_total = pd.read_csv(datadir + "/CTRP_response_class_common_waterfall_revision1.csv")
drug_names = y_total.columns[[1,2,3,6,7]]

#print(drug_names)
for drug_name in drug_names:
	os.system(
		'python' + ' client_DRP_revision1.py' + ' --dataset=' + drug_name + ' --datadir=' + datadir +  ' --epochs=30' + ' --batch-size=64')
	print(drug_name)




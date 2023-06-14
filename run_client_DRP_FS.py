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
y_total = pd.read_csv(datadir + "/CTRP_response_class_common_waterfall.csv")

drug_names = y_total.columns[1:6]
#print(drug_names)

for drug_name in drug_names:
	os.system(
		'python' + ' client_DRP.py' + ' --dataset=' + drug_name + ' --datadir=' + datadir +  ' --epochs=30' + ' --batch-size=64')
	print(drug_name)

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype, load_breast_cancer

import custom_utils as cut
import federated_feature_selection as ffs
from sklearn import preprocessing


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--datadir', type=str, default='/data/commonWaterfall', help='data directory name')
	parser.add_argument('--drug_name', type=str, default='PLX4720', help='drug name used for training')
	# AZD7762
	parser.add_argument('--log_path', type=str, default='fedFS_DRP', help='the log path')
	parser.add_argument('--MAX_ROUND', type=int, default='60', help='the maximum round')
	args = parser.parse_args()
	return args


def main(datadir, drug_name, log_path, MAX_ROUND):

	cell_features1 = pd.read_csv(datadir + '/GDSC2_Expr_feature_common_revision2.csv')
	x_origin1 = cell_features1.iloc[:, 1:]
	x_origin1 = pd.DataFrame(x_origin1, columns=x_origin1.columns)
	x_origin1 = x_origin1[x_origin1.columns.sort_values()]
	drug_class1 = pd.read_csv(datadir + '/GDSC2_response_class_common_waterfall_revision2.csv')
	# y_origin1 = drug_class1[dataset]
	y_origin1 = drug_class1[drug_name]

	X_train1, X_test1, y_train1, y_test1 = train_test_split(x_origin1, y_origin1, test_size=0.2, random_state=66)
	X_train1 = np.array(X_train1, dtype='float32')
	scaler = preprocessing.StandardScaler()
	X_train1 = scaler.fit_transform(X_train1)

	X_test1 = np.array(X_test1, dtype='float32')
	scaler = preprocessing.StandardScaler()
	X_test1 = scaler.fit_transform(X_test1)

	y_train1 = np.array(y_train1, dtype='int64')
	y_test1 = np.array(y_test1, dtype='int64')

	cell_features2 = pd.read_csv(datadir + '/CTRP_Expr_feature_common_revision2.csv')
	x_origin2 = cell_features2.iloc[:, 1:]

	x_origin2 = pd.DataFrame(x_origin2, columns=x_origin2.columns)
	x_origin2 = x_origin2[x_origin2.columns.sort_values()]
	drug_class2 = pd.read_csv(datadir + '/CTRP_response_class_common_waterfall_revision2.csv')
	y_origin2 = drug_class2[drug_name]

	X_train2, X_test2, y_train2, y_test2 = train_test_split(x_origin2, y_origin2, test_size=0.2, random_state=66)
	X_train2 = np.array(X_train2, dtype='float32')
	scaler = preprocessing.StandardScaler()
	X_train2 = scaler.fit_transform(X_train2)

	X_test2 = np.array(X_test2, dtype='float32')
	scaler = preprocessing.StandardScaler()
	X_test2 = scaler.fit_transform(X_test2)

	y_train2 = np.array(y_train2, dtype='int64')
	y_test2 = np.array(y_test2, dtype='int64')

	X_train = np.vstack([X_train1, X_train2])
	y_train = np.concatenate([y_train1, y_train2])

	net_1_num = np.array(range(0, y_train1.shape[0]))
	net_2_num = np.array(range(y_train1.shape[0], y_train1.shape[0] + y_train2.shape[0]))
	idxs = {0: net_1_num, 1: net_2_num}

	# log_path = '../stats/fedfs-bc/'
	# log_path = '../stats/fedfs-DRP/'
	if not os.path.isdir(log_path):
		print('Creating log dir: {}'.format(log_path))
		os.mkdir(log_path)
	print('Running Federated Crossentropy with alpha {}'.format(alpha_smooth))
	for frac_nodes in perc_nodes:

		for frac_data in perc_data:
			print(
				'*** New Configuration: [nodes {}, data {}]'.format(frac_nodes, frac_data))
			fs, net, prob, conv_round, fed_running_fs, fed_running_prob, dwl_data_before_conv, max_rounds, alpha_per_round = ffs.federated_feature_selection(
				X_train, y_train, idxs, fraction_selected_workers=frac_nodes, fraction_local_subsample=frac_data,
				early_stop_tolerance=tolerance, ce_selection_prob=selection_prob, ce_tolerance=ce_tol,
				alpha=alpha_smooth, alpha_step=alpha_step, max_sample_ce=ce_max_samples, max_step_ce=ce_max_steps,
				fed_max_round=MAX_ROUND, verbose=True, verbose_scenario=False, njobs=njobs)

			print('Saving logs...')
			sfx = '-nodes({})_data({})-alpha_({}).csv'.format(frac_nodes, frac_data, alpha_smooth)
			cut.write_tocsv(log_path + 'fed_fs' + sfx, map(lambda x: [x], fs))
			cut.write_tocsv(log_path + 'fed_prob' + sfx, map(lambda x: [x], prob))
			cut.write_tocsv(log_path + 'fed_dwl' + sfx,
							map(lambda x: [x], dwl_data_before_conv))
			cut.write_tocsv(log_path + 'fed_running_prob' + sfx, fed_running_prob)
			cut.write_tocsv(log_path + 'fed_running_fs' + sfx, fed_running_fs)

			cut.write_tocsv(log_path + 'fed_running_alpha' +
							sfx, map(lambda x: [x], alpha_per_round))
			cut.write_tocsv(log_path + 'fed_running_stats' + sfx, [
				[conv_round, net, max_rounds, tolerance, ce_max_samples, ce_max_steps, int(num_workers * frac_nodes)]],
							header=['Convergence Round', 'Net. Overhead', 'Max Rounds', 'Tolerance', 'CE max samples',
									'CE max steps', 'Active workers'])


if __name__ == "__main__":
	args = get_args()
	main(args.datadir, args.drug_name, args.log_path, args.MAX_ROUND)



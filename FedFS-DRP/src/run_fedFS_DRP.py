import pandas as pd
import os
#datadir = "./data/commonWaterfall"
parent_dir = os.path.dirname(os.getcwd())
datadir = (os.path.dirname(parent_dir) + '/data/commonWaterfall')
y_total = pd.read_csv(datadir + "/CTRP_response_class_common_waterfall_revision1.csv")

drug_names = y_total.columns[1:8]

for drug_name in drug_names:
	print('drug ' + drug_name)
	os.system(
		'python' + ' main_fedfs.py' + ' --datadir=' + datadir + ' --drug_name=' + drug_name + ' --log_path=fedFS_' + drug_name + '_revision1/' + ' --MAX_ROUND=30')

import sys
import numpy as np
import tqdm 
from typing import List
import time 
from sklearn.model_selection import GridSearchCV
import pandas as pd
import model_utils as mu
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import argparse
from collections import defaultdict
import os
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse
import permutation_utils as pu
import utils 



def main():

	# args
	parser = argparse.ArgumentParser("model specific bake-off")
	parser.add_argument('-drug',help = "drug being tested")
	parser.add_argument('-settings', help = "setting string of the form tissue.balance")
	

	args = vars(parser.parse_args())

	drug:str = args['drug']
	
	settings:str = args['settings'].split(".")
	
	tissue,balanced = settings[0],settings[1]
	
	model:str = "RandomForest"
	test_type:int = 1
	seed:int = 1234
	bal_string = "balanced" if balanced=='True' else "unbalanced"
	


	res_dir = f"../results/{drug}/{tissue}/perm_tests/"

	os.makedirs(res_dir,exist_ok = True)
	
	num_balanced_datasets:int = 10
	num_permutations:int = 50
	
	icis = ['Atezo','Pembro','Ipi', 'Nivo']
	ds_string = "cri" if drug in icis else "ccle"

	
	rng = np.random.default_rng(seed)
	rstate = np.random.RandomState(seed)



	genesets:List[str] = ['EXPRESSION','FEATURES']

	expr_file = f"../expression/{ds_string}/{drug}/{tissue}/expression_full.csv"
	resp_file = f"../expression/{ds_string}/{drug}/{tissue}/response.csv"
	feat_file = f"../expression/{ds_string}/{drug}/{tissue}/features.csv"



	expression = pd.read_csv(expr_file)
	response = pd.read_csv(resp_file)
	features = pd.read_csv(feat_file)

	full_results = []
	for gs in tqdm.tqdm(genesets,leave= False):
		if gs =='EXPRESSION':
			


			# with open("../genesets/{g}.txt".format(g=gs),"r") as istream:
			# 	lines = istream.readlines()
			# lines = [x.rstrip() for x in lines]
			genes = utils.fetch_union_genesets()

			keep_cols = [x for x in expression.columns if x in genes]
			X_full = np.log2(expression[keep_cols].values+1)
		
		else:
			X_full = features[features.columns[1:]].values
		
		y_full = response['Response'].values


		# X_full, y_full are the original unpermuted or subsampled datasets

		if balanced.lower()=="true":
			results = pu.run_balanced_test(
				X_full, 
				y_full, 
				num_balanced_datasets, 
				num_permutations,
				test_type,
				model,
				rng)
		else:
			results = pu.run_full_test(
				X_full, 
				y_full, 
				num_permutations,
				test_type,
				model,
				rng)
		
		results['geneset'] = gs
		full_results.append(results)
	results = pd.concat(full_results,axis=0)
	fname = f"{res_dir}{bal_string}.csv"
	results.to_csv(fname, index=False)
if __name__ == '__main__':
	main()
		
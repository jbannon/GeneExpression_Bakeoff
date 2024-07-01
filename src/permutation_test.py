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

	"""
	Running 
		- drug (the drug under consideration)
		- tissue (the tissue the drug is looking at)
		- model (the type of model we're using)
		- split

		ex: 

		python3 permutation_test.py -drug Atezo -model RandomForest -settings BLCA.LOO
	"""



	parser = argparse.ArgumentParser("model specific bake-off")
	parser.add_argument('-drug',help = "drug being tested")
	parser.add_argument('-model',help = "Classifier model class")
	parser.add_argument('-settings', help = "setting string of the form tissue.split")
	

	args = vars(parser.parse_args())

	drug:str = args['drug']
	model:str = args['model']
	settings:str = args['settings'].split(".")
	
	tissue, split_type = settings[0],settings[1]
	
	split_type = split_type.upper()
	assert split_type in ["LOO","MC"], "split_type must be one of ['MC','LOO']"
	

	print("\n*********** PERMUTATION TESTS **************")
	print(f"*\t Working on {drug}")
	print(f"*\t In {tissue}")
	print(f"*\t Using a {split_type} split.")
	print(f"*\t And a {model} Model")
	print("\n*************************\n")
	train_pct:float = 0.8
	test_type:int = 1
	seed:int = 12345
	
	
	test_string = "Type_A" if test_type == 1 else "Type_B"

	res_dir = f"../results/{drug}/{tissue}/perm_tests/{model}/{test_string}/"

	os.makedirs(res_dir,exist_ok = True)
	
	num_splits:int = 10
	num_permutations:int

	if split_type == "MC":
		num_permutations = 25
	elif split_type == "LOO":
		num_permutations = 10
	
	icis = ['Atezo','Pembro','Ipi', 'Nivo']
	ds_string = "cri" if drug in icis else "ccle"

	
	rng = np.random.default_rng(seed)
	rstate = np.random.RandomState(seed)



	genesets:List[str] = ['cosmic','kegg','vogelstein','mdsig','auslander','EXPRESSION','FEATURES']

	expr_file = f"../expression/{ds_string}/{drug}/{tissue}/expression_full.csv"
	resp_file = f"../expression/{ds_string}/{drug}/{tissue}/response.csv"
	feat_file = f"../expression/{ds_string}/{drug}/{tissue}/features.csv"



	expression = pd.read_csv(expr_file)
	response = pd.read_csv(resp_file)
	features = pd.read_csv(feat_file)

	full_results = []
	
	for gs in tqdm.tqdm(genesets,leave= False):
		if gs =='EXPRESSION':
			
			genes = utils.fetch_union_genesets()
			keep_cols = [x for x in expression.columns if x in genes]
			X_full = np.log2(expression[keep_cols].values+1)
		
		elif gs=='FEATURES':
			
			X_full = features[features.columns[1:]].values
		else:
			
			genes = utils.fetch_geneset(gs)
			keep_cols = [x for x in expression.columns if x in genes]
			X_full = np.log2(expression[keep_cols].values+1)
			
		y_full = response['Response'].values


		# X_full, y_full are the original unpermuted or subsampled datasets
		
		results = pu.run_perm_test(
			X_full, 
			y_full, 
			num_permutations,
			test_type,
			split_type,
			model,
			num_splits,
			train_pct,
			rng,
			rstate)
		

		
		results['geneset'] = gs
		full_results.append(results)
	results = pd.concat(full_results,axis=0)
	fname = f"{res_dir}/{split_type}.csv"
	results.to_csv(fname, index=False)

if __name__ == '__main__':
	main()
		
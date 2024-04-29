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

				

def main():
	# pick a tissue and a drug for ICI only at present
	# repeat the following for B balanced datasets:
		# compute LOOCV for the classifier on the base dataset
		# for k = 100 permutations of the labels perform the LOOCV
		# store accuracy

	parser = argparse.ArgumentParser("model specific bake-off")
	parser.add_argument('--drug',help = "drug being tested")
	parser.add_argument('--settings', help = "tissue and model setting of form 'Model.Tissue'")
	

	args = vars(parser.parse_args())

	drug:str = args['drug']
	settings = args['settings'].split(".")
	model,tissue = settings[0],settings[1]
	
	

	num_balanced_datasets: int = 10
	num_permutations:int = 100
	seed:int = 1234

	
	drug: str = 'Atezo'
	tissue:str = 'BLCA'
	model_name = 'LogisticRegression'
	
	model, param_grid = mu.make_model_and_param_grid(model_name)	
	
	rng = np.random.default_rng(seed)
	rstate = np.random.RandomState(seed)
	loo = LeaveOneOut()

	results = defaultdict(list)


	expr_file = f"../expression/cri/{drug}/{tissue}/expression_full.csv"
	resp_file = f"../expression/cri/{drug}/{tissue}/response.csv"
	feat_file = f"../expression/cri/{drug}/{tissue}/features.csv"



	expression = pd.read_csv(expr_file)
	response = pd.read_csv(resp_file)
	features = pd.read_csv(feat_file)


	for gs in tqdm.tqdm(genesets,leave= False):
		if gs != 'FEATURES':
			
			with open("../genesets/{g}.txt".format(g=gs),"r") as istream:
				lines = istream.readlines()
			lines = [x.rstrip() for x in lines]


			keep_cols = [x for x in expression.columns if x in lines]
			X_ = np.log2(expression[keep_cols].values+1)
		
		else:
			X_ = features[features.columns[1:]].values
		
		Y_ = response['Response'].values
	

		for ds_id in tqdm.tqdm(range(num_balanced_datasets),leave=False):
			resp_idxs = np.where(Y_==1)[0]
			num_resp = len(resp_idxs)
			nonresp_idxs = np.where(Y_==0)[0]
			nonresp_samples = rng.choice(nonresp_idxs,num_resp, replace=False)
			

			ds_idxs = np.concatenate((resp_idxs,nonresp_samples))
			
			X = X_[ds_idxs,:]
			y = Y_[ds_idxs]


			acc, roc_auc = run_and_report_loocv(X,y, model, param_grid,loo)
			
			results['drug'].append(drug)
			results['tissue'].append(tissue)
			results['geneset'].append(gs)
			results['balanced_id'].append(ds_id)
			results['data'].append('original')
			results['accuracy'].append(acc)
			results['roc_auc'].append(roc_auc)
			results['test_type'].append(test_type)
			

			for j in tqdm.tqdm(range(num_permutations),leave=False):
				y = rng.permutation(Y_[ds_idxs])
				acc, roc_auc = run_and_report_loocv(X,y, model, param_grid,loo)
				results['drug'].append(drug)
				results['tissue'].append(tissue)
				results['geneset'].append(gs)
				results['balanced_id'].append(ds_id)
				results['data'].append(f"permututation {j}")
				results['accuracy'].append(acc)
				results['roc_auc'].append(roc_auc)
				results['test_type'].append(test_type)

					
				
	results = pd.DataFrame(results)
	results.to_csv("../results/{d}/{m}_loo_balanced_permutation.csv".format(d=drug, m = model_name))

				
				


if __name__ == '__main__':
	main()



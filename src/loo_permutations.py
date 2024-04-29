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


def run_and_report_loocv(X, y, model, param_grid, loo):
	binary_predictions = []
	probability_predictions = [] 
	true_values = []
	for i, (train_index, test_index) in enumerate(loo.split(X)):
		X_train, y_train = X[train_index,:], y[train_index]
		X_test, y_test = X[test_index,:], y[test_index]
		
		clf = GridSearchCV(model, param_grid)
		clf.fit(X_train,y_train)
		pred_bin = clf.predict(X_test)
		pred_prob = clf.predict_proba(X_test)

		binary_predictions.append(pred_bin[0])
		probability_predictions.append(pred_prob[0][0])
		true_values.append(y_test[0])
	acc = accuracy_score(true_values, binary_predictions)
	roc_auc = roc_auc_score(true_values,probability_predictions)
	return acc, roc_auc
	
				

def main():
	# pick a tissue and a drug for ICI only at present
	# repeat the following for B balanced datasets:
		# compute LOOCV for the classifier on the base dataset
		# for k = 100 permutations of the labels perform the LOOCV
		# store accuracy
	genesets:List[str] = ['LINCS','COSMIC','FEATURES']
	
	num_permutations:int = 100
	

	#if drug in ['Atezo','Pembro','Ipi','Nivo', 'Ipi+Pembro']:
	drug: str = 'Atezo'
	tissues:List[str] = ['KIRC','BLCA','PANCAN'] 

	seed:int = 1234
	model_name = 'LogisticRegression'
	model, param_grid = mu.make_model_and_param_grid(model_name)	

	
	rng = np.random.default_rng(seed)
	rstate = np.random.RandomState(seed)
	loo = LeaveOneOut()

	results = defaultdict(list)
	for tissue in tissues:

		expression = pd.read_csv("../expression/cri/{d}/{g}/expression_full.csv".format(d=drug,g=tissue))
		response = pd.read_csv("../expression/cri/{d}/{g}/response.csv".format( d=drug,g= tissue))
		features = pd.read_csv("../expression/cri/{d}/{g}/features.csv".format( d=drug,g = tissue))

		for gs in genesets:
			if gs != 'FEATURES':
				
				with open("../genesets/{g}.txt".format(g=gs),"r") as istream:
					lines = istream.readlines()
				lines = [x.rstrip() for x in lines]


				keep_cols = [x for x in expression.columns if x in lines]
				X = np.log2(expression[keep_cols].values+1)
			
			else:
				X = features[features.columns[1:]].values
			
			y_ = response['Response'].values
	
			acc, roc_auc = run_and_report_loocv(X,y_, model, param_grid,loo)
			
			results['drug'].append(drug)
			results['tissue'].append(tissue)
			results['geneset'].append(gs)
			results['data'].append('original')
			results['accuracy'].append(acc)
			results['roc_auc'].append(roc_auc)
			results['test_type'].append(test_type)
			

			for j in range(num_permutations):
				y = rng.permutation(y_)
				acc, roc_auc = run_and_report_loocv(X,y, model, param_grid,loo)
				results['drug'].append(drug)
				results['tissue'].append(tissue)
				results['geneset'].append(gs)
				results['data'].append(f"permututation {j}")
				results['accuracy'].append(acc)
				results['roc_auc'].append(roc_auc)
				results['test_type'].append(test_type)

					
				
	results = pd.DataFrame(results)
	results.to_csv("../results/{d}/{m}_loo_permutation.csv".format(d=drug, m = model_name))

				
				


if __name__ == '__main__':
	main()



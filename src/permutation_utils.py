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



def run_and_report_loocv(X, y, model, param_grid):
	binary_predictions = []
	probability_predictions = [] 
	true_values = []
	loo = LeaveOneOut()
	
	for i, (train_index, test_index) in tqdm.tqdm(enumerate(loo.split(X)),leave=False, total = X.shape[0]):
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
	bal_acc = balanced_accuracy_score(true_values,binary_predictions)
	return acc, roc_auc, bal_acc
	


def make_permuted_dataset(X_, y_, rng, test_type = 1): 
	assert test_type in [1,2], "test must be either 1 or 2"
	
	if test_type ==1:
		y = rng.permutation(y_)
		X = X_.copy()
	else:
		pass 

	return X,y




def make_balanced_dataset(X_full:np.array, y_full:np.array, rng):
	
	resp_idxs = np.where(y_full==1)[0]
	num_resp = len(resp_idxs)
	nonresp_idxs = np.where(y_full==0)[0]
	nonresp_samples = rng.choice(nonresp_idxs,num_resp, replace=False)


	ds_idxs = np.concatenate((resp_idxs,nonresp_samples))

	X = X_full[ds_idxs,:].copy()
	y = y_full[ds_idxs].copy()

	return X,y

def run_balanced_test(
	X_full:np.ndarray, 
	y_full:np.array,
	num_balanced_datasets:int,
	num_permutations:int,
	test_type:int, 
	model_name:str,
	rng)->pd.DataFrame:
	
	results = defaultdict(list)

	model, param_grid = mu.make_model_and_param_grid(model_name)

	for ds_id in tqdm.tqdm(range(num_balanced_datasets),leave=False):
		X_,y_ = make_balanced_dataset(X_full,y_full,rng)
		acc, roc_auc, bal_acc = run_and_report_loocv(X_,y_,model,param_grid)
		results['balanced_id'].append(ds_id)
		results['data'].append('original')
		results['accuracy'].append(acc)
		results['roc_auc'].append(roc_auc)
		results['balanced_accuarcy'].append(bal_acc)

		for perm in tqdm.tqdm(range(num_permutations),leave=False):
			X,y = make_permuted_dataset(X_, y_, rng)
			acc, roc_auc,bal_acc = run_and_report_loocv(X,y, model, param_grid)
			results['balanced_id'].append(ds_id)
			results['data'].append(f"permutation {perm}")
			results['accuracy'].append(acc)
			results['roc_auc'].append(roc_auc)
			results['balanced_accuarcy'].append(bal_acc)

	results = pd.DataFrame(results)
	return results


def run_full_test(
	X_full:np.ndarray, 
	y_full:np.array,
	num_permutations:int,
	test_type:int, 
	model_name:str,
	rng)->pd.DataFrame:
	

	results = defaultdict(list)

	model, param_grid = mu.make_model_and_param_grid(model_name)

	acc, roc_auc, bal_acc = run_and_report_loocv(X_full,y_full,model,param_grid)
	
	results['data'].append('original')
	results['accuracy'].append(acc)
	results['roc_auc'].append(roc_auc)
	results['balanced_accuracy'].append(bal_acc)

	for perm in tqdm.tqdm(range(num_permutations),leave=False):
		X,y = make_permuted_dataset(X_full, y_full, rng)
		acc, roc_auc,bal_acc = run_and_report_loocv(X,y, model, param_grid)
		results['data'].append(f"permutation {perm}")
		results['accuracy'].append(acc)
		results['roc_auc'].append(roc_auc)
		results['balanced_accuracy'].append(bal_acc)


	results = pd.DataFrame(results)
	return results
import sys
import numpy as np
import tqdm 
from typing import List,Dict
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

def run_and_report_monte_carlo(
	X:np.ndarray,
	y:np.ndarray, 
	model,
	param_grid:Dict,
	num_splits:int,
	train_pct:float,
	rstate):
	
	
	accs, roc_aucs,bal_accs = [],[],[]
	
	for i in tqdm.tqdm(range(num_splits),leave=False):					
		X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_pct, random_state=rstate,stratify=y)
		clf = GridSearchCV(model, param_grid)
		clf.fit(X_train,y_train)
		pred_bins = clf.predict(X_test)
		pred_probs = clf.predict_proba(X_test)
		
		acc = accuracy_score(y_test, pred_bins)
		roc_auc = roc_auc_score(y_test,pred_probs[:,1])
		bal_acc = balanced_accuracy_score(y_test, pred_bins)
		
		accs.append(acc)
		bal_accs.append(bal_acc)
		roc_aucs.append(roc_auc)


	return accs, roc_aucs, bal_accs

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
	
	return [acc], [roc_auc], [bal_acc]
	


def make_permuted_dataset(X_, y_, rng, test_type = 1): 
	assert test_type in [1,2], "test must be either 1 or 2"
	
	if test_type ==1:
		y = rng.permutation(y_)
		X = X_.copy()
	elif test_type==2:
		raise NotImplementedError
# 		for c in pd.unique(y_):
# 			resp_idxs = np.where(y_==c)[0]
# # 		num_resp = len(resp_idxs)
##			stack columns


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
		
def run_perm_test(
	X_full:np.ndarray, 
	y_full:np.array,
	num_permutations:int,
	test_type:int, 
	split_type:str,
	model_name:str,
	num_splits:int,
	train_pct:float,
	rng,
	rstate)->pd.DataFrame:
	

	results = defaultdict(list)

	model, param_grid = mu.make_model_and_param_grid(model_name)

	if split_type=='LOO':
		accs, roc_aucs, bal_accs = run_and_report_loocv(X_full,y_full,model,param_grid)
		results['data'].extend(1*['original'])
		
	elif split_type =="MC":
		accs, roc_aucs, bal_accs = run_and_report_monte_carlo(X_full,y_full, model, param_grid, num_splits,train_pct,rstate)
		results['data'].extend(num_splits*['original'])
		
	results['accuracy'].extend(accs)
	results['roc_auc'].extend(roc_aucs)
	results['balanced_accuracy'].extend(bal_accs)



	for perm in tqdm.tqdm(range(num_permutations),leave=False):
		X,y = make_permuted_dataset(X_full, y_full, rng)
		if split_type=="LOO":
			acc, roc_auc,bal_acc = run_and_report_loocv(X,y, model, param_grid)
		elif split_type == "MC":
			acc, roc_auc,bal_acc = run_and_report_monte_carlo(X,y, model, param_grid, 1,train_pct,rstate)
		results['data'].append(f"permutation {perm}")
		results['accuracy'].extend(acc)
		results['roc_auc'].extend(roc_auc)
		results['balanced_accuracy'].extend(bal_acc)
	results = pd.DataFrame(results)
	return results


	results = pd.DataFrame(results)
	return results
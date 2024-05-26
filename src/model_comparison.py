import sys
import tqdm 
import time 
from sklearn.model_selection import GridSearchCV
import pandas as pd
from typing import Dict, List
import model_utils as mu
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score
import utils
from collections import defaultdict 
import numpy as np 
import argparse





def main():
	parser = argparse.ArgumentParser("model specific bake-off")
	parser.add_argument('-drug',help = "drug being tested")
	parser.add_argument('-balance', help = "balance weights?")
	

	args = vars(parser.parse_args())

	drug:str = args['drug']
	
	bal:int = int(args['balance'])
	
	balanced_weights:bool = bal == 1
	bal_string = 'balanced' if balanced_weights else 'unbalanced'
	
	num_iters:int = 25
	test_size: float = 0.2
	rng_seed:int = 123450
	rng = np.random.RandomState(rng_seed)
	

	tissues = utils.DRUG_TISSUE_MAP[drug]
	gene_sets:List[str] = ['cosmic1','cosmic2','cosmic','kegg','vogelstein','mdsig','auslander']

	feature_sets:List[str] = ['TIDE_VALUES']

	results = defaultdict(list)
	
	covariate_sets = gene_sets + feature_sets
	# covariate_sets = feature_sets
	for tissue in tqdm.tqdm(tissues):
		
		expression_file = f"../expression/cri/{drug}/{tissue}/expression_full.csv"
		response_file = f"../expression/cri/{drug}/{tissue}/response.csv"
		feature_file = f"../expression/cri/{drug}/{tissue}/features.csv"
		
		expression = pd.read_csv(expression_file)
		response = pd.read_csv(response_file)
		features = pd.read_csv(feature_file)
		
		
		expression = expression.merge(response,on='Run_ID')
		features = features.merge(response, on='Run_ID')
		
		for model_name in ["LogisticRegression","RandomForest"]:
			
			model, param_grid = mu.make_model_and_param_grid(model_name,balanced_weights)
			clf = GridSearchCV(model, param_grid)
			
			for covariates in tqdm.tqdm(covariate_sets,leave=False):
				if covariates in gene_sets:
					gene_names = utils.fetch_geneset(geneset_name = covariates)

					common_genes = [x for x in expression.columns if x in gene_names]

					X = np.log2(expression[common_genes].values+1)
					y = expression['Response'].values

				elif covariates in feature_sets:
					if covariates == 'TIDE_VALUES':
						feature_names = ['TIDE','TIDE_CTL_Flag','TIDE_Dysfunction', 'TIDE_Exclusion', 'TIDE_CTL']
					else:
						feature_names = covariates

					X = features[feature_names].values
					y = features['Response'].values

				for i in tqdm.tqdm(range(num_iters),leave=False):
					

					X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=rng,stratify=y)
					
					clf.fit(X_train,y_train)
					
					pred_test  = clf.predict(X_test)
					pred_prob_test = clf.predict_proba(X_test)

					acc = accuracy_score(y_test,clf.predict(X_test))
					bal_acc = balanced_accuracy_score(y_test, clf.predict(X_test))
					roc_auc = roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
					results['Drug'].append(drug)
					results['Tissue'].append(tissue)
					results['Model'].append(model_name)
					results['Features'].append(covariates)
					results['Test Accuracy'].append(acc)
					results['Test Balanced Accuracy'].append(bal_acc)
					results['Test ROC AUC'].append(roc_auc)
					
	results = pd.DataFrame(results)
	results.to_csv(f"../results/{drug}/feature_bakeoff_{bal_string}.csv",index = False)




	
	
	
				
if __name__ == '__main__':
	main()
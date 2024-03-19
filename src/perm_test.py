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



parser = argparse.ArgumentParser("model specific bake-off")
parser.add_argument('--drug',help = "drug being tested")
parser.add_argument('--model', help = "Model")
parser.add_argument('--niter', help = 'number of iters')


def main():
	args = vars(parser.parse_args())
	drug:str = args['drug']
	genesets:List[str] = ['LINCS','COSMIC','FEATURES']
	
	if drug in ['Atezo','Pembro','Ipi','Nivo', 'Ipi+Pembro']:
		ds = "cri"
	else:
		ds = "ccle"
	
	if ds == 'ccle':
		genesets = genesets[:2]
		group = "BROAD"
	else:
		group = "PANCAN"

	seed:int = 1234
	model_name = args['model']
	num_iters = int(args['niter'])
	num_perms:int = 50
	model, param_grid = mu.make_model_and_param_grid(model_name)	

	rng = np.random.default_rng(seed)
	rstate = np.random.RandomState(seed)

	print("../expression/{ds}/{d}/{g}/expression_full.csv".format(ds=ds, d=drug,g=group))
	expression = pd.read_csv("../expression/{ds}/{d}/{g}/expression_full.csv".format(ds=ds, d=drug,g=group))
	response = pd.read_csv("../expression/{ds}/{d}/{g}/response.csv".format(ds=ds, d=drug,g= group))
	features = pd.read_csv("../expression/{ds}/{d}/{g}/features.csv".format(ds=ds, d=drug,g = group))
	


	
	results = defaultdict(list)
	os.makedirs("../results/{d}/".format(d=drug),exist_ok = True)
	for gs in genesets:
		if gs != 'FEATURES':
			with open("../genesets/{g}.txt".format(g=gs),"r") as istream:
				lines = istream.readlines()
			lines = [x.rstrip() for x in lines]


			keep_cols = [x for x in expression.columns if x in lines]
			X = np.log2(expression[keep_cols].values+1)
			
		else:
			X = features[features.columns[1:]].values
		y = response['Response'].values
		for stratify in [True,False]:
			for perm in tqdm.tqdm(range(num_perms)):

				for i in tqdm.tqdm(range(num_iters)):
					if stratify:
						X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify = y, random_state = rstate)
					else:
						X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state = rstate)
					
					clf = GridSearchCV(model, param_grid)
					clf.fit(X_train,y_train)
					preds = clf.predict(X_test)
					results['iter'].append(i)
					results['perm'].append(perm)
					results['geneset'].append(gs)
					results['stratified'].append(stratify)
					results['model'].append(model_name)
					results['test accuracy'].append(np.round(accuracy_score(y_test,preds),3))
					results['test balanced accuracy'].append(np.round(balanced_accuracy_score(y_test,preds),3))
					if model_name in ['RBF_SVC',"Poly_SVC",'Linear_SVC']:
						results['Test_ROC'] = -1
					else:
						prob_preds = clf.predict_proba(X_test)
					results['Test_ROC'].append(np.round(roc_auc_score(y_test,prob_preds[:,1]),3))
	results = pd.DataFrame(results)
	
	results.to_csv("../results/{d}/{m}_perm.csv".format(d=drug, m = model_name))

if __name__ == '__main__':
	main()
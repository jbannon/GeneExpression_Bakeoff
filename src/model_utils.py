import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



def make_model_and_param_grid(
	model_name:str,
	balanced_weights:bool = False
	):
	
	bal_status = 'balanced' if balanced_weights else None
	
	if model_name == "RandomForest":
	
		model = Pipeline([('clf',RandomForestClassifier(class_weight = bal_status))])
		param_grid = {
			'clf__n_estimators':[2**j for j in range(6)],
			'clf__max_depth':[2**j for j in range(6)],
			'clf__min_samples_leaf':[3,5,7,9,11,13,15,20]
		}
		# param_grid = {
		# 	'clf__n_estimators':[2**j for j in range(2)],
		# 	'clf__max_depth':[2**j for j in range(2)],
		# 	'clf__min_samples_leaf':[1,2]
		# }
	
	elif model_name == "LogisticRegression":
		
		model = Pipeline([('scale',StandardScaler()),
			('clf',LogisticRegression(max_iter = 10000, solver = 'liblinear', class_weight = bal_status))])
		param_grid = {
			'clf__penalty':['l1','l2'],
			'clf__C':[10**j for j in np.linspace(-5,0.5,50)]
		}
	
	elif model_name == "RBF_SVC":
	
		model = Pipeline([('scale',StandardScaler()),
			('clf',SVC(class_weight = bal_status))])
		param_grid = {
			'clf__C':[10**j for j in np.linspace(-5,0.5,20)],
			'clf__gamma':['scale','auto']+[x for x in np.linspace(2,200,50)]
		}
	
	elif model_name == "Poly_SVC":
		model = Pipeline([('scale',StandardScaler()),
			('clf',SVC(kernel = 'poly',class_weight = bal_status))])
		param_grid = {
			'clf__C':[10**j for j in np.linspace(-5,0.5,20)],
			'clf__degree':np.arange(1,6)
		}
	
	elif model_name == 'Linear_SVC':
	
		model = Pipeline([('scale',StandardScaler()),
			('clf',LinearSVC(max_iter = 1000000, class_weight = bal_status,dual ='auto'))])
		param_grid = {
			'clf__penalty':['l1','l2'],
			'clf__C':[10**j for j in np.linspace(-5,0.5,20)],
		}
	elif model_name == 'GradientBoosting':
		model = Pipeline([('scale',StandardScaler()),
			('clf',GradientBoostingClassifier())])
		param_grid = {
			'clf__loss':['log_loss','exponential'],
			'clf__n_estimators':[2**j for j in range(9)],
			'clf__max_depth':[2**j for j in range(4)],
			'clf__max_features':['sqrt','log2'],
			'clf__min_samples_leaf':[1,2,3,4,5],
			'clf__learning_rate':[10**j for j in np.linspace(-5, 0,20)]
		}

	return model, param_grid



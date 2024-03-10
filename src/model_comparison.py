import time 
from sklearn.model_selection import GridSearchCV
import pandas as pd
import model_utils as mu
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score





mn = "Poly_SVC"
num_iters = 100

with open("../genesets/LINCS.txt","r") as istream:
	lines = istream.readlines()


lines = [x.rstrip() for x in lines]

expr = pd.read_csv("../expression/cri/Atezo/PANCAN/expression_full.csv")
respr = pd.read_csv("../expression/cri/Atezo/PANCAN/response.csv")
# feat = pd.read_csv("../expression/cri/Atezo/PANCAN/features.csv")

keep_cols = [x for x in expr.columns if x in lines]
expr = expr[['Run_ID'] + keep_cols]


X = expr[expr.columns[1:]].values
y = respr['Response'].values

model, param_grid = mu.make_model_and_param_grid(mn)
def main():
	
for i in range(num_iters):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
		random_state=42, stratify = y)
	
	clf = GridSearchCV(model, param_grid)
	clf.fit(X_train,y_train)
	
	
	
	print(accuracy_score(y_test,clf.predict(X_test)))
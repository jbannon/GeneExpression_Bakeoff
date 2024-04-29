import sys
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns




drug = "erlotinib"
tissue = "BROAD"
ds = "unbalanced"

df = pd.read_csv(f"../results/{drug}/{tissue}/perm_tests/{ds}.csv")
if ds == 'unbalanced':
	for gs in pd.unique(df['geneset']):
		temp = df[df['geneset']==gs]
		num_perm = temp[temp['data']!='original'].shape[0]
		base_acc = -1
		perm_accs = []
		for idx, row in temp.iterrows():
			if row['data']=='original':
				base_acc = row['accuracy']
			else:
				perm_accs.append(row['accuracy'])

		num_better = len([x for x in perm_accs if x>=base_acc])
		p_value = 1.0*num_better/(num_perm+1)
		print("\n")
		print(f"Drug:\t{drug}")
		print(f"Tissue:\t{tissue}")
		print(f"Balanced?:\t{ds}")
		print(f"Covariates:\t {gs}")
		print(f"Original Dataset Accuracy:\t{np.round(base_acc,2)}")
		print(f"Average Permuted Accuracy:\t{np.round(np.mean(perm_accs),2)}")
		print(f"P value:\t{np.round(p_value,3)}")
else:
	for gs in pd.unique(df['geneset']):
		base_accs = []
		avg_perms = []
		pvals = []
		for bid in pd.unique(df['balanced_id']):
			temp = df[(df['balanced_id']==bid) & (df['geneset']==gs)]
			num_perm = temp[temp['data']!='original'].shape[0]
			perms = []
			base_acc = -1
			for idx, row in temp.iterrows():
				if row['data']=='original':
					base_accs.append(row['accuracy'])
					base_acc = row['accuracy']
				else:
					perms.append(row['accuracy'])

			avg_perms.append(np.mean(perms))
			pval = 1.*len([x for x in perms if x>=base_acc])/(num_perm+1)
			pvals.append(pval)
		print("\n")
		print(f"Drug:\t{drug}")
		print(f"Tissue:\t{tissue}")
		print(f"Balanced?:\t{ds}")
		print(f"Covariates:\t {gs}")
		print(f"Original Dataset Accuracy:\t{np.round(np.mean(base_accs),2)}")
		print(f"Average Permuted Accuracy:\t{np.round(np.mean(avg_perms),2)}")
		print(f"P value:\t{np.round(np.mean(pvals),3)}")

		
		

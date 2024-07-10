import sys
from scipy.stats import mannwhitneyu
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import os

import statsmodels.stats.multitest as mt




drug_tissue_map = {'Pembro':['STAD','SKCM'], 'Nivo':['SKCM','KIRC'], 'Atezo':['KIRC','BLCA']}
gs_map = {'cosmic':'COSMIC', 'kegg':'KEGG',
	'vogelstein':'CGL', 'mdsig':'MDSIG-DB','auslander':'IMPRES','EXPRESSION':'ALL', 'FEATURES': 'IMM. FEAT'}

model_name_map = {'LogisticRegression': 'Logistic Regression', 'RandomForest':'Random Forest'}

metric_map = {'accuracy':'Accuracy', 'balanced_accuracy': 'Balanced Accuracy', 'roc_auc': 'Area Under ROC Curve'}

for drug in ['Pembro','Nivo','Atezo']:
	tissues = drug_tissue_map[drug]
	results = defaultdict(list)
	for tissue in tissues:
		os.makedirs(f"../figs/{drug}/{tissue}/",exist_ok=True)
		for model in ["LogisticRegression","RandomForest"]:
			data_file = f"../results/{drug}/{tissue}/perm_tests/{model}/Type_A/MC.csv"
			df = pd.read_csv(data_file)
			
			df['Data Type'] = df['data'].apply(lambda x: "Permuted" if "permutation" in x else "Original")
			df['Covariates'] = df['geneset'].apply(lambda x: gs_map[x])

			for metric in ['accuracy','balanced_accuracy', 'roc_auc']:
				
				df_o = df[df['Data Type']=='Original']
				df_p = df[df['Data Type'] == 'Permuted']
				df_p2 = df_p[['data','accuracy','roc_auc','balanced_accuracy','Covariates']]
				df_p2 =df_p2.groupby(['data','Covariates']).mean()
				df_p2.reset_index(drop=False,inplace=True)
				
				col_order = ['data','accuracy','roc_auc','balanced_accuracy','Covariates']
				df_o = df_o[col_order]
				
				df_p2 = df_p2[col_order]
				
				df_2 = pd.concat([df_o,df_p2],axis=0)
				df_2['Data Type'] = df_2['data'].apply(lambda x: "Permuted" if "permutation" in x else "Original")
				
				
		

				ps = []
				for cov in pd.unique(df['Covariates']):
					
					
					temp = df_2[df_2['Covariates']==cov]
					x_org = temp[temp['Data Type']=='Original'][metric].values
					x_perm = temp[temp['Data Type']=='Permuted'][metric].values
					v, p = mannwhitneyu(x_org,x_perm)
					ps.append(p)
					p_str = "<0.005" if p<0.005 else p
					
					results['Drug'].append(drug)
					results['Feature Set'].append(cov)
					results['Tissue'].append(tissue)
					results['Model'].append(model_name_map[model])
					results['Metric'].append(metric_map[metric])
					results['P-str'].append(p_str)
					results['P'].append(p)
				adj_p = mt.multipletests(ps,method = 'fdr_bh')
				results['adj_p'].extend(list(adj_p[1]))
				

				ax = sns.boxplot(data=df, x = 'Covariates',y=metric,hue='Data Type')
				ax.set(title = f"{drug.title()}  {tissue}  {model_name_map[model]}",
					xlabel = 'Covariates',
					ylabel = metric_map[metric])
				plt.savefig(f"../figs/{drug}/{tissue}/{model}_{metric}.png",dpi=500)
				plt.close()
				
				ax = sns.boxplot(data=df_2, x = 'Covariates',y=metric,hue='Data Type')
				ax.set(title = f"{drug.title()}  {tissue}  {model_name_map[model]}",
					xlabel = 'Covariates',
					ylabel = metric_map[metric])		
				plt.savefig(f"../figs/{drug}/{tissue}/{model}_{metric}_averaged.png",dpi=500)
				plt.close()
			
		results  = pd.DataFrame(results)
		results.to_csv(f"../results/{drug}/{tissue}/p_values.csv",index=False)
		sys.exit(1)



import warnings
warnings.filterwarnings("ignore")
import sys
from scipy.stats import mannwhitneyu
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import os

import statsmodels.stats.multitest as mt




drug_tissue_map = {'Pembro':['STAD','SKCM'], 'Nivo':['SKCM','KIRC'], 'Atezo':['KIRC','BLCA'],'Ipi':['SKCM']}
gs_map = {'cosmic':'COSMIC', 'kegg':'KEGG',
	'vogelstein':'CGL', 'mdsig':'MDSIG-DB','auslander':'IMPRES','EXPRESSION':'ALL', 'FEATURES': 'IMM. FEAT'}

model_name_map = {'LogisticRegression': 'Logistic Regression', 'RandomForest':'Random Forest'}

metric_map = {'accuracy':'Accuracy', 'balanced_accuracy': 'Balanced Accuracy', 'roc_auc': 'Area Under ROC Curve'}

for drug in ['Nivo','Atezo','Pembro']:
	tissues = drug_tissue_map[drug]
	best_feats = defaultdict(list)
	top_two = defaultdict(list)
	for tissue in tissues:
		results = defaultdict(list)
		os.makedirs(f"../figs/{drug}/{tissue}/",exist_ok=True)
		for model in ["LogisticRegression","RandomForest"]:
			data_file = f"../results/{drug}/{tissue}/perm_tests/{model}/Type_A/MC.csv"
			df = pd.read_csv(data_file)
			
			df['Data Type'] = df['data'].apply(lambda x: "Permuted" if "permutation" in x else "Original")
			df['Covariates'] = df['geneset'].apply(lambda x: gs_map[x])
			
			for metric in ['accuracy','balanced_accuracy','roc_auc']:
				
				df_o = df[df['Data Type']=='Original']
				df_p = df[df['Data Type'] == 'Permuted']
				df_p2 = df_p[['data','accuracy','roc_auc','balanced_accuracy','Covariates']]
				df_p2 =df_p2.groupby(['data','Covariates']).mean()
				df_p2.reset_index(drop=False,inplace=True)
				
				col_order = ['data','accuracy','roc_auc','balanced_accuracy','Covariates']
				df_o = df_o[col_order]
				


				_df_o = df_o[['accuracy','roc_auc','balanced_accuracy','Covariates']].groupby(['Covariates']).mean()
				_df_o.sort_values(metric,ascending=False,inplace=True)
				
				top_two['Drug'].append(drug)
				top_two['Tissue'].append(tissue)
				top_two['Metric'].append(metric)
				top_two['Model'].append(model)
				top_two['Best Value'].append(_df_o[metric].values[0])
				top_two['Second Best Value'].append(_df_o[metric].values[1])
				top_two['Range'].append(_df_o[metric].values[0]-_df_o[metric].values[-1])
				top_two['Second Gap'].append(_df_o[metric].values[0]-_df_o[metric].values[1])
				bf = _df_o.index[np.argmax(_df_o[metric])]
				best_feats['Drug'].append(drug)
				best_feats['Metric'].append(metric)
				best_feats['Best Feature'].append(bf)
				best_feats['Model'].append(model)
				best_feats['Tissue'].append(tissue)
				best_feats['Mean Metric value'].append(_df_o.loc[bf,metric])


				
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
					if cov == bf:
						best_feats['p value'].append(p_str)
						# top_two['p value'].append(p_str)
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
				ax.set_title(f"{drug.title()}  {tissue}  {model_name_map[model]}", fontsize = 18)
				ax.set_xlabel('Covariates', fontsize = 18,labelpad=5)
				ax.set_ylabel(metric_map[metric],fontsize = 18)
				ax.tick_params('x',labelsize=15,labelrotation=30)
				ax.tick_params('y',labelsize=15)
				# sns.set(font_scale = 1.3)
				# ax.set(title = f"{drug.title()}  {tissue}  {model_name_map[model]}",
				#  	xlabel = 'Covariates',
				# 	ylabel = metric_map[metric])
				legend = ax.legend()
				legend.remove()
				plt.tight_layout()
				plt.savefig(f"../figs/{drug}/{tissue}/{model}_{metric}_averaged.png",dpi=500)
				plt.close()
			
		results  = pd.DataFrame(results)
		results.to_csv(f"../results/{drug}/{tissue}/p_values.csv",index=False)
	feat_res = pd.DataFrame(best_feats)
	feat_res.to_csv(f"../results/{drug}/best_features.csv",index = False)
	top_two = pd.DataFrame(top_two)
	
	top_two.to_csv(f"../results/{drug}/gaps.csv",index = False)



def check_x(x):
	if isinstance(x,str):
		if x =="<0.005":
			return 1
		elif float(x)<=0.1:
			return 1
	return 0
	
df_list = []
for drug in ['Pembro','Nivo','Atezo']:
	df = pd.read_csv(f"../results/{drug}/best_features.csv")
	df_list.append(df)


df = pd.concat(df_list, axis =0)

df['p_status'] = df['p value'].apply(lambda x: check_x(x))

df_ = df[df['p_status']==1]

feature = pd.DataFrame(df_['Best Feature'].value_counts())
ax = sns.catplot(feature,x='Best Feature',y='count',kind='bar')
ax.set(xlabel = 'Feature',ylabel = "Count",title = "Best Performing Feature Sets")
plt.tight_layout()
plt.savefig("../figs/feat_counts.png",dpi=300)
plt.close()

# feature.reset_index(inplace = True)
# print(feature)



model = pd.DataFrame(df['Model'].value_counts())
ax = sns.catplot(model,x='Model',y='count',kind='bar')
ax.set(xlabel = 'Model',ylabel = "Count")
plt.tight_layout()
plt.savefig("../figs/model_counts.png",dpi=300)
plt.close()

for model in pd.unique(df['Model']):
	temp = df_[df_['Model']==model]
	temp = pd.DataFrame(temp['Best Feature'].value_counts())
	ax = sns.catplot(temp,x='Best Feature',y='count',kind='bar')
	ax.set(xlabel = 'Feature',ylabel = "Count",title = f"Best Performing Feature Sets in {model_name_map[model]}")
	# legend = ax.legend()
	# legend.remove()
	plt.tight_layout()
	plt.savefig(f"../figs/{model}_feat_counts.png",dpi=300)
	plt.close()

for metric in pd.unique(df['Metric']):
	temp = df_[df_['Metric']==metric]
	temp = pd.DataFrame(temp['Best Feature'].value_counts())
	ax = sns.catplot(temp,x='Best Feature',y='count',kind='bar')
	ax.set(xlabel = 'Feature',ylabel = "Count",title = f"Best Performing Feature Sets For {metric_map[metric]}")
	# legend = ax.legend()
	# legend.remove()
	plt.tight_layout()

	plt.savefig(f"../figs/{metric}_feat_counts.png",dpi=300)
	plt.close()




df_list = []
for drug in ['Pembro','Nivo','Atezo']:
	df = pd.read_csv(f"../results/{drug}/gaps.csv")
	df_list.append(df)
df = pd.concat(df_list, axis =0)


print("\n")
print("\n")
for metric in pd.unique(df['Metric']):
	temp = df[df['Metric']==metric]
	print("\n---------")
	print(f"For {metric} the average best/worst gap = {np.mean(temp['Range'].values)}")
	print("---------\n")
	lr_mean = np.mean(temp[temp['Model']=='LogisticRegression']['Range'].values)
	rf_mean = np.mean(temp[temp['Model']=='RandomForest']['Range'].values)
	ax = sns.displot(data = temp, x = 'Range',kind='kde',hue='Model',fill=True,color = {'LogisticRegression':'blue','RandomForest':'orange'})
	
	ax.fig.suptitle(f"{metric_map[metric]}", fontsize = 18)
	ax.set_xlabels(f"Gap Between Best and Worst Feature Set", fontsize = 18)
	ax.set_ylabels("Density",fontsize = 18)
	ax.tick_params('x',labelsize=15)
	ax.tick_params('y',labelsize=15)
	print(lr_mean)
	print(rf_mean)
	if np.abs(lr_mean-rf_mean)<=0.001:
		plt.axvline(0.5*(lr_mean+rf_mean),0,1,color='black')
	else:
		plt.axvline(lr_mean, 0,1,color='blue')
		plt.axvline(rf_mean, 0,1,color='orange')
	# legend = ax.legend()
	# legend.remove()
	ax._legend.remove()
	plt.tight_layout()
	plt.savefig(f"../figs/{metric}_gap.png",dpi=300)
	plt.close()


	print(f"For {metric} the average best second best gap = {np.mean(temp['Second Gap'].values)}")

	lr_mean = np.mean(temp[temp['Model']=='LogisticRegression']['Second Gap'].values)
	rf_mean = np.mean(temp[temp['Model']=='RandomForest']['Second Gap'].values)
	ax = sns.displot(data = temp, x = 'Second Gap',kind='kde',hue='Model',fill=True,color = {'LogisticRegression':'blue','RandomForest':'orange'})
	# ax.set(xlabel = f"Gap Between Best and Second Best Feature Set", ylabel = "Density", title = f"{metric_map[metric]}")
	ax.fig.suptitle(f"{metric_map[metric]}", fontsize = 18)
	ax.set_xlabels(f"Gap Between Best and Second Best Feature Set", fontsize = 18)
	ax.set_ylabels("Density",fontsize = 18)
	ax.tick_params('x',labelsize=15)
	ax.tick_params('y',labelsize=15)
	if np.abs(lr_mean-rf_mean)<=0.001:
		plt.axvline(0.5*(lr_mean+rf_mean),0,1,color='black')
	else:
		plt.axvline(lr_mean, 0,1,color='blue')
		plt.axvline(rf_mean, 0,1,color='orange')
	# legend = ax.legend()
	# legend.remove()
	ax._legend.remove()
	plt.legend([],[], frameon=False)
	plt.tight_layout()
	plt.savefig(f"../figs/{metric}_second gap.png",dpi=300)
	plt.close()
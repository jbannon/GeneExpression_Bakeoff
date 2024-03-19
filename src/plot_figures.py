import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import os


for drug in ['Pembro', 'Nivo', 'erlotinib', 'crizotinib', 'sorafenib', 'sunitinib']:

	os.makedirs("../figs/{d}".format(d=drug), exist_ok = True)
	path = "../results/{d}/".format(d=drug)


	model_files = [f for f in os.listdir(path) if not f.startswith(".")]

	df_list = []
	for mf in model_files:
		model = mf.split(".")[0]
		df = pd.read_csv("{p}{m}".format(p=path,m=mf),index_col = 0)
		ax = sns.violinplot(df, x = "geneset", y = "test accuracy")
		plt.title("{d} {m}".format(d=drug,m= model))
		plt.savefig("../figs/{d}/{d}_{m}.png".format(d=drug, m= model))
		plt.close()
		df['model'] = model
		df_list.append(df)

	remap = {"LINCS":"LINCS","COSMIC":"COSMIC","FEATURES":"Immune feats."}
	all_data = pd.concat(df_list,axis=0)
	all_data['geneset'] = all_data['geneset'].apply(lambda x: remap[x])
	ax = sns.violinplot(all_data, x = "model", y = "test accuracy", hue = "geneset")
	plt.title("{d} Full Comparison".format(d=drug))
	sns.move_legend(
	    ax, "lower center",
	    bbox_to_anchor=(0.55, 0), ncol=3, title=None, frameon=False,
	)
	plt.savefig("../figs/{d}/{d}_full.png".format(d=drug))
	plt.close()
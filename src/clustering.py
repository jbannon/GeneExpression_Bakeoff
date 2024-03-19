import seaborn as sns 
import matplotlib.pyplot as plt
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
from sklearn.manifold import SpectralEmbedding, MDS, LocallyLinearEmbedding
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline




drug = 'Atezo'
ds  = 'cri'
group = 'KIRC'
genesets:List[str] = ['LINCS','COSMIC','FEATURES']


for group in ['KIRC','BLCA','PANCAN']:
	expression = pd.read_csv("../expression/{ds}/{d}/{g}/expression_full.csv".format(ds=ds, d=drug,g=group))
	response = pd.read_csv("../expression/{ds}/{d}/{g}/response.csv".format(ds=ds, d=drug,g= group))
	features = pd.read_csv("../expression/{ds}/{d}/{g}/features.csv".format(ds=ds, d=drug,g = group))
	y = response['Response'].values
	for gs in genesets:
		if gs != 'FEATURES':
			with open("../genesets/{g}.txt".format(g=gs),"r") as istream:
				lines = istream.readlines()
			lines = [x.rstrip() for x in lines]
			keep_cols = [x for x in expression.columns if x in lines]
			X = np.log2(expression[keep_cols].values+1)
		else:
			X = features[features.columns[1:]].values

		emb = SpectralEmbedding(n_components = 2)
		X_emb = emb.fit_transform(X)
		resdf = pd.DataFrame({'coord1':X_emb[:,0], 'coord2':X_emb[:,1],'Class':y})
		resdf['Class'] = resdf['Class'].apply(lambda x: "Responder" if x ==1 else "Non-Responder")
		ax = sns.scatterplot(resdf, x = 'coord1', y = 'coord2', hue = 'Class')
		plt.title("{d} {g} {gs} Laplacian Eigenmaps".format(d=drug, g=group, gs = gs))
		plt.savefig("{d}_{g}_{gs}_lap.png".format(d=drug, g = group, gs = gs))
		plt.close()

		emb = MDS(n_components=2, normalized_stress='auto')
		X_emb = emb.fit_transform(X)
		resdf = pd.DataFrame({'coord1':X_emb[:,0], 'coord2':X_emb[:,1],'Class':y})
		resdf['Class'] = resdf['Class'].apply(lambda x: "Responder" if x ==1 else "Non-Responder")
		ax = sns.scatterplot(resdf, x = 'coord1', y = 'coord2', hue = 'Class')
		plt.title("{d} {g} {gs} MetricMDS".format(d=drug, g=group, gs = gs))
		plt.savefig("{d}_{g}_{gs}_MMDS.png".format(d=drug, g = group, gs = gs))
		plt.close()
		


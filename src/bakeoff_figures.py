import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

drug = "Nivo"
df = pd.read_csv(f"../results/{drug}/feature_bakeoff_unbalanced.csv")
for tissue in pd.unique(df['Tissue']):
	temp = df[df['Tissue']==tissue]
	for criterion in ['Test Accuracy','Test Balanced Accuracy', 'Test ROC AUC']:
		ax = sns.boxplot(data= temp,x='Features',y=criterion,hue='Model')
		ax.set(title = f"{drug} {criterion} in {tissue}")
		plt.tight_layout()
		plt.savefig(f"../figs/{drug}/{tissue}_{criterion}_balanced.png")
		plt.close()

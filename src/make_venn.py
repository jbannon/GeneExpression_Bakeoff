from matplotlib_venn import venn3, venn3_circles
import utils
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("../expression/cri/Atezo/BLCA/expression_full.csv")

p = utils.fetch_union_genesets()


p = [x for x in p if x in df.columns]
print(len(p))


with open("../genesets/kegg.txt") as istream:
	lines = istream.readlines()
kegg = [x.rstrip() for x in lines]

with open("../genesets/vogelstein.txt") as istream:
	lines = istream.readlines()
vogelstein= [x.rstrip() for x in lines]

with open("../genesets/auslander.txt") as istream:
	lines = istream.readlines()
auslander= [x.rstrip() for x in lines]


ax = venn3([set(kegg),set(vogelstein), set(auslander)],["Kegg","vogelstein","Auslander"])
plt.savefig("../figs/venn1.png")
plt.close()
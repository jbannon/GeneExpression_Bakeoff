

with open("../genesets/Kegg_dup.txt","r") as istream:
	lines = istream.readlines()

lines = [gene.rstrip() for gene in lines]
kegg = [x for x in lines]

lines = [x+"\n" for x in list(set(lines))]
with open("../genesets/kegg.txt","w") as ostream:
	ostream.writelines(lines)


with open("../genesets/CHG/top10/CHG_top10_all.txt","r") as istream:
	lines = [gene.rstrip() for gene in lines]

lines = [x.upper()+"\n" for x in list(set(lines))]
with open("../genesets/CHG_top10.txt","w") as ostream:
	ostream.writelines(lines)

with open("../genesets/CHG/all/chg_all_dup.txt","r") as istream:
	lines = [gene.rstrip().upper() for gene in lines]
chg = [x for x in lines]

lines = [x+"\n" for x in list(set(lines))]
with open("../genesets/CHG_all.txt","w") as ostream:
	ostream.writelines(lines)


print(len(kegg))
print(len(chg))
print(len(set(chg).intersection(set(kegg))))
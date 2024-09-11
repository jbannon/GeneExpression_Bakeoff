from typing import List


DRUG_TISSUE_MAP = {'Atezo':['BLCA','KIRC','PANCAN'],
	'Nivo':['SKCM','KIRC','PANCAN'],
	'Pembro':['STAD','SKCM','PANCAN'],
	'Ipi':['SKCM'],
	'Ipi+Pembro':['SKCM']}


def fetch_geneset(
	geneset_name:str
	)->List[str]:
	
	gsn_2_file = {
		'cosmic1':'cosmic_1.txt',
		'cosmic2':'cosmic_2.txt',
		'cosmic':'cosmic_all.txt',
		'kegg':'kegg.txt',
		'vogelstein':'vogelstein.txt',
		'auslander':'auslander.txt',
		'mdsig':'mdsig_hallmarks.txt'
		}
	geneset_name = geneset_name.lower()
	
	gs_file = f"../genesets/{gsn_2_file[geneset_name]}"

	with open(gs_file,"r") as istream:
		lines = istream.readlines()

	gene_names = [x.rstrip() for x in lines]
	return gene_names


def fetch_union_genesets(
	genesets:List[str] = ['kegg','auslander','vogelstein']
	)->List[str]:
	
	all_genes = []
	gsn_2_file = {
		'cosmic1':'cosmic_1.txt',
		'cosmic2':'cosmic_2.txt',
		'cosmic':'cosmic_all.txt',
		'kegg':'kegg.txt',
		'vogelstein':'vogelstein.txt',
		'auslander':'auslander.txt',
		'mdsig':'mdsig_hallmarks.txt'
		}
	
	for gs in genesets:
		gs_file = f"../genesets/{gsn_2_file[gs]}"
		with open(gs_file,"r") as istream:
			lines = istream.readlines()
		gene_names = [x.rstrip() for x in lines]

		all_genes.extend(gene_names)


	all_genes = sorted(list(set(all_genes)))
	return all_genes

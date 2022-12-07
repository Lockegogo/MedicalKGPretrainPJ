## for data
import pandas as pd
import numpy as np
import random


# file -> txt
def to_txt(file, name):
    path_name = 'data/BioKG/entities/' + name + '.txt'
    with open(path_name, 'w') as f:
        for i in file:
            f.write(i + '\n')


biokg = pd.read_csv('data/BioKG/kg.csv', index_col=0)

## to avoid information leakage
# dti = pd.read_csv('data/BioKG/dti.csv')
# all_biokg = pd.concat([biokg, dti])
# all_biokg.drop_duplicates(inplace=True)

all_biokg = biokg


PROTEIN = []
PATHWAY = []
DISORDER = []
DRUG = []
DISEASE = []

# Complexes are composites of proteins which represent a set of physically interacting proteins.
COMPLEX = []




# DDI
ddi = all_biokg[all_biokg['relation'] == 'DDI'].reset_index(drop=True)
ddi.to_csv('./data/BioKG//relations/drug_drug.csv')

drug_head = ddi['head'].values.tolist()
drug_tail = ddi['tail'].values.tolist()

DRUG.extend(drug_head)
DRUG.extend(drug_tail)


# PPI
ppi = all_biokg[all_biokg['relation'] == 'PPI'].reset_index(drop=True)
ppi.to_csv('./data/BioKG//relations/protein_protein.csv')

protein_head = ppi['head'].values.tolist()
protein_tail = ppi['tail'].values.tolist()

PROTEIN.extend(protein_head)
PROTEIN.extend(protein_tail)


# PROTEIN_DISEASE_ASSOCIATION
pro_disease = all_biokg[
    all_biokg['relation'] == 'PROTEIN_DISEASE_ASSOCIATION'
].reset_index(drop=True)
pro_disease.to_csv('./data/BioKG//relations/protein_disease.csv')

protein_head = pro_disease['head'].values.tolist()
disease_tail = pro_disease['tail'].values.tolist()

PROTEIN.extend(protein_head)
DISEASE.extend(disease_tail)


# PROTEIN_PATHWAY_ASSOCIATION
# 这里面的 pathway来自 KEGG, Reactome, Drugbank(SMPDB)
# biokg(21 species):250118
pro_path = all_biokg[
    all_biokg['relation'] == 'PROTEIN_PATHWAY_ASSOCIATION'
].reset_index(drop=True)
pro_path.to_csv('./data/BioKG//relations/protein_pathway.csv')

protein_head = pro_path['head'].values.tolist()
pathway_tail = pro_path['tail'].values.tolist()

PROTEIN.extend(protein_head)
PATHWAY.extend(pathway_tail)


# MEMBER_OF_COMPLEX
pro_comp = all_biokg[all_biokg['relation'] == 'MEMBER_OF_COMPLEX'].reset_index(
    drop=True
)
pro_comp.to_csv('./data/BioKG//relations/protein_complex.csv')

protein_head = pro_comp['head'].values.tolist()
complex_tail = pro_comp['tail'].values.tolist()

PROTEIN.extend(protein_head)
COMPLEX.extend(complex_tail)


# DRUG_DISEASE_ASSOCIATION
drug_disease = all_biokg[
    all_biokg['relation'] == 'DRUG_DISEASE_ASSOCIATION'
].reset_index(drop=True)
drug_disease.to_csv('./data/BioKG//relations/drug_disease.csv')

drug_head = drug_disease['head'].values.tolist()
disease_tail = drug_disease['tail'].values.tolist()

DRUG.extend(drug_head)
DISEASE.extend(disease_tail)


# Drug_Target_Interaction
drug_target = all_biokg[all_biokg['relation'] == 'Drug_Target_Interaction'].reset_index(
    drop=True
)
drug_target.to_csv('./data/BioKG//relations/drug_target.csv')

drug_head = drug_target['head'].values.tolist()
protein_tail = drug_target['tail'].values.tolist()

DRUG.extend(drug_head)
PROTEIN.extend(protein_tail)


# COMPLEX_IN_PATHWAY
comp_path = all_biokg[all_biokg['relation'] == 'COMPLEX_IN_PATHWAY'].reset_index(
    drop=True
)
comp_path.to_csv('./data/BioKG//relations/complex_pathway.csv')

comp_head = comp_path['head'].values.tolist()
pathway_tail = comp_path['tail'].values.tolist()

COMPLEX.extend(comp_head)
PATHWAY.extend(pathway_tail)


# COMPLEX_TOP_LEVEL_PATHWAY
comp_top_path = all_biokg[
    all_biokg['relation'] == 'COMPLEX_TOP_LEVEL_PATHWAY'
].reset_index(drop=True)
comp_top_path.to_csv('./data/BioKG//relations/complex_top_pathway.csv')

comp_head = comp_top_path['head'].values.tolist()
pathway_tail = comp_top_path['tail'].values.tolist()

COMPLEX.extend(comp_head)
PATHWAY.extend(pathway_tail)


# DRUG_PATHWAY_ASSOCIATION
drug_path = all_biokg[all_biokg['relation'] == 'DRUG_PATHWAY_ASSOCIATION'].reset_index(
    drop=True
)
drug_path.to_csv('./data/BioKG//relations/drug_pathway.csv')

drug_head = drug_path['head'].values.tolist()
pathway_tail = drug_path['tail'].values.tolist()

DRUG.extend(drug_head)
PATHWAY.extend(pathway_tail)


# DISEASE_GENETIC_DISORDER
disease_disorder = all_biokg[
    all_biokg['relation'] == 'DISEASE_GENETIC_DISORDER'
].reset_index(drop=True)
disease_disorder.to_csv('./data/BioKG//relations/disease_disorder.csv')

disease_head = disease_disorder['head'].values.tolist()
disorder_tail = disease_disorder['tail'].values.tolist()

DISEASE.extend(disease_head)
DISORDER.extend(disorder_tail)


# RELATED_GENETIC_DISORDER
pro_disorder = all_biokg[
    all_biokg['relation'] == 'RELATED_GENETIC_DISORDER'
].reset_index(drop=True)
pro_disorder.to_csv('./data/BioKG//relations/protein_disorder.csv')

pro_head = pro_disorder['head'].values.tolist()
disorder_tail = pro_disorder['tail'].values.tolist()

PROTEIN.extend(pro_head)
DISORDER.extend(disorder_tail)


# DISEASE_PATHWAY_ASSOCIATION
disease_path = all_biokg[
    all_biokg['relation'] == 'DISEASE_PATHWAY_ASSOCIATION'
].reset_index(drop=True)
disease_path.to_csv('./data/BioKG//relations/disease_pathway.csv')

disease_head = disease_path['head'].values.tolist()
pathway_tail = disease_path['tail'].values.tolist()

DISEASE.extend(disease_head)
PATHWAY.extend(pathway_tail)


PROTEIN = set(PROTEIN)
PATHWAY = set(PATHWAY)
DISORDER = set(DISORDER)
DRUG = set(DRUG)
DISEASE = set(DISEASE)
COMPLEX = set(COMPLEX)


to_txt(PROTEIN, 'PROTEIN')
to_txt(PATHWAY, 'PATHWAY')
to_txt(DISORDER, 'DISORDER')
to_txt(DRUG, 'DRUG')
to_txt(DISEASE, 'DISEASE')
to_txt(COMPLEX, 'COMPLEX')


########### add GENE ############
GENE = []

protein_gene = pd.read_csv('data/BioKG/uniprot-gene-protein.tsv', sep='\t')
protein_gene = protein_gene[['Entry', 'Gene Names']]
protein_gene.rename(columns={'Entry':'head', 'Gene Names':'tail'}, inplace=True)

# 543979
protein_gene.dropna(axis=0, how='any', inplace=True)
# keep the first gene
def save_first(x):
    x = str(x)
    x = x.split(' ')[0]
    return x

protein_gene['tail'] = protein_gene['tail'].apply(save_first)

protein_gene = protein_gene[protein_gene['head'].isin(PROTEIN)].reset_index(drop=True)

gene = protein_gene['tail'].values.tolist()
GENE.extend(gene)
GENE = set(GENE)

to_txt(GENE, 'GENE')

protein_gene.to_csv('./data/BioKG//relations/protein_gene.csv')














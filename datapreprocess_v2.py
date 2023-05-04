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


# biokg = pd.read_csv('data/BioKG/kg.csv', index_col=0)
# # all_biokg = biokg

# # to avoid information leakage
# dti = pd.read_csv('data/BioKG/dti.csv')
# all_biokg = pd.concat([biokg, dti])
# all_biokg.drop_duplicates(inplace=True)

######## 2023.4.4 新增 PrimeKG #########
# 效果好像并不好

all_biokg = pd.read_csv('data/biokg_primekg.csv', index_col=0)


PROTEIN = []
PATHWAY = []
DISORDER = []
DRUG = []
DISEASE = []
GENE = []

# Complexes are composites of proteins which represent a set of physically interacting proteins.
COMPLEX = []



# DDI
ddi = all_biokg[all_biokg['relation'] == 'DDI'].reset_index(drop=True)
# 删除 ddi中的重复项
ddi.drop_duplicates(inplace=True)
ddi.to_csv('./data/BioKG//relations/drug_drug.csv')

drug_head = ddi['head'].values.tolist()
drug_tail = ddi['tail'].values.tolist()

DRUG.extend(drug_head)
DRUG.extend(drug_tail)


# PPI
ppi = all_biokg[all_biokg['relation'] == 'PPI'].reset_index(drop=True)
# 删除 ppi中的重复项
ppi.drop_duplicates(inplace=True)
ppi.to_csv('./data/BioKG//relations/protein_protein.csv')

protein_head = ppi['head'].values.tolist()
protein_tail = ppi['tail'].values.tolist()

PROTEIN.extend(protein_head)
PROTEIN.extend(protein_tail)


# PROTEIN_DISEASE_ASSOCIATION
pro_disease = all_biokg[
    all_biokg['relation'] == 'PROTEIN_DISEASE_ASSOCIATION'
].reset_index(drop=True)
# 删除 pro_disease中的重复项
pro_disease.drop_duplicates(inplace=True)
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
# 删除 pro_path 中的重复项
pro_path.drop_duplicates(inplace=True)
pro_path.to_csv('./data/BioKG//relations/protein_pathway.csv')

protein_head = pro_path['head'].values.tolist()
pathway_tail = pro_path['tail'].values.tolist()

PROTEIN.extend(protein_head)
PATHWAY.extend(pathway_tail)


# MEMBER_OF_COMPLEX
pro_comp = all_biokg[all_biokg['relation'] == 'MEMBER_OF_COMPLEX'].reset_index(
    drop=True
)
# 删除 pro_comp 中的重复项
pro_comp.drop_duplicates(inplace=True)
pro_comp.to_csv('./data/BioKG//relations/protein_complex.csv')

protein_head = pro_comp['head'].values.tolist()
complex_tail = pro_comp['tail'].values.tolist()

PROTEIN.extend(protein_head)
COMPLEX.extend(complex_tail)


# DRUG_DISEASE_ASSOCIATION
drug_disease = all_biokg[
    all_biokg['relation'] == 'DRUG_DISEASE_ASSOCIATION'
].reset_index(drop=True)
# 删除 drug_disease 中的重复项
drug_disease.drop_duplicates(inplace=True)
drug_disease.to_csv('./data/BioKG//relations/drug_disease.csv')

drug_head = drug_disease['head'].values.tolist()
disease_tail = drug_disease['tail'].values.tolist()

DRUG.extend(drug_head)
DISEASE.extend(disease_tail)


# Drug_Target_Interaction
drug_target = all_biokg[all_biokg['relation'] == 'Drug_Target_Interaction'].reset_index(
    drop=True
)
# 删除 drug_target 中的重复项
drug_target.drop_duplicates(inplace=True)
drug_target.to_csv('./data/BioKG//relations/drug_target.csv')

drug_head = drug_target['head'].values.tolist()
protein_tail = drug_target['tail'].values.tolist()

DRUG.extend(drug_head)
PROTEIN.extend(protein_tail)



# COMPLEX_IN_PATHWAY
comp_path = all_biokg[all_biokg['relation'] == 'COMPLEX_IN_PATHWAY'].reset_index(
    drop=True
)
# 删除 comp_path 中的重复项
comp_path.drop_duplicates(inplace=True)
comp_path.to_csv('./data/BioKG//relations/complex_pathway.csv')

comp_head = comp_path['head'].values.tolist()
pathway_tail = comp_path['tail'].values.tolist()

COMPLEX.extend(comp_head)
PATHWAY.extend(pathway_tail)


# COMPLEX_TOP_LEVEL_PATHWAY
comp_top_path = all_biokg[
    all_biokg['relation'] == 'COMPLEX_TOP_LEVEL_PATHWAY'
].reset_index(drop=True)
# 删除 comp_top_path 中的重复项
comp_top_path.drop_duplicates(inplace=True)
comp_top_path.to_csv('./data/BioKG//relations/complex_top_pathway.csv')

comp_head = comp_top_path['head'].values.tolist()
pathway_tail = comp_top_path['tail'].values.tolist()

COMPLEX.extend(comp_head)
PATHWAY.extend(pathway_tail)


# DRUG_PATHWAY_ASSOCIATION
drug_path = all_biokg[all_biokg['relation'] == 'DRUG_PATHWAY_ASSOCIATION'].reset_index(
    drop=True
)
# 删除 drug_path 中的重复项
drug_path.drop_duplicates(inplace=True)
drug_path.to_csv('./data/BioKG//relations/drug_pathway.csv')

drug_head = drug_path['head'].values.tolist()
pathway_tail = drug_path['tail'].values.tolist()

DRUG.extend(drug_head)
PATHWAY.extend(pathway_tail)


# DISEASE_GENETIC_DISORDER
disease_disorder = all_biokg[
    all_biokg['relation'] == 'DISEASE_GENETIC_DISORDER'
].reset_index(drop=True)
# 删除 disease_disorder 中的重复项
disease_disorder.drop_duplicates(inplace=True)
disease_disorder.to_csv('./data/BioKG//relations/disease_disorder.csv')

disease_head = disease_disorder['head'].values.tolist()
disorder_tail = disease_disorder['tail'].values.tolist()

DISEASE.extend(disease_head)
DISORDER.extend(disorder_tail)


# RELATED_GENETIC_DISORDER
pro_disorder = all_biokg[
    all_biokg['relation'] == 'RELATED_GENETIC_DISORDER'
].reset_index(drop=True)
# 删除 pro_disorder 中的重复项
pro_disorder.drop_duplicates(inplace=True)
pro_disorder.to_csv('./data/BioKG//relations/protein_disorder.csv')

pro_head = pro_disorder['head'].values.tolist()
disorder_tail = pro_disorder['tail'].values.tolist()

PROTEIN.extend(pro_head)
DISORDER.extend(disorder_tail)


# DISEASE_PATHWAY_ASSOCIATION
disease_path = all_biokg[
    all_biokg['relation'] == 'DISEASE_PATHWAY_ASSOCIATION'
].reset_index(drop=True)
# 删除 disease_path 中的重复项
disease_path.drop_duplicates(inplace=True)
disease_path.to_csv('./data/BioKG//relations/disease_pathway.csv')

disease_head = disease_path['head'].values.tolist()
pathway_tail = disease_path['tail'].values.tolist()

DISEASE.extend(disease_head)
PATHWAY.extend(pathway_tail)


# PROTEIN_GENE
pro_gene = all_biokg[all_biokg['relation'] == 'PROTEIN_GENE'].reset_index(drop=True)
# 删除 pro_gene 中的重复项
pro_gene.drop_duplicates(inplace=True)
pro_gene.to_csv('./data/BioKG//relations/protein_gene.csv')

pro_head = pro_gene['head'].values.tolist()
gene_tail = pro_gene['tail'].values.tolist()

PROTEIN.extend(pro_head)
GENE.extend(gene_tail)



# GENE_GENE
gene_gene = all_biokg[all_biokg['relation'] == 'GGI'].reset_index(drop=True)
# 删除 gene_gene 中的重复项
gene_gene.drop_duplicates(inplace=True)
gene_gene.to_csv('./data/BioKG//relations/gene_gene.csv')

gene_head = gene_gene['head'].values.tolist()
gene_tail = gene_gene['tail'].values.tolist()

GENE.extend(gene_head)
GENE.extend(gene_tail)


# DRUG_GENE
drug_gene = all_biokg[all_biokg['relation'] == 'DRUG_GENE'].reset_index(drop=True)
# 删除 drug_gene 中的重复项
drug_gene.drop_duplicates(inplace=True)
drug_gene.to_csv('./data/BioKG//relations/drug_gene.csv')

drug_head = drug_gene['head'].values.tolist()
gene_tail = drug_gene['tail'].values.tolist()

DRUG.extend(drug_head)
GENE.extend(gene_tail)


# PATHWAY_PATHWAY
pathway_pathway = all_biokg[
    all_biokg['relation'] == 'PATHWAY_PATHWAY'
].reset_index(drop=True)
# 删除 pathway_pathway 中的重复项
pathway_pathway.drop_duplicates(inplace=True)
pathway_pathway.to_csv('./data/BioKG//relations/pathway_pathway.csv')

pathway_head = pathway_pathway['head'].values.tolist()
pathway_tail = pathway_pathway['tail'].values.tolist()

PATHWAY.extend(pathway_head)
PATHWAY.extend(pathway_tail)


# GENE_PATHWAY
gene_pathway = all_biokg[
    all_biokg['relation'] == 'GENE_PATHWAY'
].reset_index(drop=True)
# 删除 gene_pathway 中的重复项
gene_pathway.drop_duplicates(inplace=True)
gene_pathway.to_csv('./data/BioKG//relations/gene_pathway.csv')

gene_head = gene_pathway['head'].values.tolist()
pathway_tail = gene_pathway['tail'].values.tolist()

GENE.extend(gene_head)
PATHWAY.extend(pathway_tail)

# DISEASE_DISEASE
disease_disease = all_biokg[
    all_biokg['relation'] == 'DISEASE_DISEASE'
].reset_index(drop=True)
# 删除 disease_disease 中的重复项
disease_disease.drop_duplicates(inplace=True)
disease_disease.to_csv('./data/BioKG//relations/disease_disease.csv')

disease_head = disease_disease['head'].values.tolist()
disease_tail = disease_disease['tail'].values.tolist()

DISEASE.extend(disease_head)
DISEASE.extend(disease_tail)

# PATHWAY_GENE
pathway_gene = all_biokg[
    all_biokg['relation'] == 'PATHWAY_GENE'
].reset_index(drop=True)
# 删除 pathway_gene 中的重复项
pathway_gene.drop_duplicates(inplace=True)
pathway_gene.to_csv('./data/BioKG//relations/pathway_gene.csv')

pathway_head = pathway_gene['head'].values.tolist()
gene_tail = pathway_gene['tail'].values.tolist()

PATHWAY.extend(pathway_head)
GENE.extend(gene_tail)


# DISEASE_GENE
disease_gene = all_biokg[
    all_biokg['relation'] == 'DISEASE_GENE'
].reset_index(drop=True)
# 删除 disease_gene 中的重复项
disease_gene.drop_duplicates(inplace=True)
disease_gene.to_csv('./data/BioKG//relations/disease_gene.csv')

disease_head = disease_gene['head'].values.tolist()
gene_tail = disease_gene['tail'].values.tolist()

DISEASE.extend(disease_head)
GENE.extend(gene_tail)



###################################

PROTEIN = set(PROTEIN)
PATHWAY = set(PATHWAY)
DISORDER = set(DISORDER)
DRUG = set(DRUG)
DISEASE = set(DISEASE)
COMPLEX = set(COMPLEX)
GENE = set(GENE)

print("PROTEIN:", len(PROTEIN))
print("PATHWAY:", len(PATHWAY))
print("DISORDER:", len(DISORDER))
print("DRUG:", len(DRUG))
print("DISEASE:", len(DISEASE))
print("COMPLEX:", len(COMPLEX))
print("GENE:", len(GENE))

to_txt(PROTEIN, 'PROTEIN')
to_txt(PATHWAY, 'PATHWAY')
to_txt(DISORDER, 'DISORDER')
to_txt(DRUG, 'DRUG')
to_txt(DISEASE, 'DISEASE')
to_txt(COMPLEX, 'COMPLEX')
to_txt(GENE, 'GENE')








# #### 下面这段代码运行一次就可以了 ####
# ########### add GENE ############
# GENE = []

# protein_gene = pd.read_csv('data/BioKG/uniprot-gene-protein.tsv', sep='\t')
# protein_gene = protein_gene[['Entry', 'Gene Names']]
# protein_gene.rename(columns={'Entry':'head', 'Gene Names':'tail'}, inplace=True)

# # 543979
# protein_gene.dropna(axis=0, how='any', inplace=True)
# # keep the first gene
# def save_first(x):
#     x = str(x)
#     x = x.split(' ')[0]
#     return x

# protein_gene['tail'] = protein_gene['tail'].apply(save_first)

# protein_gene = protein_gene[protein_gene['head'].isin(PROTEIN)].reset_index(drop=True)

# gene = protein_gene['tail'].values.tolist()
# GENE.extend(gene)
# GENE = set(GENE)

# to_txt(GENE, 'GENE')
# print("GENE:", len(GENE))

# protein_gene.to_csv('./data/BioKG//relations/protein_gene.csv')


# # 将 protein_gene 补充到 biokg 中
# # 给 protein_gene 中间增加一列 relation
# protein_gene['relation'] = 'PROTEIN_GENE'
# # 调整列的顺序
# protein_gene = protein_gene[['head', 'relation', 'tail']]
# # 将 protein_gene 补充到 biokg 中
# biokg = pd.concat([biokg, protein_gene], axis=0)
# # 重置索引
# biokg.reset_index(drop=True, inplace=True)
# # 保存
# # index=False 表示不保存索引
# biokg.to_csv('./data/BioKG/kg.csv')















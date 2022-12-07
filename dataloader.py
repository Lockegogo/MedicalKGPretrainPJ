## For data
import pandas as pd
import numpy as np

## For machine learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import os
import pickle


## drug-target


class DataLoaderFinetune:
    def __init__(self, dPath=None):
        self.dataset_path = dPath
        self.relation_path = os.path.join(self.dataset_path, 'dti.csv')

    def load(self):
        """
        Load dataset
        """
        dti = pd.read_csv(self.relation_path)
        drug_nodes = dti['head'].values.tolist()
        drug_nodes = set(drug_nodes)
        target_nodes = dti['tail'].values.tolist()
        target_nodes = set(target_nodes)

        idx_node_map = {}
        idx_node_id_map = {}

        idx_node_map['DRUG'] = {j: i for i, j in enumerate(drug_nodes)}
        idx_node_id_map['DRUG'] = {i: j for i, j in enumerate(drug_nodes)}

        idx_node_map['PROTEIN'] = {j: i for i, j in enumerate(target_nodes)}
        idx_node_id_map['PROTEIN'] = {i: j for i, j in enumerate(target_nodes)}

        dti['head'] = dti['head'].map(idx_node_map['DRUG'])
        dti['tail'] = dti['tail'].map(idx_node_map['PROTEIN'])

        return idx_node_map, idx_node_id_map, dti

    def to_graph(self, emb_dim, use_info, use_pretrain_emb,pretrained_emb_path=''):
        """
        Data to Graph
        """
        idx_node_map, idx_node_id_map, dti = self.load()

        graph_data = {}
        graph_k = ('DRUG', 'drug_target', 'PROTEIN')
        graph_v = (
            torch.tensor(dti['head'].values).long(),
            torch.tensor(dti['tail'].values).long(),
        )
        graph_data[graph_k] = graph_v

        # reverse
        graph_r_k = ('PROTEIN', 'R_drug_target', 'DRUG')
        graph_r_v = (
            torch.tensor(dti['tail'].values).long(),
            torch.tensor(dti['head'].values).long(),
        )
        graph_data[graph_r_k] = graph_r_v

        graph = dgl.heterograph(graph_data)

        # Random initialization
        graph = self.random_init(graph, emb_dim)

        # use known information to initialize node emb
        if use_info:
            graph = self.info_init(graph, emb_dim, idx_node_id_map)

        # use pretrained emb to initialize
        if use_pretrain_emb:
            graph = self.pretrain_init(graph, emb_dim, idx_node_id_map, pretrained_emb_path)

        return graph, idx_node_map, idx_node_id_map

    def random_init(self, graph, emb_dim):
        """
        Random initialization of graph node embedding
        """
        for ntype in graph.ntypes:
            emb = nn.Parameter(
                torch.Tensor(graph.number_of_nodes(ntype), emb_dim),
                requires_grad=False,
            )
            nn.init.xavier_uniform_(emb)
            graph.nodes[ntype].data["feature"] = emb

        return graph

    def info_init(self, graph, emb_dim, idx_node_id_map):
        """
        Initialize emb with the node's structure information or other information
        """
        # smiles of drugs
        fp_id = pd.read_csv(os.path.join(self.dataset_path, 'comp_struc.csv'))['head']
        drug_feats = pd.read_csv(os.path.join(self.dataset_path, 'fp_df.csv'))

        drug_feats = drug_feats.iloc[:, :-1]

        # sequences of proteins
        df_proseq = pd.read_csv(os.path.join(self.dataset_path, 'pro_seq.csv'))
        pro_id = df_proseq['pro_id']
        # list of protein descriptors, pro_ids
        pro_feats = pd.read_csv(os.path.join(self.dataset_path, 'prodes_df.csv'))
        # 147
        pro_feats = pro_feats.iloc[:, :-1]

        mms = MinMaxScaler(feature_range=(0, 1))

        pro_feats_scaled = mms.fit_transform(pro_feats)
        pro_feats_scaled2 = PCA(n_components=emb_dim).fit_transform(pro_feats_scaled)
        pro_feats_scaled3 = mms.fit_transform(pro_feats_scaled2)
        prodes_df = pd.concat([pro_id, pd.DataFrame(pro_feats_scaled3)], axis=1)

        drug_feats_scaled2 = PCA(n_components=emb_dim).fit_transform(drug_feats)
        drug_feats_scaled3 = mms.fit_transform(drug_feats_scaled2)
        fp_df = pd.concat([fp_id, pd.DataFrame(drug_feats_scaled3)], axis=1)
        fp_df.rename(columns={'head': 'drug_id'}, inplace=True)

        notFind = 0
        for i in range(graph.num_nodes('DRUG')):

            drug_name = idx_node_id_map['DRUG'][i]
            if drug_name in fp_df['drug_id'].values:
                temp = (
                    fp_df[fp_df['drug_id'] == drug_name]
                    .iloc[0, 1:]
                    .values.astype('float')
                )
                temp = torch.tensor(temp)
                graph.nodes['DRUG'].data['feature'][i] = temp
            else:
                notFind += 1

        # print(
        #     "The number of DRUG nodes for which no structural information was found is: ",
        #     notFind,
        # )

        notFind = 0
        for i in range(graph.num_nodes('PROTEIN')):
            protein_name = idx_node_id_map['PROTEIN'][i]
            if protein_name in prodes_df['pro_id'].values:
                temp = (
                    prodes_df[prodes_df['pro_id'] == protein_name]
                    .iloc[0, 1:]
                    .values.astype('float')
                )
                temp = torch.tensor(temp)
                graph.nodes['PROTEIN'].data['feature'][i] = temp
            else:
                notFind += 1

        # print(
        #     "The number of PROTEIN nodes for which no structural information was found is: ",
        #     notFind,
        # )

        return graph

    def pretrain_init(self, graph, emb_dim, idx_node_id_map, pretrained_emb_path):
        # read emb dict
        with open(pretrained_emb_path, 'rb') as f:
            node_emb_dict = pickle.load(f)
            # print(node_emb_dict['GENE']['Rbm47'])

            notFind = 0
            for i in range(graph.num_nodes('DRUG')):

                drug_name = idx_node_id_map['DRUG'][i]
                if drug_name in node_emb_dict['DRUG']:
                    graph.nodes['DRUG'].data['feature'][i] = torch.FloatTensor(node_emb_dict['DRUG'][drug_name])
                else:
                    notFind += 1

            # print(
            #     "The number of DRUG nodes for which no pretrained emb was found is: ",
            #     notFind,
            # )

            notFind = 0
            for i in range(graph.num_nodes('PROTEIN')):

                target_name = idx_node_id_map['PROTEIN'][i]
                if target_name in node_emb_dict['PROTEIN']:
                    graph.nodes['PROTEIN'].data['feature'][i] = torch.FloatTensor(node_emb_dict['PROTEIN'][target_name])
                else:
                    notFind += 1

            # print(
            #     "The number of PROTEIN nodes for which no pretrained emb was found is: ",
            #     notFind,
            # )

        return graph



if __name__ == "__main__":
    dataset_path = 'data/BioKG'
    dti = DataLoaderFinetune(dPath=dataset_path)
    pretrained_emb_path = 'pretrained_emb_dict.pkl'
    g, idx_node_map, idx_node_id_map = dti.to_graph(emb_dim=128, use_info=True, pretrained_emb_path=pretrained_emb_path)


    ## test
    index = idx_node_map['DRUG']['DB00313']
    print(g.nodes['DRUG'].data['feature'][index])

    with open(pretrained_emb_path,'rb') as f:
        node_emb_dict = pickle.load(f)
        print(node_emb_dict['DRUG']['DB00313'])




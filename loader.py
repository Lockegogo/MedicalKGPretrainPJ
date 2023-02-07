import os
import pickle

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class BioDataset:
    def __init__(self, dPath=None):
        self.dataset_path = dPath
        self.entities_path = os.path.join(self.dataset_path, 'entities/')
        self.relations_path = os.path.join(self.dataset_path, 'relations/')

    def load(self):
        """
        Load dataset
        """
        node_types = os.listdir(self.entities_path)
        # node_types = ['GENE','PROTEIN','PATHWAY','COMPLEX','DISEASE','DISORDER','DRUG']

        idx_node_map = {}
        idx_node_id_map = {}
        for ntype in node_types:
            npath = self.entities_path + ntype
            idx = np.genfromtxt(npath, dtype=np.dtype(str), delimiter='\t')
            # remove suffix
            ntype_name = ntype.split('.')[0]
            idx_node_map[ntype_name] = {j: i for i, j in enumerate(idx)}
            idx_node_id_map[ntype_name] = {i: j for i, j in enumerate(idx)}

        edge_types = os.listdir(self.relations_path)
        edges_u_v = {}
        for etype in edge_types:
            epath = self.relations_path + etype
            edge_csv = pd.read_csv(epath, index_col=0)
            # remove suffix
            etype_name = etype.split('.')[0]
            rel = etype[:-4].upper().split('_')
            head = rel[0]
            tail = rel[-1]
            # TARGET -> PROTEIN
            if tail == 'TARGET':
                tail = 'PROTEIN'
            edge_csv['head'] = edge_csv['head'].map(idx_node_map[head])
            edge_csv['tail'] = edge_csv['tail'].map(idx_node_map[tail])
            edges_u_v[etype_name] = edge_csv

        return idx_node_map, idx_node_id_map, edges_u_v

    # 修改一下，改成继续用之前的 emb
    def to_graph(self, emb_dim, use_info, use_linkpred_emb=False, use_SRRSC_emb=False):
        """
        Data to Heterogeneous Graph
        """
        idx_node_map, idx_node_id_map, edges_u_v = self.load()
        graph_data = {}
        for k, v in edges_u_v.items():
            rel = k.upper().split('_')
            head = rel[0]
            tail = rel[-1]
            # TARGET -> PROTEIN
            if tail == 'TARGET':
                tail = 'PROTEIN'
            graph_k = (head, k, tail)
            graph_v = (
                torch.tensor(v['head'].values).long(),
                torch.tensor(v['tail'].values).long(),
            )
            graph_data[graph_k] = graph_v

            # reverse
            graph_r_k = (tail, 'R_' + k, head)
            graph_r_v = (
                torch.tensor(v['tail'].values).long(),
                torch.tensor(v['head'].values).long(),
            )
            graph_data[graph_r_k] = graph_r_v

        graph = dgl.heterograph(graph_data)

        # Random initialization
        graph = self.random_init(graph, emb_dim)

        # use known information to initialize node emb
        if use_info:
            graph = self.info_init(graph, emb_dim, idx_node_id_map)

        # use pretrained emb to initialize node emb
        if use_linkpred_emb:
            graph = self.pretrained_init(graph, emb_dim, idx_node_id_map, use_linkpred=True, use_SRRSC=False)

        if use_SRRSC_emb:
            graph = self.pretrained_init(graph, emb_dim, idx_node_id_map, use_linkpred=False, use_SRRSC=True)

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

        print(
            "The number of DRUG nodes for which no structural information was found is: ",
            notFind,
        )

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

        print(
            "The number of PROTEIN nodes for which no structural information was found is: ",
            notFind,
        )

        return graph

    def pretrained_init(self, graph, emb_dim, idx_node_id_map, use_linkpred, use_SRRSC):
        if use_linkpred:
            # 使用链接预测预训练的节点向量进行初始化
            pretrained_emb_path = 'pretrain_emb/pretrained_emb_dict_6_256_GAT.pkl'
        elif use_SRRSC:
            pretrained_emb_path = 'pretrain_emb/pretrained_emb_dict_SRRSC_10000.pkl'

        # read emb dict
        with open(pretrained_emb_path, 'rb') as f:
            node_emb_dict = pickle.load(f)

            for ntype in graph.ntypes:
                for i in range(graph.num_nodes(ntype)):
                    node_name = idx_node_id_map[ntype][i]
                    if node_name in node_emb_dict[ntype]:
                        graph.nodes[ntype].data['feature'][i] = torch.FloatTensor(node_emb_dict[ntype][node_name])

        return graph



if __name__ == "__main__":
    dataset_path = 'data/BioKG'
    biokg = BioDataset(dPath=dataset_path)
    g, _, _ = biokg.to_graph(emb_dim=128, use_info=True)
    for ntype in g.ntypes:
        print(f"The number of {ntype} node：{g.num_nodes(ntype)}")

    for etype in g.canonical_etypes:
        print(f"The number of {etype} edge：{g.number_of_edges(etype=etype)}")


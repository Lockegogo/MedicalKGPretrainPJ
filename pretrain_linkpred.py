import argparse

from loader import BioDataset
from util import ScorePredictor
from model import GNN

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import sklearn.linear_model as lm
import sklearn.metrics as skm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import pickle


def blockloader(graph, batch_size, num_workers):
    """
    For link prediction:
    graph data -> block
    """
    # define neighbor sampler
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    # define negative sampler
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(9)

    eid_dict = {etype: graph.edges(etype=etype, form='eid') for etype in graph.etypes}

    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, negative_sampler=neg_sampler
    )

    dataloader = dgl.dataloading.DataLoader(
        graph,
        eid_dict,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    return dataloader


def compute_loss(pos_score, neg_score):
    """
    compute loss
    """
    loss = 0
    for etype, p_score in pos_score.items():
        if len(p_score) != 0:
            n = p_score.shape[0]
            loss += (
                (neg_score[etype].view(n, -1) - p_score.view(n, -1) + 1)
                .clamp(min=0)
                .mean()
            )

    return loss


def train(args, model, device, loader, predictor, optimizer):
    model.train()

    train_loss_accum = 0
    with tqdm(loader) as tq:
        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(tq):
            blocks = [b.to(device) for b in blocks]
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            inputs = blocks[0].srcdata['feature']
            outputs = model(blocks, inputs)
            pos_score = predictor(pos_graph, outputs)
            neg_score = predictor(neg_graph, outputs)

            loss = compute_loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.detach().cpu().item())
            tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)

    return train_loss_accum / (step + 1)


def inference(model, graph, input_features, batch_size, emb_dim, device):
    """
    get the pre-trained emb
    """
    model.eval()

    nid_dict = {ntype: graph.nodes(ntype=ntype) for ntype in graph.ntypes}

    # one layer at a time, taking all neighbors
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.NodeDataLoader(
        graph,
        nid_dict,
        sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    with torch.no_grad():

        output_features = {
            node: torch.zeros(graph.number_of_nodes(node), emb_dim)
            for node in graph.ntypes
        }

        for input_nodes, output_nodes, blocks in tqdm(dataloader):
            blocks = [b.to(device) for b in blocks]
            x = blocks[0].srcdata['feature']
            h = model(blocks, x)
            for key, value in h.items():
                output_features[key][output_nodes[key].type(torch.long)] = value.cpu()
        input_features = output_features
    return output_features


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='DGL implementation of pre-training of medical KG'
    )
    parser.add_argument(
        '--device', type=int, default=0, help='which gpu to use if any (default: 0)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='input batch size for training (default: 1024)',
    )
    parser.add_argument(
        '--epochs', type=int, default=3, help='number of epochs to train (default: 10)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001, help='learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--decay', type=float, default=0, help='weight decay (default: 0)'
    )
    parser.add_argument(
        '--num_layer',
        type=int,
        default=2,
        help='number of GNN message passing layers (default: 2).',
    )
    parser.add_argument(
        '--emb_dim', type=int, default=128, help='embedding dimensions (default: 128)'
    )
    parser.add_argument(
        '--dropout_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)'
    )
    parser.add_argument('--gnn_type', type=str, default="GAT")

    parser.add_argument('--use_info', type=str, default=False)

    parser.add_argument(
        '--seed', type=int, default=42, help="Seed for splitting dataset."
    )
    parser.add_argument(
        '--runseed', type=int, default=0, help="Seed for running experiments."
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='number of workers for dataset loading',
    )
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # set up dataset
    dataset_path = 'data/BioKG'
    dataset = BioDataset(dPath=dataset_path)
    graph, idx_node_map, idx_node_id_map = dataset.to_graph(
        emb_dim=args.emb_dim, use_info=args.use_info
    )
    dataloader = blockloader(
        graph, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # set up model
    model_name = args.gnn_type
    model = GNN(
        in_features=args.emb_dim,
        hidden_features=args.emb_dim,
        out_features=args.emb_dim,
        etypes=graph.etypes,
        gnn_type=args.gnn_type,
    )

    model.to(device)

    predictor = ScorePredictor()
    predictor.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.decay,
    )

    # 1. train
    for epoch in range(1, args.epochs + 1):
        print("==== epoch " + str(epoch) + " ====")
        train_loss = train(args, model, device, dataloader, predictor, optimizer)
        print(f"train loss in epoch {epoch} is: ", train_loss)

    # 2. save model
    model_name = (
        str(args.gnn_type) + "_" + str(args.epochs) + "_" + str(args.emb_dim) + ".pth"
    )
    model_path = os.path.join('model', model_name)
    torch.save(model.state_dict(), model_path)

    ## 3. save emb
    total_nodes = 0
    for node_type in graph.ntypes:
        total_nodes += graph.num_nodes(node_type)

    emb = inference(
        model, graph, graph.ndata['feature'], total_nodes, args.emb_dim, args.device
    )

    for key, value in emb.items():
        graph.nodes[key].data['feature'] = value

    save_emb(
        graph, idx_node_map, idx_node_id_map, args.epochs, args.emb_dim, args.gnn_type
    )
    ## if you want to read this emb file:
    # with open("pretrained_emb_dict.pkl",'rb') as f:
    #     node_emb_dict = pickle.load(f)
    #     print(node_emb_dict['GENE']['Rbm47'])


def save_emb(graph, idx_node_map, idx_node_id_map, epoch, emb_dim, gnn_type):
    node_feature_dict = {}
    for ntype in graph.ntypes:
        node_feature_dict[ntype] = {}
        for i in range(graph.num_nodes(ntype)):
            node_name = idx_node_id_map[ntype][i]
            node_feature_dict[ntype][node_name] = (
                graph.nodes[ntype].data['feature'][i].cpu().tolist()
            )

    emb_name = (
        "pretrained_emb_dict_"
        + str(epoch)
        + "_"
        + str(emb_dim)
        + "_"
        + str(gnn_type)
        + '.pkl'
    )
    emb_path = os.path.join('pretrain_emb', emb_name)
    with open(emb_path, 'wb') as f:
        pickle.dump(node_feature_dict, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()


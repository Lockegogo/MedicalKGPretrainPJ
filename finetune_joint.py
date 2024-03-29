import argparse
import pickle

import dgl
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dataloader import DataLoaderFinetune
from model import MLPPredictor
from util import split_train_test


# link prediction for drug-target: dti.csv #

def compute_loss_homo(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    return (
        (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()
    )


def LinkPredictionEvaluate(model, test_pos_graph, test_neg_graph):
    pos_score = model(test_pos_graph, test_pos_graph.ndata['feature'])
    neg_score = model(test_neg_graph, test_neg_graph.ndata['feature'])
    link_logits = torch.cat([pos_score, neg_score])
    link_probs = link_logits.sigmoid()
    link_labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    result = roc_auc_score(
        link_labels.cpu().detach().numpy(), link_probs.cpu().detach().numpy()
    )
    return result


def train(
        args,
        model,
        optimizer,
        train_pos_graph,
        train_neg_graph,
        val_pos_graph,
        val_neg_graph,
        best_model_path,
):
    best_accuracy = 0
    train_loss = []
    val_auc = []
    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train()

        pos_score = model(train_pos_graph, train_pos_graph.ndata['feature'])
        neg_score = model(train_neg_graph, train_neg_graph.ndata['feature'])
        loss = compute_loss_homo(pos_score, neg_score)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()

        auc = LinkPredictionEvaluate(model, val_pos_graph, val_neg_graph)
        val_auc.append(auc)
        if best_accuracy < auc:
            best_accuracy = auc
            torch.save(model, best_model_path)

    return train_loss, val_auc, best_accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='DGL implementation of pre-training of medical KG'
    )
    parser.add_argument(
        '--device', type=int, default=0, help='which gpu to use if any (default: 0)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=500,
        help='number of epochs to train (default: 500)',
    )
    parser.add_argument(
        '--lr', type=float, default=0.001, help='learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--decay', type=float, default=0, help='weight decay (default: 0)'
    )
    parser.add_argument(
        '--emb_dim', type=int, default=512, help='embedding dimensions (default: 128)'
    )
    parser.add_argument('--use_info', type=str, default=True)
    parser.add_argument('--use_pretrain_emb', type=str, default=True)

    parser.add_argument(
        '--seed', type=int, default=42, help="Seed for splitting dataset."
    )
    parser.add_argument(
        '--runseed', type=int, default=0, help="Seed for running experiments."
    )
    parser.add_argument(
        '--model_file_path',
        type=str,
        default='model/',
        help='filename to output the pre-trained model',
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
    dataset = DataLoaderFinetune(dPath=dataset_path)
    # list 格式
    pretrained_emb_path_1 = 'pretrain_emb/pretrained_emb_dict_6_256_GAT.pkl'
    # numpy 格式
    pretrained_emb_path_2 = 'pretrain_emb/pretrained_emb_dict_SRRSC_1000.pkl'

    # with open(pretrained_emb_path_1, 'rb') as f1:
    #     node_emb_dict_1 = pickle.load(f1)
    #     # print(node_emb_dict['GENE']['Rbm47'])
    #     with open(pretrained_emb_path_2, 'rb') as f2:
    #         node_emb_dict_2 = pickle.load(f2)

    #         # 把两者拼接起来：
    #         for node_type in node_emb_dict_1:
    #             for node_name in node_emb_dict_1[node_type]:
    #                 # 要先转换以下才能拼接
    #                 temp = node_emb_dict_2[node_type][node_name].tolist()
    #                 node_emb_dict_1[node_type][node_name] = node_emb_dict_1[node_type][node_name] + temp

    #         pretrained_emb_path = 'pretrain_emb/pretrained_emb_dict_joint.pkl'
    #         with open(pretrained_emb_path, 'wb') as f:
    #             pickle.dump(node_emb_dict_1, f, pickle.HIGHEST_PROTOCOL)

    pretrained_emb_path = 'pretrain_emb/pretrained_emb_dict_joint.pkl'
    graph, idx_node_map, idx_node_id_map = dataset.to_graph(
        emb_dim=args.emb_dim,
        use_info=args.use_info,
        use_pretrain_emb=args.use_pretrain_emb,
        pretrained_emb_path=pretrained_emb_path,
    )

    # Heterogeneous graph -> Homogeneous graph
    homo_g = dgl.to_homogeneous(graph, ndata=['feature'])

    (
        test_pos_graph,
        test_neg_graph,
        val_pos_graph,
        val_neg_graph,
        train_pos_graph,
        train_neg_graph,
    ) = split_train_test(graph=homo_g, test_ratio=0.2, val_ratio=0.2, device=device)

    model = MLPPredictor(h_feats=args.emb_dim)
    model.to(device)
    best_model_path = 'best_MLP_model.pt'

    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=args.decay)

    # 1. train
    train_loss, val_auc, best_accuracy = train(
        args,
        model,
        optimizer,
        train_pos_graph,
        train_neg_graph,
        val_pos_graph,
        val_neg_graph,
        best_model_path,
    )
    print("final train loss is {}".format(train_loss[-1]))
    print("final val auc is {}".format(val_auc[-1]))
    print("best val auc is {}".format(best_accuracy))

    # 2. visualization
    # plot(train_loss, val_auc)

    # 3. test
    model = torch.load(best_model_path)
    test_auc = LinkPredictionEvaluate(model, test_pos_graph, test_neg_graph)
    print(f'test auc is: {test_auc}')


if __name__ == "__main__":
    main()

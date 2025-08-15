import argparse
import os.path as osp
import random
from typing import Dict
import torch
from torch_geometric.utils import to_networkx
import os
from src import *
from src.models import Encoder, Model
import copy
from tqdm import tqdm
import networkx as nx
import numpy as np
import json

def train(epoch: int) -> int:
    model.train()
    optimizer.zero_grad()

    edge_index_1 = ced(data.edge_index, data.edge_weight, p=param['ced_drop_rate_1'], threshold=args.ced_thr)
    edge_index_2 = ced(data.edge_index, data.edge_weight, p=param['ced_drop_rate_2'], threshold=args.ced_thr)

    x1 = cav(data.x, node_weight, param["cav_drop_rate_1"], max_threshold=args.cav_thr)
    x2 = cav(data.x, node_weight, param['cav_drop_rate_2'], max_threshold=args.cav_thr)

    z1 = model(x1, edge_index_1)
    z2 = model(x2, edge_index_2)
    
    loss = model.loss(z1, z2, batch_size=0)

    loss.backward() 
    optimizer.step() 
    return loss.item()

def test() -> Dict:
    model.eval() 
    with torch.no_grad():
        z = model(data.x, data.edge_index)
    
    res = {}
    seed = np.random.randint(0, 32767)
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1,
                           generator=torch.Generator().manual_seed(seed))
    evaluator = MulticlassEvaluator()

    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            cls_acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}',
                                     num_epochs=800)
            accs.append(cls_acc['acc'])
        acc = sum(accs) / len(accs)
    else:
        cls_acc = log_regression(z, dataset, evaluator, split='rand:0.1',
                                 num_epochs=3000, preload_split=split)
        acc = cls_acc['acc']
        f1 = cls_acc['f1']
    res["acc"] = acc
    res["f1"] = f1
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--dataset_path', type=str, default="./datasets")
    parser.add_argument('--param', type=str, default='local:cora.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--verbose', type=str, default='train,eval')
    parser.add_argument('--cls_seed', type=int, default=12345)
    parser.add_argument('--val_interval', type=int, default=100)
    parser.add_argument('--ced_thr', type=float, default=1.0)
    parser.add_argument('--cav_thr', type=float, default=1.0)
    parser.add_argument('--dth', default=0.01, help='dismantling_threshold')
    parser.add_argument('--sort_strategy', default='default', choices=['default', 'quick'])  
    parser.add_argument('--perturb_strategy', default='default', choices=['default', 'remove'])  
    parser.add_argument('--belta', default=None)

    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'ced_drop_rate_1': 0.3,
        'ced_drop_rate_2': 0.4,
        'cav_drop_rate_1': 0.1,
        'cav_drop_rate_2': 0.2,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        'n_rules':5,
    }
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    if not args.device == 'cpu':
        args.device = 'cuda'

    print(f"training settings: \n"
          f"data: {args.dataset}\n"
          f"device: {args.device}\n"
          f"batch size if used: {args.batch_size}\n"
          f"ced rate: {param['ced_drop_rate_1']}/{param['ced_drop_rate_2']}\n"
          f"cav rate: {param['cav_drop_rate_1']}/{param['cav_drop_rate_2']}\n"
          f"epochs: {param['num_epochs']}\n"
          f"tau: {param['tau']}\n"
          f"weight_decay: {param['weight_decay']}\n"
          f"learning_rate: {param['learning_rate']}\n"
          )

    random.seed(12345)
    torch.manual_seed(args.seed)
    # for node classification branch
    if args.cls_seed is not None:
        np.random.seed(args.cls_seed)
    device = torch.device(args.device)
    path = osp.join(args.dataset_path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0] 
    data = data.to(device)
    netname = args.dataset
    data.edge_weight = torch.zeros(data.edge_index.size(1),dtype=torch.float)
    
    with open(f'results/{netname}/VE_value.txt', 'r') as file:
        node_weight_lines = file.readlines()
    node_weight_str = [float(line.strip()) for line in node_weight_lines]
    node_weight_array = np.array(node_weight_str, dtype=np.float32)
    node_weight =torch.from_numpy(node_weight_array).to(data.edge_index.device)

    with open(f'results/{netname}/lg_VE_value.txt', 'r') as file:
        edge_weight_lines = file.readlines()
    edge_weight_str = [float(line.strip()) for line in edge_weight_lines]
    edge_weight_array = np.array(edge_weight_str)

    g = to_networkx(data, to_undirected=True)
    l_g = nx.line_graph(g)
    node_mapping = {}  
    new_edgeweight_id = 0
    edges_old = list(l_g.edges)
    for edge in edges_old:
        if edge[0] != edge[1]:  
            if edge[0] not in node_mapping:
                node_mapping[edge[0]] = edge_weight_array[new_edgeweight_id]
                new_edgeweight_id += 1
            if edge[1] not in node_mapping:
                node_mapping[edge[1]] = edge_weight_array[new_edgeweight_id]
                new_edgeweight_id += 1

    for value, key in node_mapping.items():
        value1,value2=value
        key=float(key)
        i=0
        while i<data.edge_index.size(1):
            if data.edge_index[0,i]==value1 and data.edge_index[1,i]==value2:
                data.edge_weight[i]= key
            if data.edge_index[0,i]==value2 and data.edge_index[1,i]==value1:
                data.edge_weight[i] =key
            i=i+1

    encoder = Encoder(dataset.num_features,
                      param['num_hidden'],
                      get_activation(param['activation']),
                      param['n_rules']
                      ).to(device)
    model = Model(encoder,
                  param['num_hidden'],
                  param['num_proj_hidden'],
                  param['tau']).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=param['learning_rate'],
                                 weight_decay=param['weight_decay'])
    last_epoch = 0
    log = args.verbose.split(',')

    best_acc = 0
    best_f1 = 0
    best_embeddings = None
    for epoch in range(1 + last_epoch, param['num_epochs'] + 1):
        loss = train(epoch) 
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
        if epoch % args.val_interval == 0:
            res = test()
            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_acc = {res["acc"]},avg_f1 = {res["f1"]}')

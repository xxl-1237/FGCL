import argparse
import os.path as osp
import random
from typing import Dict

import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import to_networkx
from tqdm import tqdm
import os
import time
import copy

from src import *
from src.models import Encoder, Model
from src.eval import label_classification


TRAIN_VAL_SPLIT_SEED = 12345

def generate_beta(lambda2):
    gap = 100
    a = list(np.linspace(0, 5 / lambda2, gap))
    b = list(np.linspace(a[-1], 10 / lambda2, gap))
    return a[0:-1] + b


def VertexEnt(G, belta=None, perturb_strategy='default'):
    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).todense()
    assert 0 in G, "Node 0 should be in the input graph!"
    assert np.allclose(A, A.T), "adjacency matrix should be symmetric"
    L = np.diag(np.array(sum(A)).flatten()) - A
    N = G.number_of_nodes()

    start_eig = time.time()
    eigValue, eigVector = np.linalg.eigh(L)
    print("Finish calculating eigen values!")
    eigValue = eigValue.real
    eigVector = eigVector.real
    eig_time = time.time() - start_eig
    print(f"[Q1 VE Timing] Eigen decomposition: {eig_time:.2f} seconds")

    if belta is None:
        num_components = nx.number_connected_components(G)
        belta = generate_beta(eigValue[num_components])

    S = np.zeros(len(belta))
    for i in range(len(belta)):
        b = belta[i]
        Z_ = np.sum(np.exp(eigValue * -b))
        t = -b * eigValue
        S[i] = -sum(np.exp(t) * (t / np.log(2) - np.log2(Z_)) / Z_)

    lambda_ral = np.zeros((N, N))
    if perturb_strategy == 'default':
        for v_id in tqdm(range(N), desc="Computing eigenValues", unit="node"):
            neibour = list(G.neighbors(v_id))
            kx = G.degree(v_id)
            A_loc = A[neibour][:, neibour]
            N_loc = kx + np.sum(A_loc) / 2
            weight = 2 * N_loc / kx / (kx + 1)
            if weight == 1:
                lambda_ral[v_id] = eigValue
            else:
                neibour.append(v_id)
                neibour = sorted(neibour)
                dA = weight - A[neibour][:, neibour]
                dA = dA - np.diag([weight] * (kx + 1))
                dL = np.diag(np.array(sum(dA)).flatten()) - dA
                for j in range(N):
                    t__ = eigVector[neibour, j].T @ dL @ eigVector[neibour, j]
                    lambda_ral[v_id, j] = eigValue[j] + (
                        t__[0, 0] if isinstance(t__, np.ndarray) else t__
                    )
    elif perturb_strategy == 'remove':
        for v_id in tqdm(range(N), desc="Computing eigenValues for removed networks", unit="node"):
            neibour = list(G.neighbors(v_id))
            pt_A = copy.deepcopy(A)
            pt_A[v_id, :] = 0
            pt_A[:, v_id] = 0
            dA = pt_A - A
            dA = dA[neibour][:, neibour]
            dL = np.diag(np.array(sum(dA)).flatten()) - dA
            for j in range(N):
                lambda_ral[v_id, j] = eigValue[j] + (
                    eigVector[neibour, j].T @ dL @ eigVector[neibour, j]
                )[0, 0]

    E = np.zeros((len(belta), N))
    for x in tqdm(range(N), desc="Searching minimum entanglement", unit="node"):
        xl_ = lambda_ral[x, :]
        for i in range(len(belta)):
            b = belta[i]
            Z_ = np.sum(np.exp(-b * xl_))
            t = -b * xl_
            E[i, x] = -sum(np.exp(t) * (t / np.log(2) - np.log2(Z_)) / Z_) - S[i]

    VE = np.min(E, axis=0)
    return VE


def train(epoch: int) -> float:
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


def evaluate(n_repeats: int = 1, random_state=TRAIN_VAL_SPLIT_SEED) -> Dict:
 
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

    res = label_classification(
        z, data.y,
        ratio=0.1, val_ratio=0.1,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    return {
        "val_acc":  res["val_ACC"]["mean"],
        "val_f1":   res["val_F1Ma"]["mean"],
        "test_acc": res["test_ACC"]["mean"],
        "test_f1":  res["test_F1Ma"]["mean"],
        "raw": res,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--dataset_path', type=str, default="./datasets")
    parser.add_argument('--param', type=str, default='local:cora.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--verbose', type=str, default='train,eval')
    parser.add_argument('--val_interval', type=int, default=100)
    parser.add_argument('--ced_thr', type=float, default=1.0)
    parser.add_argument('--cav_thr', type=float, default=1.0)
    parser.add_argument('--perturb_strategy', default='default', choices=['default', 'remove'])
    parser.add_argument('--belta', default=None)
    parser.add_argument('--n_clusters', type=int, default=7)

    default_param = {
        'learning_rate': 0.01, 'num_hidden': 256, 'num_proj_hidden': 32,
        'activation': 'prelu', 'ced_drop_rate_1': 0.3, 'ced_drop_rate_2': 0.4,
        'cav_drop_rate_1': 0.1, 'cav_drop_rate_2': 0.2, 'tau': 0.4,
        'num_epochs': 3000, 'weight_decay': 1e-5, 'n_rules': 5,
    }
    for key in default_param.keys():
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')

    args = parser.parse_args()
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')
    for key in default_param.keys():
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    if args.device != 'cpu' and not torch.cuda.is_available():
        print("[Warning] CUDA requested but not available, falling back to CPU.")
        args.device = 'cpu'
    elif args.device != 'cpu':
        args.device = 'cuda'

    print(f"Training settings: data={args.dataset}, device={args.device}, n_clusters(C)={args.n_clusters}")

    random.seed(12345)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    path = osp.join(args.dataset_path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0].to(device)
    netname = args.dataset
    data.edge_weight = torch.zeros(data.edge_index.size(1), dtype=torch.float, device=device)

    results_path = os.path.join('results', netname)
    os.makedirs(results_path, exist_ok=True)
    ve_value_path = os.path.join(results_path, 'VE_value.txt')

    if not os.path.exists(ve_value_path):
        G = to_networkx(data.cpu(), to_undirected=True)
        VE = VertexEnt(G, belta=args.belta, perturb_strategy=args.perturb_strategy)
        np.savetxt(ve_value_path, VE, fmt='%.16f')
    else:
        VE = np.loadtxt(ve_value_path)

    node_weight = torch.from_numpy(np.array(VE, dtype=np.float32)).to(device)

    with open(f'results/{netname}/lg_VE_value.txt', 'r') as file:
        edge_weight_str = [float(line.strip()) for line in file.readlines()]
    edge_weight_array = np.array(edge_weight_str)

    g = to_networkx(data.cpu(), to_undirected=True)
    l_g = nx.line_graph(g)
    node_mapping = {}
    new_edgeweight_id = 0
    for edge in list(l_g.edges):
        if edge[0] != edge[1]:
            if edge[0] not in node_mapping:
                node_mapping[edge[0]] = edge_weight_array[new_edgeweight_id]
                new_edgeweight_id += 1
            if edge[1] not in node_mapping:
                node_mapping[edge[1]] = edge_weight_array[new_edgeweight_id]
                new_edgeweight_id += 1

    for value, key in node_mapping.items():
        value1, value2 = value
        key = float(key)
        i = 0
        while i < data.edge_index.size(1):
            if ((data.edge_index[0, i] == value1 and data.edge_index[1, i] == value2) or
                    (data.edge_index[0, i] == value2 and data.edge_index[1, i] == value1)):
                data.edge_weight[i] = key
            i += 1

    # ==================== Model initialization ====================
    encoder = Encoder(dataset.num_features,
                      param['num_hidden'],
                      get_activation(param['activation']),
                      param['n_rules'],
                      n_clusters=args.n_clusters).to(device)

    model = Model(encoder,
                  param['num_hidden'],
                  param['num_proj_hidden'],
                  param['tau']).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=param['learning_rate'],
                                 weight_decay=param['weight_decay'])

    # ==================== Training loop ====================
    last_epoch = 0
    log = args.verbose.split(',')

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_test_acc = 0.0
    best_test_f1 = 0.0
    best_state = None
    best_epoch = 0

    for epoch in range(1 + last_epoch, param['num_epochs'] + 1):
        loss = train(epoch)
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

        if epoch % args.val_interval == 0:
            res = evaluate(n_repeats=1, random_state=TRAIN_VAL_SPLIT_SEED)

            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d} | '
                      f'val_acc={res["val_acc"]:.4f}, val_f1={res["val_f1"]:.4f} | '
                      f'test_acc={res["test_acc"]:.4f}, test_f1={res["test_f1"]:.4f}')

            if res["val_acc"] > best_val_acc:
                best_val_acc = res["val_acc"]
                best_val_f1 = res["val_f1"]
                best_test_acc = res["test_acc"]
                best_test_f1 = res["test_f1"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                if 'eval' in log:
                    print(f'    -> New best at epoch {epoch} '
                          f'(val_acc={best_val_acc:.4f}), state saved.')

    print(f'(---------------) best_epoch = {best_epoch}')
    print(f'(---------------) best_val_acc = {best_val_acc:.4f}, best_val_f1 = {best_val_f1:.4f}')
    print(f'(---------------) corresponding test_acc = {best_test_acc:.4f}, test_f1 = {best_test_f1:.4f}')

    print("\n=== Final Evaluation using best model (50 repeats, different splits) ===")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        final_z = model(data.x, data.edge_index)

    final_nc = label_classification(
        final_z, data.y,
        ratio=0.1, val_ratio=0.1,
        n_repeats=50, random_state=None,
    )

    print(f'Final Test       (50 different splits): '
          f'ACC = {final_nc["test_ACC"]["mean"]:.4f}±{final_nc["test_ACC"]["std"]:.4f}, '
          f'F1Ma = {final_nc["test_F1Ma"]["mean"]:.4f}±{final_nc["test_F1Ma"]["std"]:.4f}')
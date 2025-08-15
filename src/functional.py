import numpy as np
import networkx as nx
import torch
from typing import Sequence
from cdlib import algorithms
from cdlib.utils import convert_graph_formats
from torch_geometric.utils import  subgraph

def ced(edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        p: float,
        threshold: float = 1.) -> torch.Tensor:
    edge_weight = abs(edge_weight)  
    edge_weight = edge_weight / edge_weight.mean() * (1. - p) 
    edge_weight = edge_weight.where(edge_weight > (1. - threshold), torch.ones_like(edge_weight) * (1. - threshold))
    edge_weight = edge_weight.where(edge_weight < 1, torch.ones_like(edge_weight) * 1)
    sel_mask = torch.bernoulli(edge_weight).to(torch.bool)
    return edge_index[:, sel_mask]

def cav(feature: torch.Tensor,
        node_weight: np.ndarray,
        p: float,
        max_threshold: float = 0.7) -> torch.Tensor:
    x = feature.abs()
    node_cs = node_weight.abs()  
    device = feature.device
    w = x.t() @ torch.tensor(node_cs).to(device)
    w[torch.nonzero(w == 0)] = w.max()  
    w = w.log()
    w = (w.max() - w) / (w.max() - w.min())
    w = w / w.mean() * p
    w = w.where(w < max_threshold, max_threshold * torch.ones(1).to(device))
    w = w.where(w > 0, torch.zeros(1).to(device))
    drop_mask = torch.bernoulli(w).to(torch.bool)
    feature = feature.clone()
    feature[:, drop_mask] = 0.
    return feature

def transition(communities: Sequence[Sequence[int]],
               num_nodes: int) -> np.ndarray:
    classes = np.full(num_nodes, -1)
    for i, node_list in enumerate(communities):
        classes[np.asarray(node_list)] = i
    return classes

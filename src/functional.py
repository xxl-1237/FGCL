import torch

def ced(edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        p: float,
        threshold: float = 1.) -> torch.Tensor:
    edge_weight = abs(edge_weight)
    edge_weight = edge_weight / edge_weight.mean() * (1. - p)
    edge_weight = edge_weight.where(
        edge_weight > (1. - threshold),
        torch.ones_like(edge_weight) * (1. - threshold)
    )
    edge_weight = edge_weight.where(edge_weight < 1, torch.ones_like(edge_weight) * 1)
    sel_mask = torch.bernoulli(edge_weight).to(torch.bool)
    return edge_index[:, sel_mask]


def cav(feature: torch.Tensor,
        node_weight: torch.Tensor,
        p: float,
        max_threshold: float = 0.7) -> torch.Tensor:
    x = feature.abs()
    # node_weight must be a torch tensor on the same device as `feature`
    if not isinstance(node_weight, torch.Tensor):
        node_weight = torch.as_tensor(node_weight, dtype=feature.dtype)
    node_cs = node_weight.abs().to(feature.device)

    w = x.t() @ node_cs
    w[torch.nonzero(w == 0, as_tuple=False)] = w.max()
    w = w.log()
    w = (w.max() - w) / (w.max() - w.min())
    w = w / w.mean() * p
    w = w.where(w < max_threshold, max_threshold * torch.ones(1, device=feature.device))
    w = w.where(w > 0, torch.zeros(1, device=feature.device))
    drop_mask = torch.bernoulli(w).to(torch.bool)

    feature = feature.clone()
    feature[:, drop_mask] = 0.
    return feature
from src.Model import *
import torch
import torch.nn.functional as F
import numpy as np
import faiss
from typing import Optional


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, n_rules: int,
                 n_clusters: Optional[int] = None, base_model=DFGCN):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.n_rules = n_rules
        self.n_clusters = n_clusters

        self.base_model = base_model(self.in_channels, out_channels * 2, out_channels, 128, 60)

    def calculate_edge_weights(self, x, edge_index):
        """Compute edge weights based on soft K-means clustering similarity."""
        if self.n_clusters is None:
            n_clusters = int(np.sqrt(x.shape[0]))
        else:
            n_clusters = self.n_clusters

        niter = 20
        sigma = 1.0

        # faiss operates on CPU; this is expected
        kmeans = faiss.Kmeans(x.shape[1], n_clusters, niter=niter, verbose=False)
        kmeans.train(x.detach().cpu().numpy())
        centroids = torch.as_tensor(kmeans.centroids, dtype=x.dtype, device=x.device)

        # Compute soft clustering probabilities
        logits = []
        for c in centroids:
            dist = -torch.square(x - c).sum(1) / sigma
            logits.append(dist.view(-1, 1))

        logits = torch.cat(logits, dim=1)
        probs = F.softmax(logits, dim=1)  # [num_nodes, n_clusters]

        # Vectorized cosine similarity over all edges (was a Python loop)
        src, dst = edge_index[0], edge_index[1]
        edge_weights = F.cosine_similarity(probs[src], probs[dst], dim=1)

        return edge_weights

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        edge_weight = self.calculate_edge_weights(x, edge_index)
        x = self.base_model(x, edge_index, edge_weight)
        return x


class Model(nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        )

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int, between=True):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))
            if between is True:
                between_sim = f(self.sim(z1[mask], z2))
            else:
                between_sim = 0

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret
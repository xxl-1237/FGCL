import torch
import torch.nn as nn
import torch.nn.init as Init
import torch.nn.functional as F

# TSK FIS
class TSK(nn.Module):
    def __init__(self, in_dim, n_rules, output_dim1, output_dim2=0, ante_ms_shape='gaussian', fz=False):

        super(TSK, self).__init__()
        self.in_dim = in_dim
        self.n_rules = n_rules
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.eps = 1e-15

        self.ante_ms_shape = ante_ms_shape

        self.fz = fz  # whether do fuzzified operation

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.build_model()
        self.to(self.device)

    def build_model(self):

        if self.fz == True:
            self.fz_weight = nn.Parameter(torch.FloatTensor(size=(1, self.n_rules)), requires_grad=True)
            Init.uniform_(self.fz_weight, 0, 1)

            if self.ante_ms_shape == 'gaussian':
                self.Cs = nn.Parameter(torch.FloatTensor(size=(self.in_dim, self.n_rules)), requires_grad=True)
                self.Vs = nn.Parameter(torch.FloatTensor(size=self.Cs.size()), requires_grad=True)

    def forward(self, x):
        frs, _ = self.fuzzify(x)

    def init_center(self, init_centers, init_Vs=None):

        self.init_centers = init_centers
        self.init_Vs = init_Vs

        if self.ante_ms_shape == 'gaussian':
            self.Cs.data = torch.from_numpy(self.init_centers).float()
            if self.init_Vs is not None:
                self.Vs.data = torch.from_numpy(self.init_Vs).float()
            else:
                Init.normal_(self.Vs, mean=1, std=0.2)

    def fuzzify(self, features):
        fz_features = None
        self.to(self.device)
        if self.ante_ms_shape == 'gaussian':
            fz_degree = -(features.unsqueeze(dim=2) - self.Cs) ** 2 / ((2 * self.Vs ** 2) + self.eps)
            if self.fz == True:
                fz_degree = torch.exp(fz_degree)
                weighted_fz_degree = torch.max(fz_degree, dim=2)[0]
                fz_features = torch.mul(features, weighted_fz_degree)  #

        return fz_degree, fz_features



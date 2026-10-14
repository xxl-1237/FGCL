from Mamdani.lib.models import *
from Mamdani.lib.inits import *
from torch_geometric.nn import GCNConv
import torch.nn as nn


class DFGCN(nn.Module):
    def __init__(self, input, hidden_1, hidden_2, output, n_rules,
                 dropout=0, init='kmean', mm_shape='gaussian'):
        super(DFGCN, self).__init__()
        self.output = output
        self.n_rules = n_rules
        self.init = init
        self.mm_shape = mm_shape

        self.hidden1 = hidden_1
        self.hidden2 = hidden_2
        self.gc1 = GCNConv(input, self.hidden1)
        self.gc2 = GCNConv(self.hidden1, self.hidden2)

        self.rule_1 = self.n_rules
        self.fls_1 = TSK(self.hidden1, self.rule_1, self.hidden2,
                         ante_ms_shape=self.mm_shape, fz=True)
        self.first_1 = True

        self.rule_2 = self.n_rules
        self.fls_2 = TSK(self.hidden2, self.rule_2, self.output,
                         ante_ms_shape=self.mm_shape, fz=True)
        self.first_2 = True

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_weight):
        x = self.dropout(x)

        x = self.gc1(x, edge_index, edge_weight)
        if self.first_1:
            x_train = x.detach().cpu().numpy()
            if self.init == 'kmean':
                cs, vs = kmean_init(x_train, self.rule_1)
            elif self.init == 'fcm':
                cs, vs = fcm_init(x_train, self.rule_1)
            else:
                raise ValueError(f"Unknown init method: {self.init}")
            self.fls_1.init_center(cs, vs)
            self.first_1 = False

        _, x = self.fls_1.fuzzify(x)

        x = self.dropout(x)
        x = self.gc2(x, edge_index, edge_weight)

        if self.first_2:
            x_train = x.detach().cpu().numpy()
            if self.init == 'kmean':
                cs, vs = kmean_init(x_train, self.rule_2)
            elif self.init == 'fcm':
                cs, vs = fcm_init(x_train, self.rule_2)
            else:
                raise ValueError(f"Unknown init method: {self.init}")
            self.fls_2.init_center(cs, vs)
            self.first_2 = False

        _, x = self.fls_2.fuzzify(x)

        return x
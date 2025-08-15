from Mamdani.lib.models import *
from Mamdani.lib.inits import *
from torch_geometric.nn import GCNConv
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, input, hidden1, hidden2, act=nn.ReLU, dropout=0):
        super(GCN, self).__init__()
        self.input = input
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.act = act
        self.dropout = nn.Dropout(p=dropout)

        self.gcn1 = GCNConv(input, hidden1)
        self.gcn2 = GCNConv(hidden1, hidden2)

    def forward(self, x, edge_index, edge_weight):
        x = self.dropout(x)
        x = self.gcn1(x, edge_index, edge_weight)
        x = self.act(x)

        x = self.gcn2(x, edge_index, edge_weight)
        x = self.act(x)

        return x


class DWFGCN(nn.Module):
    def __init__(self, input, hidden_1, hidden_2, output, n_rules, dropout=0, ampli=0, init='kmean', mm_shape='gaussian'):
        super(DWFGCN, self).__init__()
        self.output = output
        self.n_rules = n_rules
        self.ampli = ampli
        self.init = init
        self.mm_shape = mm_shape

        self.hidden1 = hidden_1
        self.hidden2 = hidden_2
        self.gc1 = GCNConv(input, self.hidden1)
        self.gc2 = GCNConv(self.hidden1, self.hidden2)
        self.gc3 = GCNConv(self.hidden2, self.output)

        self.rule_1 = self.n_rules
        self.fls_1 = TSK(self.hidden1, self.rule_1, self.hidden2, ante_ms_shape=self.mm_shape, fz=True)
        self.first_1 = True

        self.rule_2 = self.n_rules
        self.fls_2 = TSK(self.hidden2, self.rule_2, self.output, ante_ms_shape=self.mm_shape, fz=True)
        self.first_2 = True
        self.mul = nn.Linear(self.hidden2, self.output, bias=True)

        self.rule_3 = self.n_rules
        self.fls_3 = TSK(self.output, self.rule_3, self.output, ante_ms_shape=self.mm_shape, fz=True)
        self.first_3 = True

        self.act = nn.ReLU(True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_weight):
        x = self.dropout(x)

        x = self.gc1(x, edge_index, edge_weight)
        if self.first_1:
            x_train = x.cpu().detach().numpy()
            if self.init == 'kmean':
                cs, vs = kmean_init(x_train, self.rule_1)
            elif self.init == 'fcm':
                cs, vs = fcm_init(x_train, self.rule_1)
            else:
                exit()
            self.fls_1.init_center(cs, vs)
            self.first_1 = False

        _, x = self.fls_1.fuzzify(x)

        x = self.dropout(x)
        # x = self.act(x)
        x = self.gc2(x, edge_index)

        if self.first_2:
            x_train = x.cpu().detach().numpy()
            if self.init == 'kmean':
                cs, vs = kmean_init(x_train, self.rule_2)
            elif self.init == 'fcm':
                cs, vs = fcm_init(x_train, self.rule_2)
            else:
                exit()
            self.fls_2.init_center(cs, vs)

            self.first_2 = False

        _, x = self.fls_2.fuzzify(x) 
        
        return x
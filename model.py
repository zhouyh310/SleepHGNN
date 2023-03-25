from layers import HGT
from util import count_parameters
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, in_dim, emb_dim, out_dim, n_node_types, n_relation_types, n_heads=8, n_node_in_graph=11, dropout=0.2,
                 n_HGTs=3, lin_dims=None, lin_dropout=0.2):
        super().__init__()

        self.emb = nn.Linear(in_dim, emb_dim) 

        self.lin_dims = lin_dims
        self.lin_dropout = lin_dropout

        self.HGTs = nn.ModuleList()
        for _ in range(n_HGTs):
            self.HGTs.append(HGT(emb_dim, emb_dim, n_node_types, n_relation_types, n_heads=n_heads, dropout=dropout))

        readout_dim = n_node_in_graph * emb_dim

        cur_dim = readout_dim
        self.lins = nn.ModuleList()
        self.BNs = nn.ModuleList()
        for lin_dim in self.lin_dims:
            self.lins.append(nn.Linear(cur_dim, lin_dim))
            self.BNs.append(nn.BatchNorm1d(lin_dim))
            cur_dim = lin_dim

        self.classifier = nn.Linear(lin_dims[-1], out_dim)

    def forward(self, graph):
        res = self.emb(graph.x)

        for hgt in self.HGTs:
            res = hgt(res, graph.edge_index, graph.node_type, graph.edge_type)

        out = res.view(graph.num_graphs, -1)

        for lin, bn in zip(self.lins, self.BNs):
            out = lin(out)
            out = bn(F.leaky_relu(out))
            out = F.dropout(out, p=self.lin_dropout)

        out = self.classifier(out)

        return F.log_softmax(out, dim=-1)

    def reset_parameters(self):
        self.emb.reset_parameters()     
        for hgt in self.HGTs:
            hgt.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.BNs:
            bn.reset_parameters()
        self.classifier.reset_parameters()

    def show_parameter_num(self):
        cnt = 0
        cnt += count_parameters(self.emb, 'emb')

        for i, h in enumerate(self.HGTs):
            cnt += count_parameters(h, f'hgt{i}')

        for i in range(len(self.lins)):
            cnt += count_parameters(self.lins[i], f'lin{i + 1}')
            
        cnt += count_parameters(self.classifier, 'classifier')

        print(f'{cnt:,} total parameters.')
        print('----------------------------------------')

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from .inits import uniform

class MyNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, nofcommu, normalize=False, bias=True, 
        gn_lambda =0.0, edge_update = False, **kwargs):
        super(MyNNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nofcommnu = nofcommu
        self.normalize = normalize # normalize is set to False when initializing LI_Net
        
        self.linear = torch.nn.Linear(in_channels, out_channels * nofcommu, bias=False)
        
        self.group_bn   = torch.nn.ModuleList([torch.nn.BatchNorm1d(out_channels) for _ in range(nofcommu)])
        self.gn_lambda  = gn_lambda if gn_lambda > torch.finfo().eps else 0.0
        self.group_norm = (gn_lambda > torch.finfo().eps)
        self.edge_update = edge_update
        if edge_update:
            #self.KeyMap = torch.nn.Linear(out_channels, out_channels)
            self.QueryMap = torch.nn.Linear(out_channels, out_channels)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
#        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)
        for m in self.modules():
            if isinstance(m,torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x, edge_index,  edge_weight=None, commu_attn = None, size=None):
        if edge_weight is not None:
            edge_weight = edge_weight.squeeze()
       
        x = self.linear(x).view(-1, self.nofcommnu, self.out_channels) 
        if commu_attn is not None :
            x = x * commu_attn.unsqueeze(-1)

        if self.group_norm:
            z = x.transpose(0,1)
            tmp = [self.group_bn[i](z[i]) for i in range(self.nofcommnu)]
            z = torch.stack(tmp,1)
            x = (1-self.gn_lambda)* x + self.gn_lambda * z
        
        x = torch.sum(x,dim=1) 
        
        row, col = edge_index
        if self.edge_update:
            keyX   = x 
            queryX = self.QueryMap(x)
            attnX  = torch.sigmoid(torch.sum(keyX[row] * queryX[col],1))
            edge_weight = attnX
            norm = None
        else:
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight, norm=norm)

    def message(self, x_j, edge_weight, norm):
        if edge_weight is not None:
            x_j = (edge_weight.view(-1,1) * x_j if len(edge_weight.shape) == 1 else
                    torch.sum(edge_weight.unsqueeze(1) * x_j.unsqueeze(-1), dim=-1))
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out, x):
        aggr_out = aggr_out + x 
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


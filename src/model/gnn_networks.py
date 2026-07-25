import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import MemPooling,  MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from .MyTopK import TopKPooling
from .MySAG import SAGPooling
from .graphconv import MyNNConv

import numpy as np
import functools
import math
from utils.torch_history import TorchHistory 

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, num_edge_type,aggr="add"):
        super(GINConv, self).__init__()
        self.num_edge_type = num_edge_type
        # Multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim))
        #self.edge_embedding = torch.nn.Embedding(num_edge_type, emb_dim)
        self.edge_embedding =  torch.nn.Linear(num_edge_type,emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        #torch.nn.init.kaiming_normal_(self.edge_embedding.weight, mode='fan_out') # weight or weight.data
        self.aggr = aggr

    def forward(self, x, edge_index, edge_weight):
        if len(edge_weight.shape) == 1:
            edge_index, edge_weight = add_remaining_self_loops(
                    edge_index, edge_weight, 1, x.size(0))
        else:
            edge_weight0 = torch.stack([ add_remaining_self_loops(
                    edge_index, edge_weight[:,i], 1, x.size(0))[1]
                for i in range(edge_weight.shape[1]) ], dim=1)
            edge_index,_ = add_remaining_self_loops(
                    edge_index, edge_weight[:,0], 1, x.size(0))
            edge_weight = edge_weight0
        edge_embeddings = self.edge_embedding(edge_weight)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

def choose_func(actype,bnfg,bn,bn_dim, fc,leaky_nslope=0.7):
    batch_norm_first = False 
    fn_dict ={0:F.tanh, 
            1:F.selu,
            2:F.relu, 
            3:functools.partial(F.leaky_relu,negative_slope =leaky_nslope)}
    def _func(x,batch = None):
        act_fn = fn_dict.get(actype, None) 
        assert act_fn != None, 'not an activation funtion selector'
        x = fc(x)
        if bnfg == 1 and batch_norm_first:
            x = bn(x.view(-1,bn_dim))
        x = act_fn(x)
        if bnfg == 1 and not batch_norm_first:
            x = bn(x.view(-1,bn_dim))
        return x 
    return _func

   
class GnnBlock(torch.nn.Module):
    def __init__(self, layer, gIndim, gOdim, nofcommu, pRatio, nofEdgeAttr=1, bn_fg=False,
        pMethod='topk', convmethod='LI_Net', normalize=False, gn_lambda=0.0,
        edge_update=False):
        super(GnnBlock,self).__init__()
        self.indim = gIndim
        self.gnn_Odim = gOdim
        self.nofcommu = nofcommu
        self.pool_ratio = pRatio
        self.poolmethod = pMethod
        self.convmethod = convmethod
        self.bn_fg = bn_fg
        self.layer = layer
        self.nofEdgeAttr = nofEdgeAttr
        self.normalize = normalize
        ######################
        self.pool_attn = False
        self.residual  = True  # residual 1

        if self.convmethod == 'GATConv':
            self.conv = GATConv(self.indim, self.gnn_Odim)
        elif self.convmethod == 'GCNConv':
            self.conv = GCNConv(self.indim, self.gnn_Odim, add_self_loops=False, normalize = self.normalize)
        elif self.convmethod == 'LI_Net':
            self.conv = MyNNConv(self.indim, self.gnn_Odim, self.nofcommu, normalize=False, gn_lambda=gn_lambda, edge_update= edge_update)
        elif self.convmethod == 'GINConv':
            self.conv = GINConv(self.gnn_Odim, num_edge_type=self.nofEdgeAttr)
        else:
            raise 'convmethod not supported' 

        if self.bn_fg:  
            self.gbn_layer = torch.nn.BatchNorm1d(self.gnn_Odim)

        if self.pool_attn:
            self.pool_attn_dim = self.gnn_Odim
            self.pool_attnK = nn.Linear(self.gnn_Odim, self.pool_attn_dim)
            self.pool_attnQ = nn.Linear(self.gnn_Odim, self.pool_attn_dim)
            self.pool_attnV = nn.Linear(self.gnn_Odim, self.gnn_Odim)
        
        if self.poolmethod == 'topk':
            self.pool = TopKPooling(self.gnn_Odim, ratio=self.pool_ratio, multiplier=1, nonlinearity=torch.sigmoid)
        else:
            self.pool = SAGPooling(self.gnn_Odim, ratio=self.pool_ratio, GNN=GATConv,nonlinearity=torch.sigmoid)
        if self.residual and not self.convmethod == 'GINConv':
            self.x_mapping = torch.nn.Linear(self.indim,self.gnn_Odim)

    def forward(self, x, pos, batch, edge_index, edge_attr):
        training_output={}
        x_input = x
        if self.convmethod == 'GATConv':
            x = self.conv(x, edge_index)
        elif self.convmethod == 'GCNConv':
            x = self.conv(x, edge_index, edge_attr.abs())
             
        elif self.convmethod == 'LI_Net':
            x = self.conv(x, edge_index, edge_attr, pos)
        elif self.convmethod == 'GINConv':
            x = self.conv(x, edge_index, edge_attr)
        else:
            raise 'convmethod Not supported'
        x = F.relu(x)
        if self.residual and not self.convmethod == 'GINConv':
            x = x + self.x_mapping(x_input)

        if self.bn_fg:
            x = self.gbn_layer(x.view(-1, self.gnn_Odim))
            
        training_output[f'intgc{self.layer}'] = x
        
       
        if self.pool_ratio == 1: 
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            training_output[f'intpool{self.layer}'] = x
            return x, x1, edge_index, edge_attr, batch, pos, training_output

        if self.pool_attn:
            b = batch[-1]+1
            pool_input = x.view(b,-1,self.gnn_Odim)
            Q = self.pool_attnQ(pool_input)
            K = self.pool_attnK(pool_input)
            V = self.pool_attnV(pool_input)
            sf = torch.sqrt(torch.tensor(self.pool_attn_dim))
            pool_attn = torch.softmax(Q @ K.transpose(1,2)/sf,dim=-1) @ V # dim: batchsize, no_nodes, gnn_Odim
            pool_attn = pool_attn.view(-1,self.gnn_Odim) 
        else:
            pool_attn = None
       
        x, edge_index, edge_attr, batch, perm, score1 = self.pool(x, edge_index, edge_attr, batch, attn=pool_attn)
        
        training_output[f's{self.layer}'] = score1
        if self.convmethod == 'LI_Net':
            pos = pos[perm]  # prepare pos for return, will be used in next gnn layer
        training_output[f'w{self.layer}'] = self.pool.weight   
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        training_output[f'intpool{self.layer}'] = x
        return x, x1, edge_index, edge_attr, batch, pos, training_output

class GnnPlusMempool(torch.nn.Module):
    def __init__(self, gnn_plist, mpool_p, xinput_p, inc_global,
                        predicate_create_shared, sharable_modules, imported_modules,
                        residual,l1_reg = 0.0, p_dropout = 0):
        super(GnnPlusMempool,self).__init__()
        if gnn_plist[0]['convmethod'] == 'GINConv':
            self.residual = False 
        else:
            self.residual = residual 
        self.inc_global = inc_global
        self.gnn_blks = torch.nn.ModuleList([GnnBlock(**gnnp) for gnnp in gnn_plist])
        self.sharable_modules = sharable_modules
        self.mpool_p = mpool_p
        if predicate_create_shared('mempool'):
            if mpool_p is not None:
                out_gcn = gnn_plist[-1]['gOdim']
                if not isinstance(mpool_p,list):
                    self.mempool = MemPooling(out_gcn, **mpool_p)
                else:
                    mempools = []
                    in_dim = out_gcn
                    for mp in mpool_p:
                        mempools.append(MemPooling(in_dim, **mp))
                        in_dim = mp['out_channels']
                    self.mempool = torch.nn.ModuleList(mempools)
            else:
                self.mempool = None
            self.sharable_modules['mempool'] = self.mempool
        else:
            self.sharable_modules['mempool'] = imported_modules['mempool']

        if self.residual and (xinput_p is not None) and (mpool_p is not None):
            xmapping_indim = xinput_p['nofNodes'] * xinput_p['indim']  
            self.nofNodes =  xinput_p['nofNodes']
            if isinstance(mpool_p,dict): 
                xmapping_Odim = mpool_p['num_clusters'] * mpool_p['out_channels']
                self.num_clusters = mpool_p['num_clusters']
            else: # a list of mpool, take the last mempool
                xmapping_Odim = mpool_p[-1]['num_clusters'] * mpool_p[-1]['out_channels']
            self.x_mapping = torch.nn.Sequential(nn.Linear(xmapping_indim, xmapping_Odim), nn.ReLU(), nn.Dropout(p=p_dropout))

    def get_gbn_gamma_list(self):
        gammars =[gnn.gbn_layer.weight for gnn in self.gnn_blks]
        return gammars

    def forward(self, x, pos, batch, edge_index, edge_attr):
        score2, score3, loss3  = 0.0, 0.0, 0.0 
        w1, w2 = 0.0, 0.0
        training_output = {'s2':score2,'s3':score3,'loss3':loss3,'w1':w1,'w2':w2}
        glb_x=[]
        nofSample = batch[-1]+1
        x_input = x.view(nofSample,-1)
        for gnn in self.gnn_blks:
            x, x1, edge_index, edge_attr, batch, pos, xto_dict = gnn(x, pos, batch, edge_index, edge_attr)
            training_output.update(xto_dict)
            glb_x.append(x1)

        mempool = self.sharable_modules['mempool']
        if mempool is not None:
            if isinstance(mempool, MemPooling):
                x, score3  = mempool(x, batch) 
                loss3 = mempool.kl_loss(score3)  
                loss3 = loss3/(self.nofNodes * self.num_clusters) * 2000
                training_output['mp1'] = score3
            else: 
                loss3 = 0.0
                x1 = x
                nofNodes = self.nofNodes
                for idx,m in enumerate(mempool):
                    x, score3 = m(x1,batch)
                    num_clusters = self.mpool_p[idx]['num_clusters']
                    b1 = [[i]*x.shape[1] for i in range(x.shape[0])]
                    batch = torch.tensor(b1,device=x.device).ravel()
                    x1 = x.view(-1,x.shape[2])
                    loss3 = loss3 + m.kl_loss(score3)/(nofNodes * num_clusters) * 2000
                    nofNodes = num_clusters
                    training_output[f'mp{idx+1}'] = score3
            training_output['loss3'] = loss3
            tx = x.view(nofSample,-1) 
            if self.residual:
                tx += self.x_mapping(x_input).view(nofSample,-1)

            if self.inc_global:
                glb_x.append(tx)
                x = torch.cat(glb_x, dim=1) 
            else:
                x = tx
            training_output['intmempool'] = x
        else:
            x = x.view(nofSample,-1)
            if self.inc_global:
                glb_x.append(x)
                x = torch.cat(glb_x, dim=1) 
        return x, training_output

class GphEmbedding(torch.nn.Module):
    def __init__(self, mopt, predicate_create_shared, sharable_modules, imported_modules):
        super(GphEmbedding,self).__init__()
        self.edge_update = False if not hasattr(mopt,'edge_update') else mopt.edge_update
        self.leaky_nslope = 0.7  if not hasattr(mopt,'actfnparm') else mopt.actfnparm
        self.ratio1 = mopt.ratio if not hasattr(mopt,'ratio1')     else mopt.ratio1
        
        self.ratio2 = mopt.ratio
        self.indim  = mopt.indim
        self.dim11  = mopt.dim1   if not hasattr(mopt,'dim11')      else mopt.dim11   #gnn1 (indim-> dim11)
        self.dim12  = mopt.dim12  
        if hasattr(mopt,'dim13'):
            self.dim13 = mopt.dim13 
        self.dimmp = mopt.dimmp 
        self.dim2   = mopt.dim2   
        self.nclass = mopt.nclass
        self.poolmethod = mopt.poolmethod
        self.convmethod = mopt.convmethod
        self.memp   = mopt.memp
        self.heads  = mopt.heads
        self.clusters  = mopt.clusters
        self.normalize = mopt.net_norm
        self.n_gc   = mopt.gc
        self.dim_cv = mopt.dim_cv_list
        self.activation_type = mopt.activationtype 
        self.bn_fg  = mopt.bnfg 
        self.inc_global = mopt.inc_global
        self.dropout = mopt.dropout
        self.k      = mopt.nocommu 
        self.R      = mopt.nogenes 
        #####################################
        self.outdim = None
        self.train_output = None
        # related to atten
        self.attn_type   = 1 
        self.attn_select = 1 
        self.residual    = False 
        self.sharable_modules = sharable_modules
        self.use_nn_lookuptable=mopt.use_nn_lookuptable if hasattr(mopt,'use_nn_lookuptable') and mopt.use_nn_lookuptable  else False
        self.l1_reg = mopt.l1_reg if hasattr(mopt,'l1_reg') else 0.0
        nofEdgeAttr = 1 if not hasattr(mopt, 'nofedgeattr') else mopt.nofedgeattr 
        if self.convmethod == 'GINConv':
            self.feat_embedding = torch.nn.Linear(self.indim, self.dim11)

        gnn_param_temp = {'layer':1, 'gIndim': self.indim, 'gOdim': self.dim11, 'nofcommu': self.k, 'pRatio':self.ratio1, 'normalize':self.normalize,
                        'bn_fg':self.bn_fg, 'pMethod':self.poolmethod, 'convmethod':self.convmethod, 'nofEdgeAttr':nofEdgeAttr,
                        'gn_lambda':mopt.grpnorm_lambda, 'edge_update': self.edge_update}
        gnn_param_1 = gnn_param_temp.copy()
        gnn_param_1.update({'layer':2, 'gIndim': self.dim11, 'gOdim': self.dim12, 'pRatio':self.ratio2})
        gnn_plist = [ gnn_param_temp,
                      gnn_param_1
                      ]
        if self.n_gc > 2:
            final_dim = self.dim13 if hasattr(self,'dim13') else self.dim12
            for i in range(2, self.n_gc -1 ):
                gnn_param_1 = gnn_param_temp.copy()
                gnn_param_1.update({'layer':i+1, 'gIndim': self.dim12, 'gOdim': self.dim12,'pRatio':self.ratio2})
                gnn_plist.append( gnn_param_1)
            gnn_param_1 = gnn_param_temp.copy()
            gnn_param_1.update({'layer':self.n_gc, 'gIndim': self.dim12, 'gOdim': final_dim,'pRatio':self.ratio2})
            gnn_plist.append(gnn_param_1)

        if self.convmethod == 'LI_Net':
            if predicate_create_shared('commu_attn'):
                if mopt.ext_embedding_tensor_ is None: 
                    if self.use_nn_lookuptable:
                        self.commu_attn = torch.nn.Sequential(
                            torch.nn.Embedding(self.R,self.k),
                            torch.nn.Softmax(dim=-1))
                    else:
                        self.commu_attn = torch.nn.Sequential(
                        torch.nn.Linear(self.R, self.k),
                        torch.nn.Softmax(dim=-1)
                        )
                else: 
                    self.commu_attn = torch.nn.Sequential(
                        torch.nn.Linear(mopt.ext_embedding_tensor_.size(1), self.k),
                        torch.nn.Softmax(dim=-1)
                        )
                
                self.sharable_modules['commu_attn'] = self.commu_attn
            else:
                self.sharable_modules['commu_attn'] = imported_modules['commu_attn']

        in_fc1_dim = 0 
        for i in range(len(gnn_plist)):
            in_fc1_dim += 2* gnn_plist[i]['gOdim']

        out_fc1 = self.dim2
        if self.memp >= 1:
            if self.memp == 1:
                mempool_p = {'out_channels':self.dimmp, 'heads':self.heads, 'num_clusters':self.clusters}
            else:
                mempool_p = []
                clusters = self.clusters
                out_chan = self.dimmp
                for _ in range(self.memp):
                    mempool_p.insert(0,{'out_channels': out_chan, 'heads':self.heads, 'num_clusters':clusters})
                    clusters = clusters *4
                    out_chan = gnn_plist[-1]['gOdim']

            in_fc1_dim = in_fc1_dim + self.clusters * self.dimmp if self.inc_global else self.clusters * self.dimmp
        else:
            mempool_p = None
            nodeNo = self.R
            for i in range(len(gnn_plist)):
                nodeNo = math.ceil(nodeNo * gnn_plist[i]['pRatio'])
                in_fc1_dim_nd = nodeNo * gnn_plist[i]['gOdim']
            in_fc1_dim = in_fc1_dim + in_fc1_dim_nd if self.inc_global else in_fc1_dim_nd          
        xinput_param = {'nofNodes': self.R, 'indim': self.indim}
        self.gnn_plus_mpool = GnnPlusMempool(gnn_plist, mempool_p, xinput_param, self.inc_global, 
                                                predicate_create_shared, sharable_modules, imported_modules,
                                                residual = mopt.mempool_residual, l1_reg= self.l1_reg, p_dropout=self.dropout)  
        self.edge_attr_attn =  None 
        if self.dim_cv != 0:
            in_fc1_dim = in_fc1_dim + self.dim_cv

        self.in_fc1_dim = in_fc1_dim
        if hasattr(mopt,'support_attn') and mopt.support_attn == -1:
            self.attn_dim = in_fc1_dim//2 # for multi edge_index attention
            self.attnQ = torch.nn.Sequential(torch.nn.Linear(self.R *self.indim, self.attn_dim), torch.nn.Tanh())
            self.attnK = torch.nn.Sequential(torch.nn.Linear(in_fc1_dim,         self.attn_dim), torch.nn.Tanh())
            self.attnV = torch.nn.Sequential(torch.nn.Linear(self.attn_dim,            1))

        xmapping_dim = in_fc1_dim  
        if self.residual:
            self.x_mapping = torch.nn.Sequential(nn.Linear(self.R *self.indim, xmapping_dim), nn.ReLU())
            
        if predicate_create_shared('fc1'): #share fc1's weights
            self.fc1 = torch.nn.Linear(in_fc1_dim, out_fc1)
            self.fc1_bn = torch.nn.BatchNorm1d(out_fc1)
            self.sharable_modules['fc1'] = (self.fc1,self.fc1_bn)
        else:
            self.sharable_modules['fc1'] = imported_modules['fc1']

        self.outdim = out_fc1

        for m in self.modules():
            if isinstance(m,torch.nn.Linear):
                if not hasattr(mopt,'init_type') or mopt.init_type == 'normal': 
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                else:
                    torch.nn.init.kaiming_uniform_(m.weight, mode='fan_out')
       
    def get_outdim(self):
        return self.outdim

    def get_gbn_gamma(self):
        gamma = self.gnn_plus_mpool.get_gbn_gamma_list()
        _,bn1 = self.sharable_modules['fc1']
        gamma.append(bn1.weight)
        return torch.hstack(gamma)
   
    def qr_commu_attn_fc1(self):
        if self.train:
            with torch.no_grad():
                new_params = torch.linalg.qr(torch.t(self.fc1.weight))[0] 
                self.fc1.weight.copy_(torch.t(new_params))

    def forward(self, x, edge_index, batch, edge_attr,cov, pos=None):
        num_sam = batch[-1]+1
        if self.convmethod == 'LI_Net' and pos is None: 
            pos = [self.R]*num_sam
        if type(pos) == np.int64: 
            pos = [pos]*num_sam
        if self.use_nn_lookuptable:
            #print('use nn.Embedding')   
            pos = torch.vstack([torch.LongTensor(list(range(self.R))) for _ in range(num_sam)])
            pos = pos.to(x.device)
        elif type(pos) == list:
            assert len(pos) == batch[-1]+1, 'pos, batch not matched!'
            pos = torch.vstack([torch.from_numpy(np.diag(np.ones(p))).float() for p in pos])
            pos = pos.to(x.device)
        
        if 'commu_attn' in self.sharable_modules:
            commu_attn  = self.sharable_modules['commu_attn']
            pos = commu_attn(pos)
            if self.use_nn_lookuptable:
                pos = pos.view(-1,self.k)

        x_input = x.view(num_sam,-1)
        if isinstance(edge_index, list): 
            assert False
        else:
            if self.edge_attr_attn is not None:
                edge_attr = self.edge_attr_attn(edge_attr).squeeze()
            in_fc1, training_output = self.gnn_plus_mpool(x,pos,batch,edge_index,edge_attr)
            
        if self.residual:  
            x_mapping = self.x_mapping(x_input)
            in_fc1 = in_fc1 + x_mapping

        if cov is not None:   
            in_fc1 = torch.cat([in_fc1, cov],dim=1)

        fc1,bn1 = self.sharable_modules['fc1']
        x = choose_func(self.activation_type,self.bn_fg, bn1,self.outdim,
                        fc1,leaky_nslope=self.leaky_nslope)(in_fc1,batch)

        training_output['x3'] = in_fc1 if self.n_gc ==1 else x
        training_output['gamma'] = self.get_gbn_gamma() if self.bn_fg else None
        self.train_output = training_output
        return x

    def take_train_output(self, keep=False):
        r = self.train_output
        if not keep:
            self.train_output = None 
        return r

class GphClassif(torch.nn.Module):
    def __init__(self, indim, mopt):
        super(GphClassif,self).__init__()
        self.indim = indim
        self.dim3  = mopt.dim3
        self.dim4  = mopt.dim4
        self.n_fc  = mopt.fc
        self.nclass = mopt.nclass
        self.target = mopt.targettype
        self.target_list = mopt.target_list
        self.activation_type = mopt.activationtype # 
        self.bn_fg = mopt.bnfg 
        self.dropout = mopt.dropout
        self.k = mopt.nocommu 
        self.R = mopt.nogenes 
        self.leaky_nslope = 0.7 if not hasattr(mopt,'actfnparm') else mopt.actfnparm
        self.train_output = None

        if self.target == -1: 
            out_fc = 1       
        elif self.target >= 0: 
            out_fc = self.nclass
        out_fc1 = self.indim
        if self.n_fc == 2:
            self.final_fc =  torch.nn.Linear(out_fc1,out_fc)
        elif self.n_fc == 3:
            self.linears  = torch.nn.ModuleList([torch.nn.Linear(out_fc1,self.dim3)])
            self.clf_bns      = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.dim3)])
            self.bns_dims = [self.dim3]
            self.final_fc =  torch.nn.Linear(self.dim3, out_fc)
        else: # fc==4
            self.linears  = torch.nn.ModuleList([torch.nn.Linear(out_fc1,self.dim3),
                            torch.nn.Linear(self.dim3,self.dim4)])
            self.clf_bns      = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.dim3),
                            torch.nn.BatchNorm1d(self.dim4)])
            self.bns_dims = [self.dim3, self.dim4]
            self.final_fc =  torch.nn.Linear(self.dim4, out_fc)
        self.L1_reg = mopt.l1_reg if hasattr(mopt,'l1_reg') else 0.0
        
        for m in self.modules():
            if isinstance(m,torch.nn.Linear):
                if not hasattr(mopt,'init_type') or mopt.init_type == 'normal': 
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                else:
                    torch.nn.init.kaiming_uniform_(m.weight, mode='fan_out')

    def get_indim(self):
        return self.indim

    def get_bn_gamma(self):
        gammars =[]
        if self.n_fc == 3:
            gammars.append(self.clf_bns[0].weight)
        elif self.n_fc == 4:
            gammars.extend([self.clf_bns[0].weight,self.clf_bns[1].weight])
        return torch.hstack(gammars)  
   
    def forward(self, x): 
        training_output={}
        if self.n_fc >2:
            for n, (linear,bn,bn_dim) in enumerate(zip(self.linears,self.clf_bns,self.bns_dims)):
                x = choose_func(self.activation_type,self.bn_fg,bn,bn_dim,linear,
                                leaky_nslope=self.leaky_nslope)(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training) 
                if n==0:
                    training_output['fc2'] = x

        x = self.final_fc(x)
        if self.target == 0:
            x = F.log_softmax(x, dim=-1)

        if self.bn_fg ==1 and self.n_fc >2:
            training_output['gamma'] = self.get_bn_gamma()
        else:
            training_output['gamma'] = None
        self.train_output = training_output
        return x
    
    def take_train_output(self, keep=False):
        r = self.train_output
        if not keep:
            self.train_output = None
        return r

class GphRegressorWrapper(torch.nn.Module):
    def __init__(self,ext_regressor):
        super(GphRegressorWrapper,self).__init__()
        self.ext_reg = ext_regressor
        self.linear = nn.Linear(ext_regressor.get_indim(), ext_regressor.get_indim())
        torch.nn.init.kaiming_normal_(self.linear.weight, mode='fan_out')
    
    def forward(self, x, batch):
        return self.ext_reg(self.linear(x),batch)

class NNGAT_Net(torch.nn.Module, TorchHistory):
    def __init__(self, opt, mopt, skip_classifier=False):
        super(NNGAT_Net, self).__init__()
        self.mopt = mopt
        self.lamb0 = mopt.lamb0
        self.l1_reg = mopt.l1_reg
        self.sharable_modules = dict()
        self.shared_module_names = (opt.shared_modules if hasattr(opt,'shared_modules')
                                         else None)
        
        predicate_create_shared = lambda name: (mopt.shared_modules is None 
                                                or name not in mopt.shared_modules)
                                               
        self.embedding = GphEmbedding(mopt,
                predicate_create_shared, self.sharable_modules, mopt.shared_modules)

        if skip_classifier:
            self.embedding_only = True
            return
        else:
            self.embedding_only = False

        if predicate_create_shared('classif'):
            classif   = GphClassif(self.embedding.get_outdim(), mopt)
            self.classif = classif
        else:
            classif = mopt.shared_modules['classif']
        self.sharable_modules['classif'] = classif
                
      
    def forward(self, x, edge_index, batch, edge_attr, cov=None, pos=None,detached=False): # pos is not used here
        classif = self.sharable_modules['classif']
        x = self.embedding(x, edge_index, batch, edge_attr,cov,pos)
        extra = self.embedding.take_train_output()
        if self.embedding_only:
             return x, {**extra}
        emb_out = x
        x = classif(x if not detached else x.detach())
        extra2 = classif.take_train_output()
        if (extra['gamma'] is None) and (extra2['gamma'] is None):
            gamma = None
        elif extra['gamma'] is None:
            gamma = extra2['gamma']
        elif extra2['gamma'] is None:
            gamma = extra['gamma']
        else:
            gamma = torch.hstack([extra['gamma'],extra2['gamma']])
        return x, {**extra, **extra2,'emb_out':emb_out, 'gamma':gamma} 
 
    def update_weights(self, first_step =True):
        def l1_regularize():
            if self.l1_reg >0.0:
                for n,p in zip(self.decay_name, self.decay):
                    if 'weight' in n and p.grad is not None:
                        p.grad = p.grad + self.l1_reg * torch.sign(p.data)
        l1_regularize()
        if first_step:
            self.optimizer.step() if not hasattr(self.optimizer,'first_step') else self.optimizer.first_step(zero_grad=True)
        else:
            self.optimizer.second_step(zero_grad=True)

    def is_optimizer_SAM(self):
        return hasattr(self.optimizer, 'second_step')

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step_scheduler(self):
        self.scheduler.step()
    
    def step_scheduler2(self):
        if hasattr(self,'scheduler2'):
            self.scheduler2.step()
            
    def step_scheduler3(self):
        if hasattr(self,'scheduler3'):
            self.scheduler3.step()

    
    def get_shared_modules(self):
        return {k:self.sharable_modules[k] for k in self.shared_module_names} if self.shared_module_names is not None else None

    def label_mapping(self, class_idx_ls):
        if hasattr(self.mopt,'label_mapping'):
            class_idx_ls = [self.mopt.label_mapping[k] for k in class_idx_ls]
        return class_idx_ls


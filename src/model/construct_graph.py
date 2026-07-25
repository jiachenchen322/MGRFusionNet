
from collections import defaultdict
import os.path as osp
from os import listdir
import sys
import h5py
import torch
import numpy as np
from torch_geometric.data import Data
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from functools import partial
import pandas as pd

# Edge source holding the prior biological network, one file per fold under
# <data_dir>/edge_info/fold-<k>/. Optional: when absent, the graph is built from
# the class-conditional and continuous-correlation edge sources alone.
PRIOR_EDGE_FILE = 'edge_info_prior.hdf5'

_anno_1_sdict = None # defaultdict(dict)

def read_data(data_dir, node_th,  target = 0, anno_1_cols = [], anno_2 = None, 
        convmethod="GCNConv", bin_edge=0, fold=0, support_attn=None, no_mostrelated=0):
    global _anno_1_sdict
    _anno_1_sdict = defaultdict(dict)

    if osp.exists(osp.join(data_dir, 'etc', 'samid_list.hdf5')):
        with h5py.File(osp.join(data_dir, 'etc', 'samid_list.hdf5'),'r') as samh5:
            samids = samh5['samids'][()]
            if isinstance(samids[0], bytes):
                samids = [s.decode() for s in samids]

    sample_file_name = None
    if osp.exists(osp.join(data_dir, f'sample_file_name_f{fold}.txt')):
        with open(osp.join(data_dir, f'sample_file_name_f{fold}.txt')) as fh:
            sample_file_name = fh.readline().strip()

    import timeit
    start = timeit.default_timer()

    if sample_file_name and osp.exists(osp.join(data_dir,sample_file_name)) and osp.exists(osp.join(data_dir, 'etc', 'samid_list.hdf5')):
        res = read_data_list(samids, osp.join(data_dir,sample_file_name), osp.join(data_dir,'all_samples_targets.csv'))
    else:
        func = partial(read_single_data, data_dir, node_th, target, anno_1_cols, anno_2, convmethod, bin_edge)
        if osp.exists(osp.join(data_dir, 'etc', 'samid_list.hdf5')):
            onlyfiles = [f'{samids[i]}.hdf5' for i in range(len(samids))]
        else:
            onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
            onlyfiles.sort()
      
        res =list( map(func,onlyfiles) )

    stop = timeit.default_timer()
    # print('Time: ', stop - start)

    if not osp.isdir(osp.join(data_dir,'edge_info')):
        return [Data(
        x = torch.from_numpy(r[0]).float(), 
        y = 0,
        z = torch.cat([torch.from_numpy(q).long() if target != -1 \
            else torch.from_numpy(q).float() for q in r[1]]).unsqueeze(0), 
        edge_index = r[4],
        edge_attr =  r[5],
        pos = r[7],
        cols =  r[2],   
        samid = r[3],
        cov = 0
        ) for r in res]

    edge_files = []
    if target >= 0:
        labels = np.unique(np.array([r[1][r[2].index('labels')] for r in res]))
        edge_files = [ f'edge_info_cat_{i}.hdf5' for i in  labels]
    edge_files.append('edge_info_cont.hdf5')
    edge_files.append(PRIOR_EDGE_FILE)

    prior_edge_path = osp.join(data_dir,'edge_info',f'fold-{fold}',PRIOR_EDGE_FILE)
    has_rsc = False
    if osp.exists(prior_edge_path):
        with h5py.File(prior_edge_path,'r') as f:
            if 'risk_scores' in f.keys():
                risk_scores = f['risk_scores'][()]
                has_rsc =True
    else:
        print(f'[read_data] no prior-network edges at {prior_edge_path}; '
              f'using the remaining edge sources only.', file=sys.stderr)

    def get_edge_info(edge_path):
        pos = None
        edge_index_ls, edge_att_ls, num_nodes_ls =[], [], []
        Xcorr,traitcorr, idx_mostrelated = None, None, None
        for edgefile in edge_files:
            if not osp.exists(osp.join(edge_path,edgefile)):
                continue
            with h5py.File(osp.join(edge_path, edgefile),'r') as temp:
                num_nodes  = temp['num_nodes'][()]
                edge_index = temp['edge_index'][()]
                edge_att   = temp['edge_att'][()]

                if edgefile == 'edge_info_cont.hdf5':
                    if 'Xcorr' in temp.keys():
                        Xcorr = temp['Xcorr'][()]
                        #pos = torch.from_numpy(Xcorr).float()
                        
                    if 'traitcorr' in temp.keys():
                        traitcorr       = temp['traitcorr'][()]
                    
                    if no_mostrelated >0 and traitcorr is not None:
                        idx_mostrelated = np.array(np.argpartition(-np.abs(traitcorr),no_mostrelated)[:no_mostrelated])       

                edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index).long(), 
                                                        torch.from_numpy(edge_att).float())
                edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                                num_nodes)
                edge_index_ls.append(edge_index) 
                edge_att_ls.append(edge_att)
                num_nodes_ls.append(num_nodes)
        return edge_index_ls, edge_att_ls, Xcorr, idx_mostrelated, traitcorr, pos
    
    labels = labels.tolist()
    edge_path = osp.join(data_dir,'edge_info',f'fold-{fold}')
    edge_index_ls, edge_att_ls, Xcorr, idx_mostrelated, traitcorr, pos = get_edge_info(edge_path)
    edge_info = (edge_index_ls,edge_att_ls)

    if support_attn is None:
        make_edge_idx = lambda r: labels.index(r[1][r[2].tolist().index('labels')].item())
    elif support_attn ==-1:
        make_edge_idx = lambda _: 0
    elif support_attn == -2:
        make_edge_idx = lambda _: -2 
    elif support_attn <= -3:
        make_edge_idx = lambda _: -1
    else:
        make_edge_idx = lambda _: support_attn

    if target == -1:
        return [Data(
        x = torch.from_numpy(r[0]).float(), 
        y = 0, 
        z = torch.cat([torch.from_numpy(q).float() for q in r[1]]).unsqueeze(0), 
        edge_index = edge_index_ls[0],           
        edge_attr = edge_att_ls[0], 
        pos = pos, 
        cols =  r[2],   
        samid = r[3],
        cov = 0
        ) for r in res]
    else:
        if idx_mostrelated is not None or has_rsc:
            for r in res:
                if idx_mostrelated is not None:
                    r[0] = np.hstack((r[0], traitcorr.reshape((-1,1)), Xcorr[:,idx_mostrelated]))
                if has_rsc:
                    r[0] = np.hstack([r[0], risk_scores])
        # When support_attn selects edges by a constant index (not per-sample label),
        # store empty edge placeholders to avoid OOM during InMemoryDataset.collate
        # for large graphs (edge_index injected later in DomainGraphDataset.get).
        _const_edge = support_attn is not None and support_attn != -1
        _empty_ei = torch.zeros((2, 0), dtype=torch.long)
        _empty_ea = torch.zeros(0, dtype=torch.float)
        return [Data(
        x = torch.from_numpy(r[0]).float(),
        y = 0,
        z = torch.cat([torch.from_numpy(q).long()  for q in r[1]]).unsqueeze(0),
        edge_index = _empty_ei if _const_edge else edge_index_ls[make_edge_idx(r)],
        edge_attr  = _empty_ea if _const_edge else edge_att_ls[make_edge_idx(r)],
        pos        = pos,
        cols       =  r[2],
        samid      = r[3],
        cov        = 0
        ) for r in res], edge_info

def read_data_list(samids, all_samples, all_sample_targets):
    df_targets = pd.read_csv(all_sample_targets)
    h5_samples = h5py.File(all_samples,'r')
    data_list = []
    edge_index, edge_att, num_nodes, Xcorr = None, None, None, None
    anno_1_cols = h5_samples['cols'][:] 
    if h5py.__version__ > '3.0.0' :
        anno_1_cols = [n.decode() for n in anno_1_cols]
    else:
        anno_1_cols = [n for n in anno_1_cols]
    colsInAnno = anno_1_cols
    for idx,sid in enumerate(samids):
        Xatt = h5_samples[f'sample/{sid}'][:]
        samid =  sid.decode() if type(sid) == bytes else sid  
        Yindicator = df_targets[anno_1_cols].iloc[[idx]].to_numpy()
        data_list.append([Xatt, Yindicator, colsInAnno, samid, edge_index, edge_att, num_nodes, Xcorr])
    return data_list

def read_single_data(data_dir, node_th, target, anno_1_cols, anno_2, convmethod, bin_edge,filename):  
    temp = h5py.File(osp.join(data_dir, filename), 'r')
    datasets = temp.keys()
    edge_index, edge_att, num_nodes, Xcorr= None, None, None, None
    if 'edge_index' in  datasets :
        num_nodes  = temp['num_nodes'][()]
        edge_index = temp['edge_index'][()]
        edge_att   = temp['edge_att'][()]
        Xcorr      = torch.from_numpy(temp['Xcorr'][()]).float() if 'Xcorr' in temp.keys() else None 
        edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index).long(), 
                                                torch.from_numpy(edge_att).float())
        edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                        num_nodes)
    Xatt = temp['sample'][:]
    samid = temp['samid'][()] # from raw data
    samid = samid.decode() if type(samid) == bytes else samid
    
    if not anno_1_cols: 
        anno_1_cols = temp['cols'][:] 
    else:
        anno_1_cols = np.array(anno_1_cols)

    Yindicator = []
    for p in anno_1_cols:
        if len(temp[p].shape) > 0:
            Yindicator.append(temp[p][()])
        else:
            v = temp[p][()] 
            if temp[p].dtype.type  == np.object_:
                sdict = _anno_1_sdict[p]
                if v not in sdict.keys():
                    sdict[v] =len(sdict)
                v = sdict[v]                 
            Yindicator.append(np.array([v]))

    if anno_2 is not None:
        for cv in list(anno_2.columns[1:]):
            Yindicator.append(anno_2[anno_2['samid'] == str(filename)[:-5]][cv].values[0]) 
        colsInAnno = np.append(anno_1_cols, list(anno_2.columns)[1:])
    else:
        colsInAnno = anno_1_cols

    return  [Xatt, Yindicator, colsInAnno, samid, edge_index, edge_att, num_nodes, Xcorr]
   

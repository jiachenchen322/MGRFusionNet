import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from os import listdir
import random,math
import os.path as osp
from .construct_graph import read_data
import copy

class BasegraphDataset(InMemoryDataset):
    def __init__(self, root, name, xopt, transform=None, pre_transform=None, anno_2 = None):
        self.root = root
        self.name = name
        self.node_th = xopt.node_th
        self.anno_1_cols = xopt.target_list
        self.target = xopt.targettype
        self.convmethod = xopt.convmethod
        self.bin_edge= xopt.bin_edge
        self.ext_embedding = xopt.ext_embedding_tensor_
        self.fold = xopt.fold

        self.anno_2 = anno_2
        self.xopt = xopt
        if hasattr(xopt,'dataset_init_process') and xopt.dataset_init_process:
            setattr(BasegraphDataset, 'process', BasegraphDataset.my_process)
        super(BasegraphDataset, self).__init__(root,transform, pre_transform)
        if hasattr(xopt,'dataset_init_process') and xopt.dataset_init_process:
            self.data, self.slices, self.extra_edge_info = torch.load(self.processed_paths[0])
        else:
            self.my_process(save=False)
        
    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root,'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        #print(onlyfiles)
        return onlyfiles
    
    @property
    def processed_file_names(self):
        targets ='-' + '-'.join(self.anno_1_cols) if self.anno_1_cols else ''
        if self.bin_edge == 1: # ZF changed bin_edge to self.bin_edge
            processed_fn = self.name + targets+ "_bin_" + str(self.node_th)  +'.pt' ## YL
        else:
            processed_fn =  self.name + targets+ "_" + str(self.node_th)  +'.pt' ## YL
            if len(processed_fn) > 250:
                processed_fn = self.name + targets[:50] + '_addtl_' + str(self.node_th) + '.pt'
        return processed_fn + f'.f{self.fold}'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def my_process(self, save=True):
        # Read data into huge `Data` list.
        print('Data Loading ...')
        print(self.raw_dir)
        # print(self.node_th)
        # print(self.target)
        data_list, edge_info = read_data(self.raw_dir, self.node_th, self.target, self.anno_1_cols,
             self.anno_2, self.convmethod, fold = self.fold, support_attn = self.xopt.support_attn,
             no_mostrelated = self.xopt.no_mostrelated)
        #print(self.data)
        if self.pre_filter is not None:
            data_list = map(self.pre_filter, data_list)
        if self.pre_transform is not None:
            data_list = map(self.pre_transform, data_list)  
        
        self.data, self.slices = self.collate(data_list)
        self.extra_edge_info   = edge_info
        if save:
            torch.save((self.data, self.slices, edge_info), self.processed_paths[0])
    
    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get(self,index):
        data = InMemoryDataset.get(self, index)
        if self.ext_embedding is not None:
            # replace pos with extern embedding
            data.pos = self.ext_embedding
        return data, index

    def len(self) :
        return InMemoryDataset.len(self)

class DomainGraphDataset(InMemoryDataset):
    def __init__(self, dataset, opt, nclass, name='Graph', mopt=None):
        self.name = name
        self.whole_set = None
        self.ds, self.ref_idx = dataset if isinstance(dataset, tuple) else (dataset, None)
        self.balanced = False
        self.support_attn = mopt.support_attn
        self.edge_info = self.ds.extra_edge_info

        if opt.target == -1:
            self.index_sets = [[],[]]
            for i in range(len(self.ds)):  
                d = self.ds[i] if self.ref_idx is None else self.ds[self.ref_idx[i]]
                if d[0].y > 0 : 
                    if mopt is not None and hasattr(mopt,'outlier_exp'):
                        if mopt.outlier_exp[0] is not None and d[0].y > mopt.outlier_exp[0]:
                            continue #throw away outlier
                    self.index_sets[0].append(i)
                else : 
                    if mopt is not None and hasattr(mopt,'outlier_exp'):
                        if len(mopt.outlier_exp) >1 and opt.outlier_exp[1] is not None and d[0].y < mopt.outlier[0]:
                            continue #throw away outlier
                    self.index_sets[1].append(i)
        else: #CATEGORICAL DATA
            self.index_sets = [[] for i in range(nclass)]
            for i,d in enumerate(self.ds):
                try:
                    self.index_sets[d[0].y].append(i)
                except IndexError:
                    self.index_sets.append([])
                    self.index_sets[d[0].y].append(i)
        self.static_index_sets = copy.deepcopy(self.index_sets)  
        self.whole_set = self._merge_index_set(self.index_sets)
        super(DomainGraphDataset,self).__init__("",None,None,None)
        self.slices = {'samid': torch.tensor(range(1+len(self.whole_set)))}
        self.data = self.ds.data

    @staticmethod
    def _merge_index_set(sets):
        def _merge_2index_set(set1, set2):
            if len(set1) == 0:
                return set2
            if len(set2) == 0:
                return set1    
            r = len(set1)/(len(set1)+len(set2))
            merged = list(range(len(set1) + len(set2)))
            j,k = 0,0
            for i in range(len(merged)):
                if ((j+1)/(i+1) <= r and j < len(set1)) or (k >=len(set2)):
                    merged[i] = set1[j]
                    j +=1
                else:
                    merged[i] = set2[k]
                    k +=1    
            return merged
        s0 = sets[0]
        for i in range(1,len(sets)):
            s0 = _merge_2index_set(s0,sets[i])
        return s0    

    @staticmethod
    def _list_extend_size(n,s):
        assert n >= len(s) and isinstance(s,list)
        if n == len(s):
            return s
        t = s*(n//len(s))
        t.extend(s[:n%len(s)])
        return t 

    @property
    def raw_file_names(self):
        return  ""

    @property   
    def processed_file_names(self):
        return ""

    def get(self,index):
        _index = self.whole_set[index]
        _idx = self.ref_idx[_index] if self.ref_idx is not None else _index
        data = self.ds[_idx]

        if self.support_attn == -1:
            datals = []
            if len(self.edge_info) == 2: #edge_index, edge_attr
                for edge_idx, edge_att in zip(*self.edge_info):
                    #extra_args = {k:getattr(data[0], k) for k in ('x','y','pos','cols','samid')}
                    d = Data(
                            x = data[0].x,
                            edge_index = edge_idx,
                            edge_attr  = edge_att,
                            )
                    datals.append(d)
            return [*data, *datals]
        else:
            # If edge was stored as empty placeholder (to avoid OOM during collate),
            # inject the correct edge from edge_info now (per-sample, before batching).
            if data[0].edge_index.numel() == 0 and len(self.edge_info[0]) > 0:
                edge_index_ls, edge_att_ls = self.edge_info
                n = len(edge_index_ls)
                if self.support_attn == -2:
                    ei_idx = n - 2
                elif self.support_attn <= -3:
                    ei_idx = n - 1
                else:
                    ei_idx = self.support_attn
                data[0].edge_index = edge_index_ls[ei_idx]
                data[0].edge_attr  = edge_att_ls[ei_idx]
            return data

    def len(self):
        return len(self.whole_set)

    def __repr__(self):
        return '{}({})'.format(self.name,len(self.ds))              
    
    def shuffleSamples(self):
        for s in self.index_sets:
            random.shuffle(s)
        idx = list(range(len(self.index_sets)))
        random.shuffle(idx)
        index_sets = [self.index_sets[i] for i in idx]
        if self.balanced:
            self._balance(self.balanced_batchsize,index_sets)
        else:
            self.whole_set = self._merge_index_set(index_sets)
    
    def reset_random(self):
        self.index_sets = copy.deepcopy(self.static_index_sets)
        
    def __len__(self):
        return len(self.whole_set)

    def balance(self, batch_size=1):
        self._balance(batch_size)
        return self

    def _balance(self,batch_size,idxsets=None):
        self.balanced = True
        self.balanced_batchsize = batch_size
        if idxsets is None:
            idxsets = self.index_sets
        sz_idxsets= [len(i) for i in idxsets]
        maxsz = ((max(sz_idxsets) +batch_size -1) //batch_size) *batch_size
        newsets =[]
        for idxset in idxsets:
            if len(idxset) >0 :
                newsets.append(self._list_extend_size(maxsz,idxset))
        self.whole_set =  self._merge_index_set(newsets)

    def size_extend(self,size):
        if size > len(self.whole_set):
            newsets = []
            for s in self.index_sets:
                random.shuffle(s)
                psize = math.floor(size*(len(s)/len(self.whole_set)))
                size = size - psize
                t = self._list_extend_size(psize, s)
                newsets.append(t)
            self.index_sets = newsets    
            self.whole_set = self._merge_index_set(self.index_sets)
        return self


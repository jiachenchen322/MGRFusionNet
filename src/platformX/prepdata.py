from model.graph_dataset import BasegraphDataset,DomainGraphDataset
from model.multiomics_dataset import MultiOmicsDataset
import torch
from utils.search_utils import target_mapping
from utils.common_loss import W_consist
from utils.utils import load_from_permute_tab, get_split_idx_path, predicate_xopt
import numpy as np
import h5py
import os
import torch_geometric
import pandas as pd
import re

_quite = False
def O_print(*arg, **args):
    if not _quite:
        print(*arg, **args)

def load_split_file(mopt, split = 'splits'):
    split_path =  get_split_idx_path(mopt, split=split) 
    if os.path.exists(split_path):
        with h5py.File(split_path,'r') as fh:
            try:
                tr_index = fh[f'{mopt.fold}/tr_index'][()]
                te_index = fh[f'{mopt.fold}/te_index'][()]
                val_index = fh[f'{mopt.fold}/val_index'][()]
            except:
                tr_index = fh['tr_index'][()]
                te_index = fh['te_index'][()]
                val_index = fh['val_index'][()]
        ret = (tr_index,te_index,val_index)
        return ret
    return None

import fcntl
from sklearn.decomposition import PCA

class FileLocker:
    def __init__(self, lck_dir):
        self.lck_file = os.path.join(lck_dir,'lockfile.lck')

    def __enter__ (self):
        self.fp = open(self.lck_file,'w')
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__ (self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()

def get_extern_embedding(opt, mopt):
    print('INFO: get extern embeddng ..')
    dirname          = mopt.root_prefix1 
    embeddings_file  = opt.extern_embedding 
    gene_symbol_file = mopt.gene_syms 

    gene2vec = pd.read_csv(os.path.join(dirname,embeddings_file),index_col=0)
    genesyms = pd.read_csv(os.path.join(dirname,gene_symbol_file),header=None,keep_default_na=False)
    commongenes = np.intersect1d(gene2vec.index.to_numpy(), genesyms[1].to_numpy())
    
    if hasattr(opt,'extern_embedding_ncomp'):
        n_components = opt.extern_embedding_ncomp
    else:
        n_components = 3
    pca0 = PCA(n_components=n_components)
    pca_vec0 = pca0.fit_transform(gene2vec.loc[commongenes].to_numpy())
    
    symbols = pd.read_csv(os.path.join(mopt.dataroot_, f'{mopt.tissue_name}-{mopt.target_}-gene_syms.txt'),header=None)
    symbols = symbols[0].to_numpy()
    
    ext_emb = np.zeros((len(symbols), pca_vec0.shape[1]))
    idx = np.nonzero(symbols[:,None] == commongenes[None,:])
    ext_emb[idx[0]] = pca_vec0[idx[1]]
    
    if len(idx[0]) < len(symbols): 
        #there are missing symbols, fill with random.normal(mean,std)
        mean = np.mean(pca_vec0, axis=0)
        std  = np.std(pca_vec0, axis=0)
        idx_missing = np.nonzero(np.logical_not(np.in1d(np.arange(len(symbols)), idx[0])))[0]
        missing = np.random.normal(mean,std,(len(idx_missing),len(mean)))
        ext_emb[idx_missing] = missing
    return ext_emb

def g_prepdata_0(opt, mopt,inf=False,allow_permute=True): 
    cv_list = mopt.cv_list
    
    if inf and hasattr(mopt,'inf_data_base_path'):
        base_path = mopt.inf_dataroot_   
    else:    
        base_path = mopt.dataroot_
    target_list = mopt.target_list
    # print('data base path: ', base_path)
    opt.net_norm = True
    O_print('nogenes:',str(mopt.nogenes))
    name = 'data'
    graph_dataroot= base_path #os.path.join(base_path, name)

    mopt.ext_embedding_tensor_ = (torch.from_numpy(get_extern_embedding(opt,mopt)).float() 
                                if hasattr(opt,'extern_embedding') else None)
    
    dataset = BasegraphDataset(graph_dataroot, name, mopt)
        
    permute = None
    if opt.target_permute and allow_permute: 
        permute = load_from_permute_tab(mopt, opt.permute_no)
    if len(cv_list) != 0:
        import pandas as pd
        anno_path = os.path.join(mopt.root_prefix1, f'{mopt.tissue_name}-anno.csv')
        anno = pd.read_csv(anno_path)
        cov = [torch.tensor(anno[cv].to_numpy()[:,None]) for cv in cv_list]
        dataset.data.cov = torch.hstack(cov)
        
    if opt.target == 0:
        tidx = [ dataset.data.cols[0].index(mopt.target_list[0]) ]
        dataset.data.y = target_mapping(dataset, tidx, permute,mopt=mopt) 
        O_print('prepdata: converted y')
    else:
        tidx =np.where(dataset.data.cols[0]== mopt.target_list[0])[0]
        dataset.data.y = target_mapping(dataset, tidx, permute)
        O_print('converted y')    
    
    if hasattr(mopt,'ztarget'):
        if isinstance(mopt.ztarget, list):
            mopt.zidx = []
            if not hasattr(opt,'znum'):
                mopt.znum = []
            for ztarget in mopt.ztarget:
                O_print(f"has ztarget: {ztarget}")
                mopt.zidx.append(np.where(dataset.data.cols[0]==ztarget)[0])
                if len(mopt.zum) < len(mopt.ztarget):
                    mopt.znum.append(np.unique(dataset.data.z[:,mopt.zidx[-1]]).shape[0])
        else:
            O_print(f"has ztarget: {mopt.ztarget}")
            mopt.zidx = np.where(dataset.data.cols[0]==mopt.ztarget)[0]
            if not hasattr(mopt,'znum'):
                mopt.znum = np.unique(dataset.data.z[:,mopt.zidx]).shape[0]
    
    if len(cv_list) == 0:
        dataset.data.cov = None
    if hasattr(mopt,'indim'):
        assert mopt.indim == dataset[0][0].x.shape[1]
    mopt.indim = dataset[0][0].x.shape[1] 
    return dataset

def prepdata(opt, mopt, inf=False, quite=True): 
    opt_predicate = lambda xopt, arg: hasattr(xopt,arg) and getattr(xopt,arg)
    global _quite
    _quite = quite
    nclass = mopt.nclass
    use_different_testset = False
    if inf and hasattr(mopt,'inf_data_base_path'):
        base_path = mopt.inf_dataroot_
        use_different_testset = True
    else:    
        base_path = mopt.dataroot_
    target_list = mopt.target_list
    dataset = g_prepdata_0(opt, mopt,inf)
    _dataset = DomainGraphDataset(dataset,opt,nclass,mopt=mopt)

    if use_different_testset:
        WW_train = 0
        mopt.WW_train = WW_train
        mopt.WW_val = WW_train
        mopt.WW_test = WW_train
        return _dataset,None,None,None

    _split_data = load_split_file(mopt, split='splits-common' if opt_predicate(opt,'common_only')  else 'splits' )
    assert _split_data is not None
    
    tr_index, te_index,val_index = _split_data
    O_print('no_tr_index:',str(len(tr_index)),'no_te_index:',str(len(te_index)))
    O_print('Dataset List Length: '+str(len(dataset)))
    if torch_geometric.__version__ < '2.2.0':
        train_mask = torch.zeros(len(dataset), dtype=torch.uint8)
        test_mask = torch.zeros(len(dataset), dtype=torch.uint8)
        val_mask = torch.zeros(len(dataset), dtype=torch.uint8)
        train_mask[tr_index] = 1
        test_mask[te_index] = 1
        val_mask[val_index] = 1
        test_dataset = dataset[test_mask]
        train_dataset = dataset[train_mask]
        val_dataset = dataset[val_mask]
    else:
        test_dataset = dataset[te_index]
        train_dataset = dataset[tr_index]
        val_dataset = dataset[val_index]
    if  opt.target == -1 and mopt.talos_dict['lamb5'][0] > 0: # for consist loss ##
        O_print('Processing WW')
        WW_train = W_consist(train_dataset.data.y)
        WW_val = W_consist(val_dataset.data.y)

        if opt.threesplit == 1:
            WW_test = W_consist(test_dataset[0].data.y)
        else:
            WW_test = WW_val
    else:
        WW_train = 0
        WW_val = 0
        WW_test = 0
    mopt.WW_train = WW_train
    mopt.WW_val = WW_val
    mopt.WW_test = WW_test  
    
    val_dataset = DomainGraphDataset(val_dataset,opt,nclass,mopt=mopt) 
    no_balance = hasattr(opt, 'no_balance') and opt.no_balance
    if opt.target == 0:
        if no_balance:
            train_dataset = DomainGraphDataset(train_dataset,opt,nclass,mopt=mopt)
        else:    
            train_dataset = DomainGraphDataset(train_dataset,opt,nclass,mopt=mopt).balance(opt.batchsize)
    else:
        train_dataset = DomainGraphDataset(train_dataset,opt,nclass,mopt=mopt)
    test_dataset = DomainGraphDataset(test_dataset,opt,nclass,mopt=mopt) 
    return _dataset,((train_dataset,[mopt]),),((val_dataset,[mopt]),),((test_dataset,[mopt]),)

def prepdata_multiomics(opt, mopts, inf=False, quite=True):
    global _quite
    _quite = quite
    assert type(mopts) == list and len(mopts) >1, 'not multiomics config!'
    nclass = mopts[0].nclass
    datasets = []
    use_different_testset = False
    for mopt in mopts:
        if inf and hasattr(mopt,'inf_data_base_path'):
            base_path = mopt.inf_dataroot_
            use_different_testset = True
        else:    
            base_path = mopt.dataroot_
        target_list = mopt.target_list
        dataset = g_prepdata_0(opt, mopt,inf)
        datasets.append(dataset)

    def _construct_datasets_m_omics(split='tr', splits_idents= None):
        _splits_dsets  = []
        if splits_idents is None:
            _splits_idents = ['splits-common','splits-com01','splits-com02','splits-com12',
                            'splits-special']
        else:
            _splits_idents = splits_idents
        
        for id in _splits_idents:
            _dsets =[]
            for m,dset in zip(mopts, datasets):
                _splits_fn = get_split_idx_path(m, split=id)
                if os.path.exists(_splits_fn):
                    with h5py.File(_splits_fn,'r') as h5:
                        if split == 'total':
                            idx = h5[f'{split}_index'][()]
                        else:
                            idx = h5[f'{m.fold}/{split}_index'][()]
                    if torch_geometric.__version__ < '2.2.0':
                        _mask = torch.zeros(len(dset), dtype=torch.uint8)
                        _mask[idx] =1
                        srt_idx = np.sort(idx)
                        ref_idx = np.nonzero(idx[:,None]== srt_idx[None,:])[1].tolist()
                        # ref_idx is the sequence of elem of idx in its sorted version srt_idx (i.e. order of dset[_mask])
                        _dsets.append((dset[_mask],m, ref_idx))
                    else:
                        _dsets.append((dset[idx],m, None))

            if len(_dsets) >0 :
                if  re.match('splits-com.*',id) and len(_dsets)>1:
                    _splits_dsets.append(list(zip(*_dsets)))
                else: # splits-special
                    _splits_dsets.extend(_dsets)

        _datasets =[]
        for ds, _mopt, idx in _splits_dsets:
            if type(ds) == tuple:
                _datasets.append((MultiOmicsDataset(opt, _mopt[0], ds, idx=idx), _mopt))
            else:
                _datasets.append((DomainGraphDataset(ds,opt,_mopt.nclass,mopt=_mopt),[_mopt]))
        
        return _datasets

    for mopt in mopts:
        WW_train = 0
        WW_val = 0
        WW_test = 0
        mopt.WW_train = WW_train
        mopt.WW_val = WW_val
        mopt.WW_test = WW_test
   
    opt_predicate = lambda xopt, arg: hasattr(xopt,arg) and getattr(xopt,arg)
    assert len(mopts) <=3 or opt_predicate(mopts[0],'samesplit') , 'only support upto 3 omics now!'
   
    _construct_datasets = _construct_datasets_m_omics
    
    splits_idents = ['splits-common'] if opt_predicate(opt,'common_only') or opt_predicate(mopts[0],'samesplit') else None
    train_dataset = _construct_datasets(split='tr',splits_idents=splits_idents)
    val_dataset   = _construct_datasets(split='val',splits_idents=splits_idents)
    test_dataset  = _construct_datasets(split='te',splits_idents=splits_idents)
    total_dataset = _construct_datasets(split='total',splits_idents=splits_idents)
    return total_dataset, train_dataset, val_dataset, test_dataset

def fetch_datasets(opt, mopts):
    if  len(mopts)>1 and predicate_xopt(opt,'multimodal_data'): 
        g_data_ls = [prepdata_multiomics(opt,mopts)]
    else:
        g_data_ls = [prepdata(opt,m) for m in mopts]

    total_dataset, train_dataset, val_dataset, test_dataset =[],[],[],[]
    for g_data in g_data_ls:
        total_ds,train_ds,val_ds,test_ds = g_data 
        train_dataset.append(train_ds)
        val_dataset.append(val_ds)
        test_dataset.append(test_ds)
        total_dataset.append(total_ds)
    return train_dataset, val_dataset, test_dataset, total_dataset


from .MultiOmicsModel import (load_saved_models,create_multiomics_model)
from .SingleModel     import create_single_models
def load_data_and_models(opt, mopts, device):
    opt.no_balance =True
    train_dataset, val_dataset, test_dataset, total_dataset = fetch_datasets(opt,mopts)
    
    if  len(mopts)> 1 and predicate_xopt(opt,'multimodal_data'): 
        super_model0, models = load_saved_models(opt,mopts,device, multiomics=True)
        super_models = [super_model0]
        opt.no_species = 1
    else:
        _,models = load_saved_models(opt,mopts,device)
        super_models = models 
        opt.no_species = len(mopts)
    return super_models, (train_dataset, val_dataset, test_dataset, total_dataset)

def create_models(opt, mopts, device):
    no_models = len(mopts)
    if no_models > 1 and predicate_xopt(opt,'multimodal_data'): 
        super_model, models = create_multiomics_model(opt,mopts,device)
        super_models = [super_model]
    else:  
        models = create_single_models(opt,mopts,device)
        super_models = models 
    return models, super_models

from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import torch
import re
from os import path
import pandas as pd
from functools import partial
from scipy.stats import kruskal

def predicate_xopt(opt,name):
    if hasattr(opt,name):
         return getattr(opt,name)
    else:
        return False

def calc_nll_loss_wt(target,nclass,thre=4,alpha=0.5):
    wt = torch.tensor([ sum(target == clz) for clz in range(nclass)], dtype=torch.float, device = target.device)
    if wt[0]/wt[1] <=4 or wt[1]/wt[0] <= 4:
        wt = len(target)/wt
    else:
        alpha = 0.5 # arbitrarily
        wt = torch.exp(alpha*len(target)/wt)
    return wt
    
def boxcox_plot(data):
    x = data.x.numpy()
    for i in range(9):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        prob = stats.probplot(x[:,i], dist=stats.norm, plot=ax1)
        ax1.set_xlabel('')
        ax1.set_title('Probplot against normal distribution')
        # ax2 = fig.add_subplot(212)
        # xt, lamb = stats.boxcox(x[:,i] - np.amin(x[:,i])+1)
        # prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
        # ax2.set_title('Probplot after Box-Cox transformation')
        # plt.show()


def boxcox_transform_train(x):
    xt, lamb = stats.boxcox(x - torch.min(x) + 1)
    #lamb = 0
    #xt_torch = xt
    xt_torch = torch.from_numpy(xt).float()
    xt_mean = torch.mean(xt_torch).float()
    xt_std = torch.std(xt_torch).float()
    xt_norm = (xt_torch-xt_mean)/xt_std
    return xt_norm,lamb,xt_mean, xt_std


def boxcox_transform_test(x,lamb, xt_mean, xt_std):
    if lamb == 0:
        y = torch.log(x)
    elif lamb!=0:
        y = ((x-torch.min(x)+1)**lamb-1)/lamb
    else:
        print("lambda is negative!")
        raise ValueError
    res = (y-xt_mean)/xt_std
    return res


def normal_transform_train(x):
    #xt, lamb = stats.boxcox(x - torch.min(x) + 1)
    lamb = 0
    #xt_torch = xt
    xt_mean = torch.mean(x).float()
    xt_std = torch.std(x).float()
    xt_norm = (x-xt_mean)/xt_std
    return xt_norm,lamb,xt_mean, xt_std


def normal_transform_test(x,lamb, xt_mean, xt_std):
    res = (x-xt_mean)/xt_std
    return res

def count_parameters(model):
    for p in model.parameters():
        if p.requires_grad:
            print('Number of parameter', p.numel())
    #return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gene_syms(mopt,gene_columns): #-> ndarray
    gene_syms = None
    if hasattr(mopt,'gene_syms'):
         #gene_syms = pd.read_csv(path.join(mopt.root_prefix1, mopt.gene_syms),header=None)
        gene_syms = pd.read_csv(path.join(mopt.root_prefix1,mopt.gene_syms), header=None) #LY
        gene_syms[1] = gene_syms[1].str.strip() #remove extra space in column 1
        gene_syms.set_index(0,inplace=True) #ens2sym table

    pat = re.compile(r'ENSG\d+')
    if pat.match(gene_columns[0]):
        assert gene_syms is not None # or return None,None
        my_gene_syms =  gene_syms.loc[gene_columns].to_numpy().squeeze()
    else:
        my_gene_syms = gene_columns.to_numpy()
    return my_gene_syms

def get_gene_syms_by_mopt(mopt):
    genes = pd.read_csv(os.path.join(mopt.root_prefix1, mopt.tissue_name+'-gene.csv'), sep=',')
    gene_names = get_gene_syms(mopt, genes.columns[1:])
    return gene_names

def get_gene_prefilter_index(mopt, gene):
    pre_filter_idx = None
    my_gene_syms = get_gene_syms(mopt,gene.columns[1:])
    non_protein_coding = 'NA|RP[0-9]+-.*|KIAA.*|LINC.*|A[LC].*\.[0-9]|.*-AS[0-9]+|.*orf[0-9]+|CT[ABCD]-.*|[AL].+\..+'
    patn = re.compile(non_protein_coding)
    pre_filter_idx = np.array([n for n,sym in enumerate(my_gene_syms,start=1) 
        if isinstance(sym,str) and not patn.match(sym)])

    if hasattr(mopt,'pc_gene'):
        pc_gene = pd.read_csv(path.join(mopt.root_prefix1, mopt.pc_gene),header=None)
        pre_filter_idx = np.array([n for n,sym in enumerate(gene.columns[1:].to_numpy(),start=1) 
                                   if sym in pc_gene.iloc[:,0].to_numpy()])
    return pre_filter_idx, my_gene_syms

def calc_kruskal(x:np.ndarray, t: np.ndarray, axis =0):
    if axis !=0:
        x = x.T
    kr = []
    for i in range(x.shape[1]):
        d = x[:,i]
        g = [d[t == c] for c in np.unique(t)]
        if np.unique(np.concatenate(g).flatten()).shape[0] == 1:
            st, pvalue = 0, 1 
        else:
            st, pvalue = kruskal(*g) 
        kr.append(st)
    return np.array(kr)
    
def calc_traitcorr(target,gene,anno,typ=-1):
    y = anno[target].to_numpy()
    traitcorr =  ( np.array([np.corrcoef(x,y)[0,1] for x in gene.iloc[:,1:].to_numpy().T]) 
        if typ == -1 else calc_kruskal(gene.iloc[:,1:].to_numpy(), t=y) )
    return traitcorr

def get_gene_index_by_sorting(mopt, gene_tr, anno_tr, pre_filter_idx=None):
    sorting_score_fn = np.var
    def make_var_corr(trait_type, trait):
        def var_corr_cont(x:np.ndarray, t:np.ndarray, axis =0): #x is a matrix, t is a vector
            return np.sqrt(np.var(x,axis=axis)) * np.abs(np.corrcoef(x, t.reshape(-1,1),row_var=False)[-1,:-1])
        def var_corr_cat(x:np.ndarray, t: np.ndarray, axis =0):
            return np.sqrt(np.var(x,axis=axis)) * np.abs(calc_kruskal(x, t, axis))
        return partial(var_corr_cat if trait_type == 0 else var_corr_cont, t=trait)

    def calc_sortingfactor(gene, nofbin=20, pre_filter_idx=None):
        if pre_filter_idx is None:
            sortingfactor =  sorting_score_fn(gene.iloc[:,1:].to_numpy(),axis=0)
        else: #np.array
            sortingfactor =  sorting_score_fn(gene.iloc[:,pre_filter_idx].to_numpy(),axis=0)
        max_factor = max(sortingfactor)
        min_factor = min(sortingfactor)
        threshold = (max_factor -min_factor)/nofbin + min_factor
        gene_idx = np.argwhere(sortingfactor > threshold).squeeze()
        if pre_filter_idx is not None:
            gene_idx = pre_filter_idx[gene_idx] -1 
        return gene_idx

    def calc_sortingfactor_gn(gene,nofgene, pre_filter_idx=None):
        if pre_filter_idx is None: 
            sortingfactor =  sorting_score_fn(gene.iloc[:,1:].to_numpy(),axis=0)
        else: #np.array
            sortingfactor =  sorting_score_fn(gene.iloc[:,pre_filter_idx].to_numpy(),axis=0)
        gene_idx = np.argpartition(-sortingfactor,nofgene)[:nofgene]
        if len(sortingfactor) < nofgene:
            print(f'warning selected gene no is {len(sortingfactor)} while nogene in ini specified as {nofgene}')
        if pre_filter_idx is not None:
            gene_idx = pre_filter_idx[gene_idx] - 1 
        return gene_idx 
    
    if hasattr(mopt,'sorting_method') and mopt.sorting_method == 'varcorr':
        print('sorting method:', 'varcorr')
        sorting_score_fn = make_var_corr(mopt.targettype,anno_tr[mopt.target_])
    if hasattr(mopt,'nofbin'):
        print('select genes by nofbin',mopt.nofbin)
        gene_idx= calc_sortingfactor(gene_tr,nofbin=mopt.nofbin, pre_filter_idx=pre_filter_idx)
    elif hasattr(mopt,'nofgene'):
        print('select genes by nofgene', mopt.nofgene)
        gene_idx= calc_sortingfactor_gn(gene_tr,mopt.nofgene,pre_filter_idx=pre_filter_idx)
    else:
        print('select genes by nofbin (default 10)')
        gene_idx= calc_sortingfactor(gene_tr,pre_filter_idx=pre_filter_idx)
    return gene_idx

######
import h5py
def get_permute_tab_path(xopt):
    target_permute_tab_fname = f'{xopt.tissue_name}-{xopt.target_}-target-permute-tab.h5'
    target_permute_tab_path = os.path.join(xopt.dataroot_,target_permute_tab_fname)
    return target_permute_tab_path

def create_and_save_permute_tab(xopt, sz):
        target_permute_tab_path = get_permute_tab_path(xopt)
        permutes = [np.random.permutation(sz) for _ in range(500)] #len(gene):nof sample
        permutes = np.stack(permutes)
        with h5py.File(target_permute_tab_path,'w') as ht:
            ht.create_dataset('permute_tab',data = permutes)

def load_from_permute_tab(mopt, num):
    permute_tab_path = get_permute_tab_path(mopt)
    with h5py.File(permute_tab_path,'r') as pf:
        tab_sz = pf['permute_tab'].shape[0]
        permute = pf['permute_tab'][num % tab_sz]
    return permute

import os
def get_perm_no(opt):
    return opt.permute_no

def get_projdir(opt):
    projdir_lst2 = opt.project_name.split('/')[:-2] 
    return os.path.join(*projdir_lst2)

def get_proj_basename(opt):
    projdir_lst = opt.project_name.split('/')
    return projdir_lst[-3]

def get_explain_file_path(opt, mopt, perm_no, exp_mode, perm_ref=None):
    projdir_lst2 = opt.project_name.split('/')[:-2] 
    if perm_no >0 :
        runseq = f'{opt.ref_runseq}-perm-{perm_no}'
    else:
        runseq = str(opt.runseq)

    perm_ref = opt.nt if perm_ref is None else perm_ref
    bd_suf = f'-{mopt.raw_graph_dirname}' if (opt.igf_bdsuffixed and exp_mode != 'ih') else ''
    return os.path.join(*projdir_lst2, runseq, 
            f'{exp_mode}{perm_ref}p{perm_no}-model.{mopt.id_}-{mopt.tissue_name}-{mopt.species}{bd_suf}.h5')

def get_explain_int_path(typ, opt, mopt, perm_no, exp_mode,  fcdim):
    projdir_lst2 = opt.project_name.split('/')[:-2] 
    fn = os.path.join(*projdir_lst2, str(perm_no),  f'{exp_mode}{opt.nt}p{perm_no}-model.{mopt.id_}-{mopt.tissue_name}-{mopt.species}-{typ}-{fcdim}.h5')
    return fn

def get_gene_idx_path(xopt, typ):
    gene_idx_fname = f'{xopt.tissue_name}-{xopt.target_}-{typ}_idx.h5'
    gene_idx_path = os.path.join(xopt.dataroot_,'raw',gene_idx_fname)
    return gene_idx_path

def get_split_idx_path(xopt, split='splits'):
    split_name =  f'{xopt.tissue_name}-{xopt.target_}-{split}-index.h5'
    split_path = os.path.join(xopt.dataroot_, 'raw', split_name)
    return split_path


from .search_utils import make_path,get_params
def get_eval_result_path(opt,xopts):
    
    situ = 1
    EXP_PATH,BEST_PATH, *_ = make_path(situ,opt,xopts[0],False)   
    if opt.whichbest == -1:
        load_params = pd.read_csv(BEST_PATH, sep='\t', header=0,names=['seq', 'params', 'best_epoch', 'val_loss', 'tr_loss', 'val_corr', 'val_lossc'])
        loaded_bestparams, seq = get_params(opt,load_params)
        opt.whichbest = seq
    tissue_list = [xopt.tissue_name for xopt in xopts]
    prefix='eval' if not opt.inf_total_dataset else 'eval-total'
    save_path = os.path.join(EXP_PATH, f"{prefix}-{'-'.join(tissue_list)}-p{opt.whichbest}.h5")
    return save_path

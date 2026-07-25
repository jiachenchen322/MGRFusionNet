import os
import sys
import argparse
import configparser
import ast
import numpy as np
import pandas as pd
import re
from collections import defaultdict

from utils.utils import predicate_xopt

pred_hasattr = predicate_xopt

def dict_to_str(dictvar):
    s='{'
    for k,v in dictvar.items():
        if type(v) == str:
            s+=  f'\'{k}\':\'{v}\',\n'
        else:
            s+= f'\'{k}\':{v},\n'
    s+='}'
    return s
def list_to_str(lstvar):
    s='['
    for elm in lstvar:
        if type(elm) == str:
            s+= f'\'{elm}\',\n'
        else:
            s+=f'{elm},\n'
    s+=']'
    return s

class InputClass:
    section_names = ['project','data','model','GlobalOptions']
    def _check_for_substitutions(self,val):
        #allow using ${key} in dict values, and subsitition happens here.
        sub = re.compile(r'\$\{([a-zA-Z][\w_]*)\}')
        if type(val) == dict:
            for k, v  in val.items():
                if type(v) != str: continue
                r = sub.search(v)
                if r and r.group(1) in val.keys():
                    val[k] = sub.sub(str(val[r.group(1)]), v)

    def _populate_opt(self, opt, config, sect):
        if sect not in config.keys(): return
        for k in config[sect]:
            if config[sect][k] != '' :
                    setattr(opt, k, ast.literal_eval(config[sect][k]))
            else:
                    setattr(opt, k, '')
            self._check_for_substitutions(getattr(opt,k))
        
    def __init__(self, config=None, bypass_check=False):
        if config==None : return
        self.bypass_check = bypass_check
        self.glbl_sections = ('project','GlobalOptions','model.alignment')
        self.gOpt = InputClass()
        self._populate_opt(self.gOpt,config,'project')
        self._populate_opt(self.gOpt,config,'GlobalOptions')
        self._populate_opt(self.gOpt,config,'model.alignment')
        self._populate_opt(self.gOpt,config,'model.default')
        self.model_default_keys = config['model.default'].keys() if 'model.default' in config.keys() else list()
        sections = config.sections()
        self.mOpt = defaultdict(InputClass)
        for s in sections:
            mch = re.match('(data|model)\.(\d+)',s)
            if mch:
                k = mch.group(2)
                self._populate_opt(self.mOpt[k], config, s)
                self.mOpt[k].id_ = int(k)  

def info(quite, *arg, **args):
    if not quite:
        print(*arg, **args)

def paramparser(choice=None, cfg_file=None, quite=True, **extra):

    if cfg_file is not None and os.path.exists(cfg_file):
        opt = InputClass()
        opt.config_path = cfg_file
        opt.use_config  = 'True'
        opt.target_permute =False
        opt.permute_no = 0
        opt.runseq     = 0
        opt.explain_layer = 'none'
        opt.whichbest  = -1
        for k,v in extra.items():
            setattr(opt,k,v)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--use_config', type=str, default='True', help='''
        Read input variables from config .ini file instead of from bash script
        please see configs/protein_encoder_template.ini for config file format.
        ''')
        parser.add_argument('--config_path', type=str, default='', help='''
        Path to .ini config file. Only reference if use_config == True''')
        parser.add_argument('--target_permute',nargs='?', const=True, default=False,help='whether to permute target')
        parser.add_argument('--permute_no',type=int,default=0)
        parser.add_argument('--runseq',type=str,default='0')
        parser.add_argument('--explain_layer',type=str,default='none')
        parser.add_argument('--oldcfg', type=str,default='')
        parser.add_argument('--newcfg', type=str, default='')
        parser.add_argument('--ADflag', action='store_true', default=False)
        parser.add_argument('--whichbest',type=int,default=-1)
        parser.add_argument('--fold', type=int, default=argparse.SUPPRESS)
        parser.add_argument('--train_verbose', action='store_true', default=False)
        parser.add_argument('--talos_start',type=int, default=argparse.SUPPRESS)
        parser.add_argument('--multiprocess',nargs='?', type=int, const='10000', default=argparse.SUPPRESS)
        parser.add_argument('--fixed_seed', action='store_true', default=argparse.SUPPRESS)
        parser.add_argument('--single_model',type=int, default=argparse.SUPPRESS)
        parser.add_argument('--project_prefix',type=str,default=argparse.SUPPRESS)
        parser.add_argument('--ref_runseq',nargs='?',type=str, default= argparse.SUPPRESS) #reference runseq for target_permute runs
        parser.add_argument('--runseqs',type=str,default=argparse.SUPPRESS)
        parser.add_argument('--avg_exp_mode', type=str,default=argparse.SUPPRESS) #used in avg_explain_scores
        parser.add_argument('--avg_split_runseqs', action='store_true', default=False) #split runseqs into folds in avg_explain_scores
        parser.add_argument('--veri',action='store_true',default = False)
        parser.add_argument('--common_only',action='store_true',default=False)
        parser.add_argument('--load_common',action='store_true',default=False)
        parser.add_argument('--static_multiomics_loader', action='store_true', default =False)
        parser.add_argument('--m_omics_correlation', action='store_true', default=argparse.SUPPRESS)
        parser.add_argument('--fdr_permute',type=int, default=argparse.SUPPRESS)
        parser.add_argument('--perm_ref',type=str,default=argparse.SUPPRESS,  
           help='''for FDR use only; to specify reference modality model where target permuted ig score come from; 
                   (it may be different from the model of original ig score)
            _t1_: single modality model,
            _t2_: two modality model,
            _t2_com_: two modality model use dataset common only''') 
            # use which model's permute to calc fdr, eg '_t2_com_'
        parser.add_argument('--explain_class_ids', type=int, default=argparse.SUPPRESS)
        parser.add_argument('--igf_bdsuffixed', action='store_true',default=True)
        parser.add_argument('--inf_total_dataset', action='store_true', default=False)
        opt = parser.parse_args()

    info(quite,f'current work dir : {os.getcwd()}')
    if len(opt.config_path) == 0: 
        return None,None,None,None,opt
    
    def update_or_set_defaults_to_final_opt(opt):
        def opt_default(par, val=None):
            if type(par)==str: 
                if not hasattr(opt,par):
                    setattr(opt,par,val)
            elif type(par)==dict:
                for p,v in par.items():
                     if not hasattr(opt,p):
                        setattr(opt,p,v)
        
        
        info(quite,'Loading Variables from config file...')
        config = configparser.ConfigParser()
        config.read_file(open(opt.config_path))

        opt0 = opt 
        bypass_check = opt.bypass_check if hasattr(opt,'bypass_check') else False
        opt = InputClass(config,bypass_check) 
        mopt =  [opt.mOpt[k] for k in sorted(opt.mOpt.keys(), key=int)] 
        model_default_keys = opt.model_default_keys
        opt = opt.gOpt    
        opt.cfg_file = opt0.config_path

        cmdline_only_pars = [ 'ref_runseq','static_multiomics_loader','runseqs', 'igf_bdsuffixed', 'inf_total_dataset',
            'target_permute', 'permute_no', 'runseq', 'whichbest','train_verbose','talos_start','single_model','common_only','perm_ref',
            'load_common']
        for par in cmdline_only_pars:
            if hasattr(opt0, par):
                setattr(opt, par, getattr(opt0,par))

        if hasattr(opt,'single_model') and opt.single_model is not None: 
            opt.multimodal_data =False
            opt.fc1_detached = False
            opt.m_omics_correlation = False
            opt.m_omics_correlation_all = False
            opt.epochsize = None
            mopt = [mopt[opt.single_model]]
            if hasattr(opt,'tissue_list'):
                opt.tissue_list = [opt.tissue_list[opt.single_model]]
           
        if opt.target_permute and opt.permute_no >0:
            assert hasattr(opt,'ref_runseq'), " please provide ref_runseq from commanline arguments"
            opt.explain_ds ='val'
            opt.runseq = f'{opt.ref_runseq}-perm-{opt.permute_no}'

        cmdline_override_pars = ['fold','multiprocess','fixed_seed','fdr_permute','explain_class_ids']
        for par in cmdline_override_pars:
            if hasattr(opt0, par):
                setattr(opt,par,getattr(opt0,par))

        cfg_prior_cmdline_pars = ['explain_layer']
        for par in cfg_prior_cmdline_pars:
            if not hasattr(opt,par):
                setattr(opt, par, getattr(opt0,par))
       
        if hasattr(opt0,'project_prefix'):
            if hasattr(opt,'project_prefix'):
                opt.project_prefix = f'{opt.project_prefix}-{opt0.project_prefix}'
            else:
                opt.project_prefix = opt0.project_prefix
        opt.project_name_ = opt.project_name 
        if hasattr(opt,'project_prefix'):
            opt.project_name = f'{opt.project_out_path}/{opt.project_prefix}/{opt.project_name}/{opt.runseq}/ck'
        else:
            opt.project_name = f'{opt.project_out_path}/{opt.project_name}/{opt.runseq}/ck'

        opt_default('support_attn', -sys.maxsize-1)
        opt_default('kfold',3)
        opt_default('fold',0)
        opt_default('from_s1',1)
       
        opt_default('use_nn_lookuptable',True)
        opt_default('threesplit',0)
        opt_default('min_epoch',25)
        opt_default('m_omics_correlation',False)
        opt_default('lamb00',1)
        opt_default('stage2',0)
        opt_default('specific_sam_update_emb', True)
        opt_default('epochsize',None)
        opt_default('adding_noise',False)
        opt_default('fdr_thr',0.05)
        opt_default('xdim1',16)
        opt_default('xdim2',0)
        opt_default('xdim_ff',128)
        opt_default('absolute_corr', False)
        opt_default('m_omics_correlation_mode','modality')
        opt_default({  
            'target':0,         
            'qr':False, 
            'memp':1,
            'min_epoch': 20, 
            'no_balance':True,
            'all_together':True,
            'ws': False,
            'load_params_selector':'val_corr',
            'threesplit': 0,
            'linet_softmax':0,
            'situation':1,
            'frac_limit':1,
            'pnorm':0,
            'nodecay_lastfc': 0,
            'whichdevice':0,
            'save_intermediate_gc':1,
            'save_intermediate_rep':1,
            'normalization':False,
            'save_best': True,
            'save_score': False,
            'whichbest':-1,
            'mempool_epoch_update':True,
            'inc_global':0,
            'm_omics_correlation':True,
            'm_omics_correlation_all':True,
            'bin_edge':0 ,
            'last_gcn_num_nodes':300,
            'multimodal_data':True,
            'batchsize': 16,
            'dim_ff': 128,
            'heads_transformer':2,
            'n_transformer_layer': 2,
            'epochsize':15,
            'stepsize': 5,
         })

        if hasattr(opt,'tissue_list'):
            opt.tissues = opt.tissue_list
            no_tissues = len(opt.tissues)
            assert no_tissues == len(mopt), "number of tissues doesn't match number of models"
            for i, tissue in enumerate(opt.tissues):
                mopt[i].tissue_name = tissue
            opt.no_tissues = no_tissues
        else:
            opt.tissues = []
            for i, m in enumerate(mopt):
                opt.tissues.append(m.tissue_name)
            opt.no_tissues = no_tissues = len(mopt)
            opt.tissue_list = opt.tissues
         
        if no_tissues == 1:
            opt.nt = '_t1_'
        else:
            opt.nt = '_t' + str(no_tissues) +  '_'
        
        if predicate_xopt(opt,'common_only'):
            opt.nt = f'{opt.nt}com_'

        if hasattr(opt,'species_omics_tab') and len(opt.species_omics_tab) >0:
            opt.mopt_list_ = []
            for s in opt.species_omics_tab:
                m = [mopt[i] for i in s]
                opt.mopt_list_.append(m)
            opt.no_species_ = len(opt.mopt_list_)
               
        elif pred_hasattr(opt,'multimodal_data'):
            opt.no_species_ = 1
            
        else:
            opt.no_species_ = len(mopt)

        assert not (pred_hasattr(opt,'multimodal_data')
             and (hasattr(opt,'species_omics_tab') and len(opt.species_omics_tab)>0)),\
                 'conflict options: multimodal_data and species_omics_tab!'

        if hasattr(opt,'wvote') and opt.wvote == 1:
            opt.transformer = 0
            opt.memp = 0
        if hasattr(opt,'transformer') and opt.transformer >0:
            opt.wvote = 0
            opt.memp = 1

        if not pred_hasattr(opt,'multimodal_data'):
            opt.vcdn_embedding = False

        return opt0, opt, mopt, model_default_keys

    opt0, opt, mopt, model_default_keys = update_or_set_defaults_to_final_opt(opt)

    for m in mopt:
        if hasattr(m,'struct_dict'):
            for k, v in m.struct_dict.items():
                setattr(m, k, v)
        if hasattr(m,'data_dict'):
            for k, v in m.data_dict.items():
                setattr(m, k, v)
        if hasattr(opt,'struct_dict'):
            for k, v in opt.struct_dict.items():
                setattr(m, k, v)
        if hasattr(opt,'data_dict'):
            for k, v in opt.data_dict.items():
                setattr(m, k, v)

    params = [] 
    for i, m in enumerate(mopt):
        if hasattr(m, 'talos_dict'):
            t_dict = m.talos_dict.copy()
            for k,v in m.talos_dict.items():
                assert isinstance(v, list)
                if len(v) == 1: 
                    setattr(m,k,v[0])
                    del(t_dict[k])
            del(m.talos_dict)
            params.append(t_dict) 
        else:
            params.append(dict())
        m.talos_sno = i

    if hasattr(opt,'common_talos_dict') : 
        ctdict = opt.common_talos_dict.copy()
        for k,v in ctdict.items():
            if len(v) == 1: 
                if hasattr(opt,'global_talos_names') and k in opt.global_talos_names:
                    setattr(opt,k,v[0])
                else:
                    for m in mopt: 
                        setattr(m,k,v[0])
                del(opt.common_talos_dict[k]) 
            elif (hasattr(opt,'single_model') and opt.single_model is not None) or predicate_xopt(opt,'common_only') or predicate_xopt(mopt[0],'samesplit'):
                if k=='align_coeff_multiomics2' or k=='align_coeff_multiomics' or k=='heat' or k=='stage2' or k=='epochsize':
                    del(opt.common_talos_dict[k])

        cparams = opt.common_talos_dict
    else:
        cparams = None  

    moptions = ['stepsize','gamma','pnorm','sorting_method','nclass','net_norm','explain_ds',
                    'memp','inc_global','adj_normalize','threesplit', 'kfold','fold',
                    'general_c_loss', 'init_type', 'support_attn','use_nn_lookuptable'] + list(model_default_keys) 
    zoptions = ['l1_reg', 'fold', 'no_mostrelated','pnorm', 'rnorm','grpnorm_lambda','bin_edge']
    nzoptions = {'rdbinarize':False,'net_norm':False,'inc_global':True, 'nofilter':False,
                    'mempool_residual':True, 'memp':1,
                    'optimizer':'Adam',
                    'poolmethod':'topk',
                    'convmethod':'LI_Net',
                    'fc':3,
                    'gc':2,
                    'exp_mode':'ig',
                    'fdr_thr':0.05, 
                    'explain_class_ids': [0,1],
                    'explain_ds': 'val',
                    'actfnparm' : 0.7,
                    'align_coeff_multiomics2': 0.5,
                    'lr': 0.0025,
                    'dim1':32,
                    'dim12':24,
                    'dimmp':16,
                    'dim2':16,
                    'dim3':16,
                    'dim4': 1,
                    
                    'weightdecay':0,
                    'gamma': 0.98,
                    'nocommu':6,
                    'heads':1,
                    'grpnorm_lambda': 0,
                    'dropout':0.2,
                    'ratio':1, # 
                    'ratio2':1,
                    'heat':0, # for rp
                    'bnfg':1,	
                    'lamb7':0,
                    'lamb8':0.1, # set to 0 if bnfg=0
                    
                    'target_list':'labels',
                    'cv_list':'',
                    'edge_pernode': 8,
                    'rep':1,
                    'best_epoch':-1,
                    'activationtype':3,
                    'lamb0':1,
                    'lamb1':0,
                    'lamb2':0,
                    'lamb4':0.1,
                    }
                    
    for m in mopt:
        for elem in set(moptions):
            if hasattr(opt,elem):
                setattr(m, elem, getattr(opt, elem))
        for elem in zoptions:
            if not hasattr(m,elem):
                setattr(m, elem, 0)
        for elem,val in nzoptions.items():
            if not hasattr(m,elem):
                setattr(m, elem, val)

        if hasattr(opt,'common_only') and opt.common_only: # run commonsam from imcomplete cfg file!
            if hasattr(m,'heat'):
                del(m.heat)
            if hasattr(opt,'align_coeff_multiomics2'):
                del(m.align_coeff_multiomics2)
            setattr(m,'m_omics_correlation',False)
            setattr(m,'m_omics_correlation_all',False)
            setattr(m,'heat',0)
            setattr(m,'stage2',0)

        m.target_list = ([str(i) for i in m.target_list.split(',')] 
                        if type(m.target_list) == str else m.target_list)
        m.cv_list =  ([str(i).strip() for i in m.cv_list.split(',') if i.strip() != '' ]
                if type(m.cv_list) == str else m.cv_list)
        m.dim_cv_list = len(m.cv_list)
        m.target_ = m.target_list[0]
        if hasattr(opt,'target') and (not hasattr(m,'targettype')):
            m.targettype = opt.target

    if hasattr(opt,'whichtissue'):
        mopt = [mopt[i] for i in opt.whichtissue]
        params = [params[i] for i in opt.whichtissue]
        opt.tissues = opt.tissue_list = [opt.tissues[i] for i in opt.whichtissue]
        opt.no_tissues = len(opt.whichtissue)

    if choice == 'gnn':
        for m in mopt:
            if hasattr(m, 'graph_file'):
                raw_graph_dirname = f"{os.path.basename(m.graph_file).split('.')[0]}-genes"
            else:
                raw_graph_dirname = 'genes'
            nodeth_f = os.path.join(m.data_base_path,f'{m.tissue_name}-{m.target_}-{raw_graph_dirname}-nodeth.txt')
            info(quite, nodeth_f)
            if os.path.exists(nodeth_f):
                nodeth_df = pd.read_csv(nodeth_f)
                m.nogenes = int(nodeth_df['nogenes'][0]) 
                m.node_th  = float(nodeth_df['nodeth'][0])
                info(quite, f'GET nogene:{m.nogenes}, nodeth:{m.node_th}')   
            datap1=f'{m.tissue_name}-{m.target_}-no_mostrelated0'
            datap2=f'{raw_graph_dirname}-{m.nogenes}'
            m.raw_graph_dirname = raw_graph_dirname
            m.dataroot_ = os.path.join(m.data_base_path, datap1, datap2) 
            info(quite, raw_graph_dirname)
            info(quite, m.dataroot_)
            info(quite, "")
            if hasattr(m,'inf_data_base_path'):
                m.inf_dataroot_ =  os.path.join(m.inf_data_base_path, datap1, datap2)
            if hasattr(opt,'last_gcn_num_nodes'):
                ratio = np.power(opt.last_gcn_num_nodes / m.nogenes, 1/m.gc)
                m.ratio = m.ratio2 = ratio if ratio <1.0 else 1.0

    all_params_empty = True
    for p in params:
        if len(p) !=0:
            all_params_empty = False
            break
    if all_params_empty:
        if len(cparams) ==0:
            cparams['dummy'] = [True]    

    return opt, mopt, params, cparams, opt0


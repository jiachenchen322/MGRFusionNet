from utils.search_utils import make_path, get_params
import pandas as pd
import ast,os
import torch
import re
from model.gnn_networks import NNGAT_Net

def load_best_params(opt, LOAD_BEST_PATH) :
    load_params = pd.read_csv(LOAD_BEST_PATH, sep='\t', header=0,names=['seq', 'params', 'best_epoch', 'val_loss', 'tr_loss', 'val_corr', 'val_lossc'])
    if opt.whichbest == -1:
        #loaded_bestparams = load_params['params'].values[-1]
        loaded_bestparams, seq = get_params(opt,load_params)
        opt.whichbest = seq
    else:
        seq = opt.whichbest
        loaded_bestparams = load_params.iloc[seq]['params']
    loaded_bestparams = loaded_bestparams.rstrip('!')  
    params = dict(item.split(":") for item in loaded_bestparams.split("$"))
    for k in params.keys():
        params[k] = ast.literal_eval(params[k])
    
    #params['best_epoch'] = -1
    params['best_epoch'] = load_params.iloc[seq]['best_epoch']
    params['whichbest'] = opt.whichbest
    print('Loaded best params:', params)
    return params,loaded_bestparams,seq

def load_model(opt,mopt,device,load_checkpoint=True):
    _,_,_, LOAD_EXP_PATH, LOAD_BEST_PATH = make_path(3,opt,mopt)
    print('model Loading path:', LOAD_EXP_PATH)
    print('param Loading path:', LOAD_BEST_PATH)
    params,loaded_bestparams,seq = load_best_params(opt,LOAD_BEST_PATH)
    
    for k,v in params.items():
        if hasattr(opt,'global_talos_names') and k in opt.global_talos_names:
            setattr(opt, k,v)
            continue
        setattr(mopt,k,v)

    checkpoint = torch.load(os.path.join(LOAD_EXP_PATH, str(seq), "model.pt"), map_location=device)
    model_state_dict = {}
    for k,v in checkpoint['model_state_dict'].items():
        if 'gnn_plus_mempool' in k:
            k = re.sub('gnn_plus_mempool','gnn_plus_mpool',k) 
        if 'embedding.gnn_plus_mpool.x_mapping.0.module' in k:
            k = re.sub('embedding\.gnn_plus_mpool\.x_mapping\.0\.module','embedding.gnn_plus_mpool.x_mapping.0',k)
        if 'embedding.bn1' in k:
            k = re.sub('embedding\.bn1', 'embedding.fc1_bn',k)
        if 'bns' in k:
            k = re.sub('\.bns', '.clf_bns',k)

        model_state_dict[k] = v
    checkpoint['model_state_dict'] = model_state_dict

    model = NNGAT_Net(opt, mopt).to(device)
    if os.path.exists(os.path.join(LOAD_EXP_PATH, str(seq), "ctrl-avg-x.pt")): 
        model.explain_ref_x = torch.load(os.path.join(LOAD_EXP_PATH, str(seq), "ctrl-avg-x.pt"))
        
    if load_checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.checkpoint = checkpoint
    model.loaded_bestparams = loaded_bestparams
    model.mopt = mopt
    mopt.model_ = model
    return model, loaded_bestparams

import numpy as np
from sklearn.metrics import roc_auc_score

from utils.utils import  get_explain_file_path,  get_perm_no

import h5py
from explanation_code.do_explain import do_explain
from collections import defaultdict

def run_validate(opt, model, dataset, check_only=True, verbose=False):
    mopts = model.mopts if hasattr(model, 'mopts') else [model.mopt]
    exp_mode = mopts[0].exp_mode if hasattr(mopts[0],'exp_mode') else 'ig'
    explain_class_ids = mopts[0].explain_class_ids
    
    svxs = [defaultdict(list) for _ in range(len(mopts))]
    sams = [defaultdict(list) for _ in range(len(mopts))]

    if not isinstance(dataset,(tuple,list)):
        dataset = [(dataset, mopts)]
    exp_corrects = []
    pred_loss = []
    preds_lst = []
    yall_lst  = []
    metric_per_mod = dict()
    for ds in dataset: #ds =  (data, mopt_lst)
        svx_scores, sam_lst, corrects, loss, preds, yall = do_explain(opt, model, ds, explain_class_ids, exp_mode = exp_mode, check_only=check_only, verbose=verbose)
        if corrects is not None:
            exp_corrects.append(corrects)
            pred_loss.append(loss)
            preds_lst.append(preds)
            yall_lst.append(yall)
            mopt_lst = ds[1]
            if len(mopt_lst) == 1:
                metric_per_mod[f'loss${mopt_lst[0].tissue_name}'] = loss
                metric_per_mod[f'acc${mopt_lst[0].tissue_name}'] = corrects[0]/corrects[1]
            else:
                metric_per_mod['loss$multi'] = loss
                metric_per_mod['acc$multi'] = corrects[0]/corrects[1]
        if svx_scores is None: #inference only
            continue
        # scores: list(m) of dict(clz) of list(nsample) of scores
        # samids:            dict(clz) of list(nsample) of string
        for j, m in enumerate(ds[1]): # dispatch scores to models
            midx = mopts.index(m)
            for clz in explain_class_ids:
                svxs[midx][clz].extend(svx_scores[j][clz])
                sams[midx][clz].extend(sam_lst[clz])
    if check_only: # inference only
        correct_num, dset_len = list(zip(*exp_corrects))
        loss_avg = sum(np.array(pred_loss))/sum(dset_len)
        raw_acc  = sum(np.array(correct_num))/sum(dset_len)
        preds    = np.vstack(preds_lst)
        yall     = np.vstack(yall_lst)
        # compute AUC for binary classification
        try:
            probs = np.exp(preds)[:, 1]
            auc = roc_auc_score(yall.squeeze(), probs)
        except Exception:
            auc = raw_acc
        res = {'acc':auc, 'raw_acc':raw_acc, 'lcall':loss_avg,
            'pred_y': preds,'y_all': yall,
            'metrics_per_mod': metric_per_mod}
        return res

    if len(exp_corrects) >0:
        tot_corrects = np.sum(np.vstack(exp_corrects),axis = 0)
        acc = tot_corrects[0]/tot_corrects[1]
        print(tot_corrects, acc)
    for mopt, svx_scores, sam_lst in zip(mopts,svxs,sams):
        fn = get_explain_file_path(opt, mopt, get_perm_no(opt), exp_mode)
        print('explain result ->', fn)
        dt = h5py.string_dtype(encoding='ascii')
        with h5py.File(fn,'w') as svxf:
            if len(explain_class_ids) >0: #svx_scores is a dictionary of list of ndarray
                for clz,scores in svx_scores.items():
                    scores = np.vstack(scores)
                    samids = sam_lst[clz]      
                    svxf.create_dataset(f'class-{clz}',data=scores)
                    svxf.create_dataset(f'samids-{clz}',(len(samids),),
                        dtype=dt,data=[s.encode('ascii','ignore') for s in samids])
            else: #svx is a list of ndarray
                scores = np.vstack(svx_scores)
                samids = sam_lst
                svxf.create_dataset('regression',data=scores)
                svxf.create_dataset('samids',(len(samids),), dtype=dt,
                    data= [s.encode('ascii','ignore') for s in samids])

def remap_index(l_oldsamid,l_newsamid,l_idx):
    #return [ l_newsamid.index(l_oldsamid[idx]) for idx in l_idx ]
    samids = [l_oldsamid[idx] for idx in l_idx]
    sidx_lst = []
    for s in samids:
        try:
            sidx_lst.append(l_newsamid.index(s))
        except ValueError: #value not found
            pass
    return sidx_lst

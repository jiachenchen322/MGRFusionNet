import os
import sys
import shutil
import copy
import torch
import random
import numpy as np

import torch_geometric
if torch_geometric.__version__ < '2.0':
    from torch_geometric.data import DataLoader
else:
    from torch_geometric.loader import DataLoader

from model.graph_dataset import DomainGraphDataset
from utils.search_utils import make_saveall_info, make_path, calc_confmat_from_res, short_path
from utils.utils import predicate_xopt
from platformX.trainlibv2 import train
from platformX.params import paramparser
from platformX.prepdata import FileLocker, fetch_datasets, create_models
from functools import reduce
from itertools import product
from utils.validation_utils import run_validate
from collections import defaultdict
from utils.torch_history import TorchHistory
from tqdm import tqdm

class Done1(Exception):
    pass
class ErrorMp(Exception):
    pass
                           
def flatten_params(params, cparams, xclusion):
    flat_params = dict()
    if isinstance(params, list):
        for n in range(len(params)):
            for k,v in params[n].items():
                flat_params[f'{n},{k}'] = v
    else:
        for k,v in params.items():
            flat_params[f'0,{k}'] = v
    if cparams:
        for k,v in cparams.items():
            if k in xclusion:
                continue
            flat_params[f'c,{k}'] = v        
    return flat_params 

def pack_params(flat_params, parlst_len): 
    parlst = [dict() for _ in range(parlst_len)]
    for k,v in flat_params.items():
        kext = k.split(',')
        if kext[0] == 'c':
            for par in parlst:
                par[kext[1]] = v
        else :
            parlst[int(kext[0])][kext[1]] = v
    return parlst

def add_params_to_opts(opt,mopts,parlst):
        def process_align_coeff(v): 
            v = [float(x) for x in v.strip().split('-')]
            return v
        
        special_settings = {'align_coeff':  process_align_coeff, 'align_coeff_multiomics':  process_align_coeff}
       
        for i,p in enumerate(parlst):
            xopts = [] 
            for m in mopts:
                if m.talos_sno == i:
                    xopts.append(m)

            for k,v in p.items():
                if k in special_settings.keys():
                    setattr(opt, k, special_settings[k](v))
                elif hasattr(opt, 'global_talos_names') and k in opt.global_talos_names: 
                    setattr(opt, k, v)
                else:
                    for m in xopts:
                        setattr(m,k,v)

def params_toString(params):     
    paramlist = ['{}:{}'.format(k,params[k]) for k in sorted(params.keys())]
    cur_params = '$'.join(paramlist)
    return cur_params

class DummyModule(torch.nn.Module, TorchHistory):
    def __init__(self):
        super(DummyModule, self).__init__()
        self.init_history() 
        self.lin = torch.nn.Linear(1,2)

class mySimpleTalos:
    def __init__(self,x, y, x_val, y_val, mk_model, config):
        self.x = x
        self.y = y
        self.x_val = x_val
        self.y_val = y_val
        self.mk_model = mk_model
        self.config_cartesian_product = [dict(zip(config, v)) for v in product(*config.values())]

    def scan(self):
        print('hyper parameter scan len', len(self.config_cartesian_product))
        for p in tqdm(self.config_cartesian_product):
            print('p->',p)
            self.mk_model(self.x,self.y,self.x_val,self.y_val,p)

    @staticmethod
    def Scan(x, y, x_val, y_val,
            params, model, experiment_name):
        talos = mySimpleTalos(x, y, x_val, y_val,mk_model=model, config=params)
        talos.scan()

def make_model(opt, mopts, device, Sentinel, Q1, Q2):
    best_best_loss = 1e10  
    best_best_acc = -10 
    best_params = None 
    talos_seq = 0

    if hasattr(opt,'talos_start'):
        import pandas as pd
        print(f'Warm start @ talos seq: {opt.talos_start}', file=sys.stderr)
        _, BEST_PATH,_ ,_, _ = make_path(1,opt,mopts[0])
        loaded_best =  pd.read_csv(BEST_PATH, sep='\t', header=0,
                    names=['seq', 'params', 'best_epoch', 'val_loss', 'tr_loss', 'val_acc', 'train_acc'])
        best_idx = loaded_best['val_acc'].argmax()
        # load best_best_loss
        best_best_loss = loaded_best['val_loss'][best_idx]
        # load best_best_acc
        best_best_acc = loaded_best['val_acc'][best_idx]

    def params_toString(params):
        paramlist = ['{}:{}'.format(k,params[k]) for k in sorted(params.keys())]
        cur_params = '$'.join(paramlist)
        return cur_params+'!'

    class multiomics_dataloader:
        def __init__(self, datasets, batch_size=None):
            self.datasets = datasets
            self.loaders = []
            self.normal = True
            for ds, mopt in datasets: 
                if batch_size is  None:
                    sz = len(ds)
                else:
                    sz= batch_size
                self.loaders.append((DataLoader(ds, batch_size=sz, shuffle=False), mopt))
        
        if opt.static_multiomics_loader:
            def __iter__(self):
                def _loader_iterator():
                    for loader, _mopt in self.loaders:
                        for batch in loader:
                            yield {'batch':batch ,'mopt': _mopt}
                return _loader_iterator()
        else: 
            def set_normal(self, normal):
                self.normal = normal

            def __iter__(self):
                self.iter_lst  = [iter(ldr) for ldr,_ in self.loaders] if self.normal else [iter(ldr) for ldr, m in self.loaders if len(m) >1 ] #LY
                self.mopt_lst  = [mopt for _, mopt in self.loaders] if self.normal else [mopt for _, mopt in self.loaders if len(mopt) >1 ] #LY
                self.restarted = set()
                self.current  = 0
                return self
                
            def __next__(self):
                k = self.current % len(self.iter_lst)
                self.current +=1
                while True:
                    try:
                        batch = next(self.iter_lst[k])
                        _mopt = self.mopt_lst[k]
                        return  {'batch':batch ,'mopt': _mopt}
                    except StopIteration:
                        del self.iter_lst[k]
                        del self.mopt_lst[k]
                        if len(self.iter_lst) == 0:
                            raise StopIteration
                        k = k % len(self.iter_lst)

        def shuffle_samples(self):
            for ds,_ in self.datasets:
                ds.shuffleSamples()
        
        def reset_random(self):
            for ds,_ in self.datasets:
                ds.reset_random()

        def get_size(self):
            return reduce(lambda x,y: x+y, [len(ds[0]) for ds in self.datasets]) if len(self.datasets) >1 else len(self.datasets[0][0])

    def loader_shuffle_samples(loader):
        if isinstance(loader, DataLoader):
            loader.dataset.shuffleSamples()
        else:
            loader.shuffle_samples()
    
    def loader_reset_random(loader):
        if isinstance(loader, DataLoader):
            loader.dataset.reset_random()
        else:
            loader.reset_random()

    def gene_model(flat_params, talos_seq, train_dataset, val_dataset):   
        nonlocal best_best_loss
        nonlocal best_best_acc
        nonlocal best_params
     
        if hasattr(opt,'fixed_seed'):
            fixed_seed = opt.fixed_seed if type(opt.fixed_seed) ==int else 1234 
            print(f'<{talos_seq}> set fixed_seed: {fixed_seed}')
            random.seed(fixed_seed)
            np.random.seed(fixed_seed)
            torch.manual_seed(fixed_seed)

        for m in mopts: 
            m.deg_ = 0
       
        talos_sno_lst = np.unique(np.array([m.talos_sno for m in mopts], dtype=np.int32))
        parlst = pack_params(flat_params, len(talos_sno_lst))
        for i,p in enumerate(parlst) : 
            print(f'<{talos_seq}> {mopts[i].tissue_name:32} @talos parameters: {params_toString(p)}')        
        add_params_to_opts(opt,mopts,parlst)
     
        models, super_models = create_models(opt, mopts, device)
        if opt.load_common:
            for m in models:
                _,LD_EXP_PATH = short_path(opt,m.mopt,nt_suffix='com_')
                LD_EXP_PATH = os.path.join(opt.project_name[:-3], LD_EXP_PATH)
                checkpoint = torch.load(os.path.join(LD_EXP_PATH, str(talos_seq), "model.pt"), map_location=device)
                m.load_state_dict(checkpoint['model_state_dict'])

        if hasattr(opt,'talos_start') and opt.talos_start > talos_seq:
            models[0].append_loss(0)
            models[0].append_metric(0)
            models[0].append_val_loss(0)
            models[0].append_val_metric(0)
            return models[0], models[0].parameters()

        train_loader = []
        val_loader   = []
        
        for t in train_dataset:
            if isinstance(t, DomainGraphDataset):
                train_loader.append(DataLoader(t,batch_size=opt.batchsize, shuffle=False))
            else:
                train_loader.append(multiomics_dataloader(t, batch_size=opt.batchsize))

        for v in val_dataset:
            if isinstance(v, DomainGraphDataset):
                val_bs = getattr(opt, 'val_batchsize', opt.batchsize)
                val_loader.append(DataLoader(v, batch_size=val_bs, shuffle=False))
            else:
                val_loader.append(multiomics_dataloader(v))
        
        best_msaved = []
        def fit_model():            
            start_epoch = 0
            end_epoch = opt.n_epochs 
            # Initialize history of net    
            best_loss = 1e10  # among epochs
            best_acc = -10 # among epochs
            best_raw_acc = -10
            #best_lossc = 1e10  # among epochs
            bestepoch = 0
            bestmodel = None
            
            nonlocal best_msaved
            if hasattr(opt,'fixed_seed'):
                for t in train_loader:
                    loader_reset_random(t)
            
            hist_metric_losswt = defaultdict(list)
            for epoch in range(start_epoch, end_epoch):
                for t in train_loader:
                    loader_shuffle_samples(t)
                    if len(mopts) > 1:
                        t.set_normal(epoch >= opt.stage2)
            
                train.epoch = epoch
                ####
                tr_res = train('train', super_models, train_loader, opt,mopts[0]) 
                sm, exp_ds = super_models[0], train_dataset[0]
                tr_res1 = run_validate(opt, sm, exp_ds) 
                tr_res.update(tr_res1)
                for _k,_v in tr_res1['metrics_per_mod'].items():
                    hist_metric_losswt[f'tr${_k}'].append(_v)

                sm, exp_ds = super_models[0], val_dataset[0]
                val_res = run_validate(opt, sm, exp_ds) 
                for _k,_v in val_res['metrics_per_mod'].items():
                    hist_metric_losswt[f'val${_k}'].append(_v)  
                      
                verbose = hasattr(opt,'train_verbose') and opt.train_verbose
                
                if not verbose: 
                    disp_every_cycle = epoch %25 == 0
                else:
                    disp_every_cycle = True
                if opt.target == 0:
                    tr_cm,_,_  = calc_confmat_from_res(tr_res,opt)
                    val_cm,_,_ = calc_confmat_from_res(val_res,opt)
                    if disp_every_cycle and verbose:
                        print('-----------------------')
                        print("training cm:")
                        print(tr_cm)
                        print("val cm:")
                        print(val_cm)
                if disp_every_cycle:
                    tr_racc  = tr_res.get('raw_acc', tr_res['acc'])
                    val_racc = val_res.get('raw_acc', val_res['acc'])
                    print(f'<{talos_seq}> epoch = {epoch:3d} tauc:{tr_res["acc"]:.3f} vauc:{val_res["acc"]:.3f} tacc:{tr_racc:.3f} vacc:{val_racc:.3f} tloss:{tr_res["lcall"]:.3f} vloss:{val_res["lcall"]:.3f}')
                    
                tr_loss  = tr_res['lcall']
                tr_acc   = tr_res['acc']  
                val_acc   = val_res['acc']  
                val_loss = val_res['lcall']
                models[0].append_loss(tr_loss)
                models[0].append_metric(tr_acc)
                models[0].append_val_loss(val_loss)
                models[0].append_val_metric(val_acc)
               
                if hasattr(opt,'load_params_selector') and opt.load_params_selector == 'val_loss':
                    cond =  val_loss  < best_loss and val_loss >= tr_loss
                else:  # 'val_corr'
                    cond =  val_acc > best_acc and val_acc <= tr_acc
                if  (cond  and epoch > opt.min_epoch ) or (bestmodel is None and epoch+1 >= end_epoch):
                    best_train_loss    = tr_loss
                    best_acc           = val_acc
                    best_raw_acc       = val_res.get('raw_acc', val_acc)
                    best_train_acc     = tr_acc
                    best_train_raw_acc = tr_res.get('raw_acc', tr_acc)
                    best_loss          = val_loss

                    bestmodel  = [{'weights':copy.deepcopy(s.state_dict()),
                                    'optim' :copy.deepcopy(s.optimizer.state_dict()),
                                    'epoch' :epoch}  for s in models]

                    bestepoch = epoch
                    if opt.target == 0:
                        bestmetric_dict = {'tr_cm':tr_cm, 'val_cm':val_cm, 'tr_acc':tr_acc, 'val_acc':val_acc}
                    else:
                        bestmetric_dict = {'tr_acc':tr_acc, 'val_acc':val_acc}
            return (best_loss, best_acc, best_raw_acc, best_train_loss, best_train_acc, best_train_raw_acc,
                    bestmetric_dict, bestepoch, bestmodel, start_epoch, end_epoch)            

        def save_models(
                        best_loss, best_acc, best_raw_acc, best_train_loss, best_train_acc, best_train_raw_acc,
                        bestmetric_dict, bestepoch, bestmodel, start_epoch, end_epoch):
                for i in range(len(models)):
                    best_model_wts = bestmodel[i]['weights']#.state_dict()
                    best_optimizer = bestmodel[i]['optim']#.optimizer.state_dict()
                    best_params    = params_toString(parlst[mopts[i].talos_sno]) #retrieve params

                    EXP_PATH, BEST_PATH,_ ,_, _ = make_path(1,opt,mopts[i],True)
                    temp_path = os.path.join(EXP_PATH,str(talos_seq))
                    lnk_path  = os.path.join(EXP_PATH, 'best_lnk')

                    saveall_train = make_saveall_info('train', temp_path,opt,mopts[i],BEST_PATH,best_params,start_epoch, end_epoch)

                    model_dict = {'seq':talos_seq,
                                'prev_best_loss':best_loss,
                                'prev_best_acc':best_acc, 'prev_best_raw_acc':best_raw_acc,
                                'prev_best_lossc':best_loss,
                                'best_model_wts':best_model_wts,'best_optimizer':best_optimizer,
                                'best_train_loss':best_train_loss,
                                'best_train_acc':best_train_acc, 'best_train_raw_acc':best_train_raw_acc,
                                'model':models[0],'best_epoch':bestepoch, 'bestmetric_dict': bestmetric_dict}
                    saveall_train(model_dict)
                return temp_path, lnk_path
        
        (best_loss, best_acc, best_raw_acc, best_train_loss, best_train_acc, best_train_raw_acc,
         bestmetric_dict, bestepoch, bestmodel, start_epoch, end_epoch) = fit_model()  

        if hasattr(opt,'load_params_selector') and opt.load_params_selector == 'val_loss':
            cond =  best_loss < best_best_loss 
        else:  # 'val_corr'
            cond =  best_acc > best_best_acc 
        if cond :
            best_best_loss = best_loss
            best_best_acc  = best_acc  #LY

        if opt.save_best:
            with FileLocker(mopts[0].dataroot_):
                temp_path, lnk_path = save_models(
                            best_loss, best_acc, best_raw_acc, best_train_loss, best_train_acc, best_train_raw_acc,
                            bestmetric_dict, bestepoch, bestmodel, start_epoch, end_epoch)
                x_avg = dict()
                for ds in train_dataset:
                    for t, m in ds:  
                        for s in t:  
                            for u,v in zip(m,s):
                                if v.y == 0:
                                    cnt0, x0 = x_avg.setdefault(u.tissue_name, (0,torch.zeros_like(v.x)))
                                    x_avg[u.tissue_name] =  (cnt0 +1, v.x + x0)
                for k in x_avg:
                    x_avg[k] = x_avg[k][1] / x_avg[k][0]
                torch.save(x_avg, os.path.join(temp_path, 'ctrl-avg-x.pt'))
              
                if cond:
                    if os.path.isdir(temp_path):
                        if os.path.islink(lnk_path):
                            os.unlink(lnk_path)
                        os.symlink(os.path.relpath(temp_path, os.path.dirname(lnk_path)),lnk_path)  
        del bestmodel
        return models[0], models[0].parameters()
       
    def prep_dummy_history_and_model_parameters():
        model = DummyModule()
        model.append_loss(0)
        model.append_metric(0)
        model.append_val_loss(0)
        model.append_val_metric(0)
        return model, model.parameters()

    if predicate_xopt(opt, 'multiprocess'):
        def worker_func(id):
            print(f' [worker {id}] {os.getpid()} ready')
            torch.set_num_threads(1)
            arg = Q1.get()
            if type(arg) == str and arg == Sentinel:
                print(f' [worker {id}]', os.getpid(),'rcvd from Q1:', arg)
                return
            print(f' [worker {id}] {os.getpid()} fetch datasets')
            train_dataset, val_dataset, *_ = fetch_datasets(opt, mopts)
            try:
                while True:
                    print(f'[worker {id}]', os.getpid(), 'rcvd from Q1:', arg)
                    if type(arg) == str and arg == Sentinel:
                        break
                    gene_model(flat_params=arg[0],talos_seq= arg[1], train_dataset=train_dataset, val_dataset=val_dataset)
                    Q2.put('Done')
                    arg = Q1.get()
            except Exception as e:
                traceback.print_exc()
                Q2.put('Error')
        
        nproc = opt.multiprocess if opt.multiprocess < 10000 else torch.get_num_threads()
        torch.set_num_threads(1)
        processes = [Process(target=worker_func, args=(i,)) for i in range(nproc)]
        
        for p in processes:
            Q2.put('Done')
            p.start()
    else:
        processes = None
        torch.set_num_threads(1)
        print(f' [main process] {os.getpid()} fetch datasets')
        train_dataset, val_dataset, *_ = fetch_datasets(opt,mopts)
    if hasattr(opt,'common_only') and opt.common_only:
        for m in mopts:
            setattr(m,'samesplit',True)
    def gene_model_mp(x_train, y_train, x_val, y_val, flat_params):
        nonlocal talos_seq
        if predicate_xopt(opt, 'multiprocess'):
            print(' main proc',os.getpid(),'in talos put flat_params', talos_seq, flat_params)
            if Q2.get() == 'Error':
                raise ErrorMp()
            Q1.put((flat_params,talos_seq))
            talos_seq +=1
            return prep_dummy_history_and_model_parameters()
        else:
            retv = gene_model(flat_params,talos_seq, train_dataset=train_dataset, val_dataset=val_dataset)
            talos_seq +=1
            return retv
     
    return gene_model_mp, processes

def final_task_after_talos_scan(opt, mopts, processes, Q1, Q2, worker_error):
    if processes is not None:
        import pandas as pd
        print(os.getpid(),'all Done')
        for _ in processes:
            Q1.put(Sentinel) 
        for p in processes:
            p.join()
        while not Q1.empty(): Q1.get()
        while not Q2.empty(): Q2.get()
        if not worker_error:
            for m in mopts:
                EXP_PATH, BEST_PATH,_ ,_, _ = make_path(1,opt,m)
                df = pd.read_csv(BEST_PATH,sep='\t' )
                seq = df['seq '][df[' Best acc '].argmax()]
                df = df.sort_values(by='seq ')
                df.to_csv(BEST_PATH,sep='\t',index=False, float_format='%0.3f')
                best_link = os.path.join(EXP_PATH,'best_lnk')
                if os.path.islink(best_link):
                        os.unlink(best_link)
                temp_path = os.path.join(EXP_PATH,str(seq))
                os.symlink(os.path.relpath(temp_path, os.path.dirname(best_link)),best_link)   
         

if __name__ == '__main__':
    import traceback
    opt,mopts,params,cparams,opt_cmd = paramparser(choice='gnn')
    
    lock_path = os.getcwd()
    if opt.target_permute and opt.permute_no >0:
        cparams = None
        params = []
        saved_proj = opt.project_name
        _tmp = opt.project_name.split('/')
        _tmp = [str(temp) for temp in _tmp]
        _tmp[-2] = str(opt.ref_runseq) 
        opt.project_name = '/'.join(_tmp)
        
        for mopt in mopts:
            from platformX.load_model import load_best_params
            _,_,_, LOAD_EXP_PATH, LOAD_BEST_PATH = make_path(3,opt,mopt)
            p, *_ = load_best_params(opt,LOAD_BEST_PATH)
            for k,v in p.items():
                p[k] = [v]
            params.append(p)
        opt.project_name = saved_proj
        
    toplevel_dir = opt.project_name[:-3]
    print('output dir:', toplevel_dir)
    
    if not hasattr(opt,'talos_start'):
        with FileLocker(lock_path):
            if not os.path.exists(f'{toplevel_dir}/ck'):
                os.makedirs(f'{toplevel_dir}/ck')
            for m in mopts:
                toplevel_dir,_,_ ,_, _ = make_path(1,opt,m)
                if os.path.exists(toplevel_dir):
                    if os.path.isdir(toplevel_dir):
                        shutil.rmtree(toplevel_dir)
                    else:
                        os.remove(toplevel_dir)
                os.makedirs(toplevel_dir)

    EPS = 1e-15
    if not hasattr(opt,'threesplit'):
        opt.threesplit = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'total number of available gpus: {torch.cuda.device_count()}')
    print(f'num of threads :{torch.get_num_threads()}')
    print(f'fold = ', opt.fold)
    assert opt.fold == mopts[0].fold

    train_dataset, val_dataset = [], []
 
    if predicate_xopt(opt,'multiprocess'):
        from multiprocess import Process, Queue, set_start_method
        if torch.cuda.is_available():
            set_start_method('spawn', force=True)
        Q1, Q2= Queue(), Queue()
        Sentinel = 'Sentinel'
    else:
        Sentinel, Q1, Q2 = None, None, None

    mk_model, processes = make_model(opt, mopts, device, Sentinel, Q1, Q2)
    worker_error = False
    try:
        scan_object = mySimpleTalos.Scan(x = train_dataset, y = train_dataset,
                                    x_val = val_dataset, y_val = val_dataset,
                                    params = flatten_params(params, cparams, []),
                                    model = mk_model,
                                    experiment_name = opt.project_name,
                            )
    except Exception as inst:
        print('seeing errors', inst)
        traceback.print_exc()
        worker_error = True
    finally:
        final_task_after_talos_scan(opt, mopts, processes, Q1, Q2, worker_error)


    if hasattr(opt, 'cfg_file'):
        subproj_dir = os.path.dirname(opt.cfg_file)
        with FileLocker(lock_path):
            if os.path.exists(opt.project_name):
                shutil.copy(opt.cfg_file, os.path.join(opt.project_name,
                    f'cfg_{os.path.basename(subproj_dir)}.txt'))
                if not os.path.exists(os.path.join(opt.project_name,'symlink_cfg_dir')):
                    os.symlink(os.path.relpath(opt.cfg_file, opt.project_name), 
                            os.path.join(opt.project_name,'symlink_cfg_dir'))

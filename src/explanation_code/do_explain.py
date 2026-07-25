from .integrated_gradient import Explainer
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch
import torch_geometric
if torch_geometric.__version__ < '2.0':
    from torch_geometric.data import DataLoader
else:
    from torch_geometric.loader import DataLoader


from functools import reduce
from sklearn.metrics import confusion_matrix
def tocpu(lst):
    return [x.cpu().detach() for x in lst]

class DataSetWrapper:
    def __init__(self, dataset, mopt):
        self.dataset = dataset
        self.mopt    = mopt
        # if isinstance(self.dataset, (tuple,list)):
        #     self.ds_sz = [len(ds) for ds,_ in  dataset]
        #     self.data_sz = np.cumsum(np.array([ds for ds,_ in dataset], dtype=int))
        #     self.mopt = [mopt for _,mopt in dataset]

    def __getitem__(self, idx):
        # if isinstance(self.dataset, (tuple,list)):
        #     idx0 = np.sum(self.data_sz <= idx) - 1
        #     idx1 = idx - self.data_sz[idx0]
        #     d = self.dataset[idx0][idx1]
        #     return {'data':d[:-1],  'mopt': self.mopt[idx0]}
        d = self.dataset[idx]
        return {'data':d[:-1],  'mopt': self.mopt}
        # data is a list of (Data), mopt is a list of (InputClass) 
            
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.dataset):
            raise StopIteration
        data = self.dataset[self.idx]
        self.idx += 1
        return data

def do_explain(opt, model, dataset_mopt, class_idx_ls, exp_mode = 'ig', check_only=False, verbose=False):
    # print(f'explain mode: {exp_mode}')

    class_idx_ls = model.label_mapping(class_idx_ls)
    dataset, mopt =  dataset_mopt
    dataset = DataSetWrapper(dataset= dataset, mopt= mopt)
    
    res        = [defaultdict(list) for _ in mopt]
    samids     = defaultdict(list)
    explainer  = Explainer(opt=opt, model= model)
    extra_forward_args= ['pos','cov']

    def check_data(idx=None):
        if idx is not None: #one sample per iterative call 
            data, mopt_lst = dataset[idx].values()
            data = [next(iter(DataLoader([d], batch_size=1))) for d in data] 
        else: # all samples in one batch call
            data = next(iter(DataLoader(dataset_mopt[0],batch_size=len(dataset_mopt[0]))))[:-1]
            mopt_lst = mopt
        data1 = data[0]
        output,loss = explainer.predict(data_lst=data,
                        mopt_lst=mopt_lst,
                        extra_forward_args=extra_forward_args)
        return loss, (data1.y.squeeze(1) == output.argmax(axis=1)).detach().cpu().numpy(),  output.detach().cpu().numpy(), data1.y.detach().cpu().numpy()

    def call_ig(idx):
        torch.set_num_threads(1)
        data = dataset[idx]
        data1 = data['data'][0]
        clz = data1.y.item()
        try: 
            # explainer = explainers[clz]
            exp_res = explainer.integrated_gradients(
                        clz = clz,
                        data=data['data'],
                        mopt=data['mopt'],
                        n_steps =200,
                        extra_forward_args=extra_forward_args)
        except ValueError:
            pass
        return exp_res
    
    nthread = torch.get_num_threads()
    one_sample_per_check = False
    if not opt.target_permute or check_only:
        if one_sample_per_check:
            chk_lst = list(map(check_data, range(len(dataset))))
            loss_lst, correct_lst, pred_lst, y_lst= list(zip(*chk_lst))
            loss = sum(np.array(loss_lst))
            correct_lst = np.vstack(correct_lst).squeeze(1) #convert to row vector
            preds = np.vstack(pred_lst)
            yall  = np.vstack(y_lst)
        else:
            loss, correct_lst, preds, yall =check_data()
        data_idx = np.arange(len(dataset))[correct_lst].tolist() #sample index of correctly predicted samples
        cm = confusion_matrix(yall, preds.argmax(axis=1))
        if verbose: 
            print(f'explain: num of correct prediction {len(data_idx)}/{len(correct_lst)}  acc {len(data_idx)/len(correct_lst)}')
            print(cm)
        corrects = np.array([len(data_idx),len(correct_lst)]) #  (correct pred counts and sample counts in dataset )
        if len(mopt) == 1:
            mopt[0].model_.saved_info = corrects
        else:
            model.saved_info = corrects
    else:
        data_idx = list(range(len(dataset)))
        corrects = None
        loss = None
    if check_only :
        return None, None, corrects, loss, preds, yall
        
    # print(data_idx)
    if hasattr(opt,'multiprocess') and opt.multiprocess:
        from multiprocess import Pool
        nproc = nthread if opt.multiprocess >=10000 else opt.multiprocess
        print('nproc:', nproc)
        chunk_sz = (len(data_idx)+nproc-1)//nproc
        data_idx_chunk = [data_idx[i*chunk_sz:(i+1)*chunk_sz] for i in range(nproc-1)]
        data_idx_chunk.append(data_idx[(nproc-1)*chunk_sz:])
        with Pool(nproc) as p:
            exp_res_ls0 = list(p.map(lambda x:list(map(call_ig,tqdm(x))), data_idx_chunk))
        exp_res_ls = reduce(lambda x,y: x+y, exp_res_ls0)
    else:
        exp_res_ls = list(map(call_ig, tqdm(data_idx)))
    
    for idx in range(len(dataset)):
        data  = dataset[idx]
        data1 = data['data'][0]
        clz = class_idx_ls.index(data1.y)

        if idx in data_idx:
            n = data_idx.index(idx)
            exp_res = exp_res_ls[n]
        else:
            exp_res = [np.array([np.NAN]*d.x.shape[0]) for d in data['data']]

        for j,eres in enumerate(exp_res):
            res[j][clz].append(eres)
        samids[clz].append(data1.samid)

    torch.set_num_threads(nthread)
    return res, samids, None, None, None, None
    
    

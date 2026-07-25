import torch
from torch import cuda 
from .SingleModel import (do_single_batch,
                          create_single_models)
from .load_model import load_model
from utils.cross_loss import align_corr, align_dist
from utils.utils import calc_nll_loss_wt, predicate_xopt
from utils.common_loss import calc_reg
from set_transformer.set_transformer import Encoder
from queue import Queue
from threading import Thread, RLock
from functools import partial
from model.gnn_networks import GphClassif


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)

def align_loss_multiomics(input:list, opt,device):
    def _contrastive_loss_for_pair(input:list, tau):
        
        assert len(input) ==2
        omics0 = input[0] / (torch.linalg.norm(input[0], dim=1, keepdim=True) + 1e-6)
        omics1 = input[1] / (torch.linalg.norm(input[1], dim=1, keepdim=True) + 1e-6)
        # pairwise dot product of omics0/1
        pairwise_dot_exp = torch.exp(torch.sum(omics0.unsqueeze(1) * omics1.unsqueeze(0),dim=2)/tau)  # dim2 is rep dim?
        self_dot_exp0    = torch.exp(torch.sum(omics0.unsqueeze(1) * omics0.unsqueeze(0),dim=2)/tau)
        self_dot_exp1    = torch.exp(torch.sum(omics1.unsqueeze(1) * omics1.unsqueeze(0),dim=2)/tau)
        
        #self_dot_exp0.fill_diagonal_(0)
        #self_dot_exp1.fill_diagonal_(0)
        diag0 = torch.diagonal(self_dot_exp0)
        diag1 = torch.diagonal(self_dot_exp1)
        diagX = torch.diagonal(pairwise_dot_exp)
        loss0 = torch.log(torch.sum(pairwise_dot_exp,dim=1) + torch.sum(self_dot_exp0,dim=1)- diag0 -diagX )\
                + torch.log(torch.sum(pairwise_dot_exp,dim=0) + torch.sum(self_dot_exp1,dim=0) - diag1 -diagX )\
                - 2 * torch.log(diagX)
        return torch.mean(loss0)

    def _rec_choose(nset,k):
        if k==0: 
            yield []
        elif len(nset) == k:
            yield nset
        elif len(nset) < k:
            pass
        else:
            for i in range(len(nset)):
                for y in _rec_choose(nset[i+1:], k-1):
                    yield [nset[i]] + y

    def _choose(n, k):
        num_set = list(range(n))
        for elem in _rec_choose(num_set,k):
            yield elem  
    
    align_loss = torch.tensor(0.0, requires_grad=True,device = device)
    if opt.align_pair_multiomics is not None:
        pair_range = opt.align_pair_multiomics
        pair_coeff = opt.align_coeff_multiomics
    else:
        pair_range = _choose(len(input),2)
        pair_coeff = torch.ones(len(pair_range), dtype=torch.float64, device=device)
    for (i,j), coeff in zip(pair_range, pair_coeff):
        coeff = float(coeff)
        if opt.fc1_align_multiomics == 'contrastive':
            tau = opt.tau_multiomics
            align_loss = align_loss + _contrastive_loss_for_pair((input[i],input[j]),tau) * coeff
        elif opt.fc1_align_multiomics == 'align_corr':
            align_loss = align_loss + align_corr((input[i],input[j]),abs=predicate_xopt(opt,'absolute_corr')) * coeff
        elif opt.fc1_align_multiomics == 'align_dist':
            align_loss = align_loss + align_dist((input[i],input[j])) * coeff
    return align_loss

class IntBaseModel(torch.nn.Module):
    def __init__(self, models, mopts):
        super(IntBaseModel,self).__init__()
        self.models = torch.nn.ModuleList(models)
        self.mopts = mopts
        self.mopt  = mopts[0]
    def train(self):
        for m in self.models:
            m.train()

    def eval(self):
        for m in self.models:
            m.eval()

    def update_weights(self, first_step=True):
        for m in self.models:
            m.update_weights(first_step)
    
    def is_optimizer_SAM(self):
        return hasattr(self.models[0].optimizer,'second_step')

    def zero_grad(self):
        for m in self.models:
            m.zero_grad()

    def step_scheduler(self):
        for m in self.models:
            m.step_scheduler()
    def step_scheduler2(self):
        for m in self.models:
            m.step_scheduler2()
    def step_scheduler3(self):
        for m in self.models:
            m.step_scheduler3()
    
    def label_mapping(self, class_idx_ls):
        return self.models[0].label_mapping(class_idx_ls)
    
class IntegratedModel3(IntBaseModel): #multi-BD, transformer
    def __init__(self, opt, mopts, *model):
        super(IntegratedModel3,self).__init__(model, mopts)
        self.opt = opt
        self.L1_reg = self.mopt.l1_reg if hasattr(self.mopt,'l1_reg') else 0.0
        self.nobalance = hasattr(opt,'no_balance') and opt.no_balance
        self.nclass = opt.nclass if not hasattr(self.mopt,'nclass') else self.mopt.nclass

        self.imported_modules  = mopts[-1].shared_modules # from last individual model
        predicate_create_shared = lambda name: (self.imported_modules is None 
                                                or name not in self.imported_modules)

        if hasattr(self.opt,'n_transformer_layer'):
            num_layers = self.opt.n_transformer_layer  
        else:
            num_layers = 2
        
        num_features = model[0].embedding.get_outdim() #fc1_out_dim
        if hasattr(self.opt, 'heads_transformer'):
            num_heads = self.opt.heads_transformer
        else:
            num_heads = 1
        if hasattr(self.opt, 'dim_ff'):
            dim_ff = self.opt.dim_ff
        else:
            dim_ff = 16
        norm_first =  False
        self.encoder_kwargs = {'num_layers':num_layers, 
                                'num_features':num_features,
                                'num_heads':num_heads, 
                                'dim_ff':dim_ff, 
                                'norm_first':norm_first}
        self.encoder = Encoder(**self.encoder_kwargs)
       
        if predicate_create_shared('classif'):
            self.classifier         = GphClassif(num_features, mopts[0])
            self.models[0].set_transformer = torch.nn.ModuleList([self.encoder, self.classifier])
            
        else:
            self.classifier = self.imported_modules['classif']
            self.models[0].set_transformer = self.encoder
            

    def forward(self, input:list):
        
        x = torch.stack(input, dim =1) # batch, seq, features
        x = self.encoder(x, False)
        embed_out = x[:,0,:]
        out = self.classifier(embed_out)
        extra_output = self.classifier.take_train_output()
        return out, embed_out, extra_output # output, emb, attn_wts
        
    def loss_fn(self, pred, target, extra_output):
        assert len(pred.shape) == len(target.shape)
        lmbd8 = self.mopt.lamb8
        target = target.squeeze(1)
        #return (nn.NLLLoss()(F.log_softmax(pred, dim=-1), target), )
        if self.nobalance:
             # wt only works for binary classes currently
            wt=calc_nll_loss_wt(target,self.nclass,thre=4,alpha=0.5)    
        else:
            wt = torch.tensor([1 for clz in range(self.nclass)], dtype=torch.float, device = target.device) 
        #return (torch.nn.functional.nll_loss(F.log_softmax(pred, dim=-1), target,weight=wt), )
        # forward() from setTransformerClassif has applied F.log_softmax
        bn_gamma_reg =  torch.tensor(0) if lmbd8 ==0 or extra_output['gamma'] is None   else lmbd8 * calc_reg(extra_output['gamma']) 
        if predicate_xopt(self.opt,'train_verbose'):
           print(f'in IG,loss_gamma={bn_gamma_reg}')
        return (torch.nn.functional.nll_loss(pred, target, weight=wt) * self.opt.lamb00, bn_gamma_reg)

def do_single_batch_multiomics(model, opt, device, epoch, data_batch, mopts =None, run='train',lossr_only=False):
        if mopts is None:
            mopts = model.mopts
        res_ls = []
        emb_ls = []
        extra_output = []
        loss_all = torch.tensor(0.0, device=device)
        lc_all = torch.tensor(0.0, device=device)
        sz_dls = mopts[0].nclass + (2 if opt.support_attn == -3 else 1)

        if cuda.device_count() <=1:
            multi_gpu = False
        else: #for multi-gpu
            multi_gpu = True    
            queue = Queue()
            sentinel = object()
            lock = RLock()
            threads = []
            def worker(id):
                mygpu = torch.device("cuda:{}".format(id))
                while True:
                    p_working = queue.get()
                    #p_working is partial function of working or an empty object
                    if p_working is sentinel:
                        break #terminate
                    p_working(device=mygpu)
                    torch.cuda.synchronize(mygpu) #wait for current gpu to complete the work.
            #start worker threads
            for gpuid in range(cuda.device_count()):
                thread = Thread(target=worker, args=(gpuid,))
                threads.append(thread)
                thread.start()   

        def working(submodel, data, device, in_thread=False,lossr_only=False):
            if in_thread: 
                submodel.to(device)
                if run != 'test':
                    optimizer_to(submodel.optimizer,device)
                    scheduler_to(submodel.scheduler,device)
            
            fc1_detached = not predicate_xopt(mopts[0],'samesplit')
            output, extra, loss, lc = do_single_batch(submodel, data, device, opt, run=run, 
                                        fc1_detached=fc1_detached,lossr_only=lossr_only) # loss from ind classif, loss not inc loss_mp
           
            def collect_results():
                nonlocal loss_all
                nonlocal lc_all
                res_ls.append(output)
                emb_ls.append(extra['emb_out'])
                extra_output.append(extra)
                
                if run != 'test':
                    loss_all = loss_all + loss.to(loss_all.device)
                    
            if in_thread:
                with lock:
                    collect_results()
            else:
                collect_results() 

        nofOmics = len(mopts)
        for n, mopt in enumerate(mopts):
            submodel = mopt.model_
            if mopt.lamb5 !=0 and hasattr(mopt,'WW_all'):
                WW_all = mopt.WW_all
            else:
                WW_all = None  
            
            data = (WW_all, data_batch[n], data_batch[nofOmics]) 
            
            if mopt.support_attn == -1: 
                dls_start = nofOmics + n* sz_dls + 1
                data = (*data,  *data_batch[dls_start: dls_start+ sz_dls] )
            
            if not multi_gpu: 
                working(submodel, data, device,lossr_only=lossr_only) 
            else:
                queue.put(partial(working, submodel=submodel,data=data,in_thread=True))

        if multi_gpu: 
            for _ in range(cuda.device_count()):
                queue.put(sentinel)
            for thread in threads:
                thread.join()
            
            device = next(model.models[0].parameters()).device
            emb_ls = [emb.to(device) for emb in emb_ls]
            res_ls = [res.to(device) for res in res_ls]
        
        multiomics_fusion   = predicate_xopt(opt,'multimodal_data')
        fusion_on_embedding = True
        if  multiomics_fusion:
            output, embedded_dout, extra_output2 = model(emb_ls) if fusion_on_embedding else  model(res_ls)
            
        if run != 'test':  
            max_lamb0 = min_lamb0 = opt.lamb00
            target = data[1].y.to(device)
            t_ret = model.loss_fn(output, target, extra_output2)  
                    
            if hasattr(opt, 'heat') and epoch >= opt.heat: # late stage    
                fac = max_lamb0 
            else:
                fac = max_lamb0 
            if lossr_only:
                loss_all = loss_all + fac  * t_ret[1].to(loss_all.device)
            else:
                loss_all = loss_all + fac  * sum(t_ret).to(loss_all.device)  
            lc_all   = lc_all + fac * t_ret[0].to(lc_all.device) 

            align_multiomics = hasattr(opt, 'align_multiomics') and opt.align_multiomics
            if nofOmics > 1 and align_multiomics:
                loss_align_all = align_loss_multiomics(emb_ls,opt,device)
                loss_all += loss_align_all.to(loss_all.device)
            
        return output, {'emb_out':embedded_dout}, loss_all, lc_all, extra_output2, data[1], data[2], extra_output 
        

def create_multiomics_model(opt,mopts,device):
    model = create_single_models(opt,mopts)
    multiomics_fusion = predicate_xopt(opt,'multimodal_data')
    
    if len(mopts) >1 and multiomics_fusion:
        super_model = IntegratedModel3(opt, mopts, *model)
        super_model.mopts =  mopts
    else:
        super_model = model[0]
    for m in model:
        m.to(device)
    return super_model, model

def load_saved_models(opt, mopts, device, multiomics=False):
    model = []
    super_model = None
    shared_modules = None
    explain_ref_x = None
    for m in mopts:
        m.shared_modules = shared_modules
        model.append(load_model(opt,m, device, load_checkpoint=False)[0])
        shared_modules = model[-1].get_shared_modules()
        if hasattr(model[-1], 'explain_ref_x'):
            explain_ref_x = model[-1].explain_ref_x
    
    if multiomics:
        multiomics_fusion = predicate_xopt(opt,'multimodal_data')
        
        if len(mopts) > 1 and multiomics_fusion:
            super_model = IntegratedModel3(opt, mopts, *model)
        else: 
            super_model = model[0]
        if explain_ref_x is not None:
            super_model.explain_ref_x = explain_ref_x
            
    for m in model:
        m.to(device)
        m.load_state_dict(m.checkpoint['model_state_dict'])
        m.checkpoint = None
    return super_model, model 

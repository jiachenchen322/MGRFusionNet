import torch
import numpy as np
from utils.common_loss import zloss,make_loss_fn
from utils.utils import predicate_xopt
from torch.optim import lr_scheduler
from model.gnn_networks import NNGAT_Net
from .load_model import load_model
import torch_geometric as pyg
from .sam import SAM

def do_single_batch(model, data_batch, device, opt, run='train', fc1_detached= False, lossr_only=False):
    mopt = model.mopt
    if hasattr(opt,'qr') and opt.qr:
        model.embedding.qr_commu_attn_fc1()
    WW_all, data0, idx_data, *datals  = data_batch
    data1 = data0.to(device)
    x =  data1.x 
    if model.training and predicate_xopt(opt,'adding_noise'):
        x0 = x.view(data1.batch[-1]+1, -1, x.shape[-1])
        if not hasattr(model, 'input_var'):
            model.input_df  = x0.shape[0] -1
            model.input_var = torch.var(x0,0,keepdim=True) * model.input_df
        else:
            model.input_df  += x0.shape[0] -1
            model.input_var += torch.var(x0,0,keepdim=True) * (x0.shape[0]- 1)
            
        std = torch.sqrt(model.input_var/model.input_df).expand(x0.shape[0],-1,-1)
        v = std.reshape(-1, x.shape[-1]) /2
        noise = torch.randn_like(x, device=x.device) * v
        x += noise 
    loss = 0
    lc = 0
   
    if len(datals) >0:
        assert mopt.support_attn == -1
        edge_index = [d.edge_index.to(device) for d in datals[:mopt.nclass]]
        edge_attr  = [d.edge_attr.to(device)  for d in datals[:mopt.nclass]]
    else:
        edge_index = data1.edge_index
        edge_attr  = data1.edge_attr
    batch = data1.batch
    if pyg.__version__ < '2.4.0':
        pos = data1.pos if 'pos' in data1.keys else None
        cov = data1.cov if 'cov' in data1.keys else None
    else:
        pos = data1.pos if 'pos' in data1.keys() else None
        cov = data1.cov if 'cov' in data1.keys() else None 
    
    output, extra_output = model(x, edge_index, batch, edge_attr, cov, pos, detached=fc1_detached) 
    mempool_epoch_update = hasattr(opt,'memp') and opt.memp >0 and predicate_xopt(opt,'mempool_epoch_update')
    if run != 'test':
        lossc   = model.loss_fn[0](opt,mopt, output,data1)
        lossr,tret = model.loss_fn[1](opt,mopt, extra_output,data1,idx_data,WW_all)
        loss_mp = tret[0]     
        extra_output['loss_mp'] = loss_mp
        
        if mempool_epoch_update:
            loss,lc = (lossr, lossc) if lossr_only else (lossc+lossr, lossc)
        else:
            loss,lc = (lossr+loss_mp, lossc) if lossr_only else (lossc+lossr+loss_mp,lossc) 
        if hasattr(mopt,'ztarget') and not hasattr(opt,'ztarget_all'): 
            alpha_z = 1.0 if not hasattr(mopt,'alpha_z') else mopt.alpha_z
            zloss1 = zloss(model, mopt, data1, extra_output) * alpha_z 
            loss = loss + zloss1
            lc = lc + zloss1
        
        if predicate_xopt(opt,'train_verbose'):
           print(f'loss_mp={loss_mp},lossc={lc},loss_gamma={tret[1]},loss_decorr={tret[2]},loss(exc loss_mp)={loss}')   
    return output,extra_output,loss,lc

def get_optimizer(model, opt, mopt):
    no_decay = list()
    decay = list()
    decay_name= list()
    for name, param in model.named_parameters():
        if 'gnn_plus_mpool' in name:
            pass 
        elif 'gbn_layer' in name or 'clf_bns' in name or 'fc1_bn' in name or 'group_bn' in name or 'weight' not in name:
            no_decay.append(param)
        else:
            decay.append(param)
            decay_name.append(name)

    if hasattr(opt,'optim_sam') and opt.optim_sam:
        print('-- use optim_sam')
        base_optimizer = torch.optim.AdamW
           
        optimizer = SAM([{'params': no_decay, 'weight_decay':0},
                {'params': decay, 'weight_decay':mopt.weightdecay }], base_optimizer= base_optimizer,lr= mopt.lr)
    else:
        optimizer = torch.optim.AdamW([{'params': no_decay, 'weight_decay':0},
            {'params': decay, 'weight_decay':mopt.weightdecay }], lr= mopt.lr)

    model.optimizer = optimizer
    model.decay = decay
    model.decay_name = decay_name

    if hasattr(opt,'lr_scheduler') and opt.lr_scheduler == 'cosine':
        print('-- use cosinelr')
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=opt.n_epochs, eta_min = mopt.lr/100)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=mopt.stepsize, gamma=mopt.gamma)
    model.scheduler = scheduler
    
    if True: 
        no_decay, decay = [],[]
        for name, param in model.embedding.named_parameters():
            if 'weight' in name:
                decay.append(param)
            else:
                no_decay.append(param)

        if hasattr(opt,'beta1'):
            model.optimizer2 = optimizer2 = torch.optim.AdamW([{'params': decay, 'weight_decay':mopt.weightdecay},{'params': no_decay, 'weight_decay':0}],
             lr=2*mopt.lr, betas=(opt.beta1,0.999)) #  betas=(0.9, 0.999)
        else:
            model.optimizer2 = optimizer2 = torch.optim.AdamW([{'params': decay, 'weight_decay':mopt.weightdecay},{'params': no_decay, 'weight_decay':0}], 
             lr=mopt.lr)
        if hasattr(opt,'lr_scheduler') and opt.lr_scheduler == 'cosine':
            print('-- use cosinelr')
            scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer2,T_max=opt.n_epochs, eta_min = mopt.lr/100)
        else: #lr_scheduler == 'steplr'
            scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=mopt.stepsize, gamma=mopt.gamma)
            #scheduler2 = lr_scheduler.CyclicLR(optimizer2, base_lr=mopt.lr, max_lr=20*mopt.lr, step_size_up=20, mode='triangular',cycle_momentum=False) # step_size_down=1, 
        model.scheduler2 = scheduler2

    loss_fn = make_loss_fn(model.lamb0, mopt.lamb1, mopt.lamb2, mopt.lamb3, mopt.lamb4,
            mopt.lamb5, mopt.lamb6, mopt.lamb7,mopt.lamb8,mopt.lamb9,mopt.ratio,mopt.ratio) # mopt.lamb0 -> model.lamb0
    model.loss_fn = loss_fn
    return optimizer, scheduler, loss_fn

def create_single_models(opt,mopts,device=None):
    no_models = len(mopts)
    model = []
    shared_modules = None
    for i  in range(no_models):
        mopts[i].shared_modules = shared_modules
        print(f'model_{i}: indim({mopts[i].indim}) gc({mopts[i].gc}),fc({mopts[i].fc}),memp({mopts[i].memp}),targets({mopts[i].target_list}),ratio({mopts[i].ratio:.3f})')
        if hasattr(opt, 'fine_tuning'):
            if opt.fine_tuning == 1:
                pretrain_opt,pretrain_mopts = opt, mopts
                if hasattr(opt, 'pretrain_target'):
                    m = load_model(pretrain_opt, pretrain_mopts[i], device)
                else:
                    raise ValueError('Transfer learning indicated, but no pretrain target indicated, include "pretrain_target = {target} in config file"')
                if hasattr(opt, 'reset_classifier') and opt.reset_classifier == 1:
                    for layer in m.classif.children():
                        if len(tuple(layer.children())) > 0:
                            for child_layer in layer.children():
                                child_layer.reset_parameters()
                        else:
                            layer.reset_parameters() 
                if hasattr(opt, 'freeze_embedding') and opt.freeze_embedding == 1:
                    for param in m.embedding.parameters():
                        param.requires_grad = False
            else:
                m = NNGAT_Net(opt, mopts[i])
        else:
            m = NNGAT_Net(opt, mopts[i])

        if hasattr(opt,'shared_modules') and i == 0:
            shared_modules = m.get_shared_modules()

        get_optimizer(m,opt,mopts[i])
        m.init_history()
        mopts[i].model_ = m
        model.append(m)
        if device is not None: # device  is also a flag 
            m.to(device)
    return model

class IntResults: # for a species
    def __init__(self,opt):
        multiomics_fusion = predicate_xopt(opt,'multimodal_data')
        if not multiomics_fusion:
            self.n_transformer_layer = 0   
        else:
            if hasattr(opt,'n_transformer_layer'):
                num_layers = opt.n_transformer_layer  
            else:
                num_layers = 2
            self.n_transformer_layer = num_layers
        
        self.output_list,self.y_all_list,self.samid_list =[],[],[]# list for an epoch
        
        self.emb_list = []  
        self.emb_out_list = [] 
        self.fc2_list = [] 
        self.last_intgc_list   = []
        self.last_intpool_list = []
        self.intmempool_list = []
        self.intgc_list = None
        self.intpool_list = None

    def collect_results(self, data0, output, emb=None ):
        self.y_all_list.append(data0.y.cpu()) 
        self.samid_list.extend(data0.samid)
        self.output_list.append(output.detach().cpu())
        self.emb_list.append(emb.detach().cpu())

    def get_stacked_results(self):
        t_emb_list = torch.vstack(self.emb_list).numpy()
        return torch.cat(self.y_all_list).numpy(), torch.vstack(self.output_list).numpy(), self.samid_list, t_emb_list

    def get_stacked_output(self):
        return torch.vstack(self.output_list).numpy()
    
    def get_stacked_y_all(self):
        return torch.cat(self.y_all_list).numpy()
    
    def get_samid_list(self):
        return self.samid_list
    
    def get_stacked_emb(self):
        return torch.vstack(self.emb_list).numpy()
    
    def collect_int_results(self, extra_output, mopt):
        self.last_intpool_list.append(extra_output[f'intpool{mopt.gc}'].detach().cpu())
        self.last_intgc_list.append(extra_output[f'intgc{mopt.gc}'].detach().cpu())
        self.intmempool_list.append(extra_output['intmempool'].detach().cpu()) # to do: mopt.memp may >1
        self.emb_out_list.append(extra_output['emb_out'].detach().cpu())
        if 'fc2' in extra_output.keys():
            self.fc2_list.append(extra_output['fc2'].detach().cpu())

    def stack_int_results(self):
        self.intgc_list = torch.vstack(self.last_intgc_list).numpy()
        self.intpool_list = torch.vstack(self.last_intpool_list).numpy()
        self.intmempool_list = torch.vstack(self.intmempool_list).numpy()
        self.emb_out_list = torch.vstack(self.emb_out_list).numpy()
        self.fc2_list = torch.vstack(self.fc2_list).numpy() if len(self.fc2_list) >0 else None

    def calc_cont_acc(self):
        y0 = torch.vstack(self.y_all_list).cpu().numpy().reshape(-1)
        yh = torch.vstack(self.output_list).detach().cpu().numpy().reshape(-1) # vstack multi batches's pred_y?
        #acc = np.dot(yh,y0) /(np.linalg.norm(yh) *np.linalg.norm(y0)) 
        acc = np.corrcoef(y0, yh, rowvar=False)[0,1]
        return acc

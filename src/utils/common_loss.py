
import torch

from .utils import calc_nll_loss_wt
from torch import linalg as LA
import torch.nn.functional as F
EPS = 1e-15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_reg(gamma): 
    reg = 0
    for i in gamma:
        sq = (1/i) * (1/i)
        reg = reg + sq
    return reg/len(gamma)

def R_decorr(manifold): # Decorrelation Term
    manifoldT = torch.transpose(manifold, 0, 1)

    mul = torch.matmul(manifoldT, manifold)

    ch = torch.div(mul, torch.numel(mul))

    fro = LA.norm(ch, 'fro')
    fro = torch.square(fro)

    d = torch.diagonal(ch, offset=0, dim1=0, dim2=1)

    d_2nd = LA.norm(d, 2)
    d_2nd = torch.square(d_2nd)
    
    term = torch.subtract(fro , d_2nd) 
    return term

def cont_correlation_loss(output,y): 
    return  (torch.linalg.norm(y) * 0.618 * 
            torch.nn.MSELoss()(
            torch.nn.functional.normalize(output.squeeze() - torch.mean(output.squeeze()),dim=0),
            torch.nn.functional.normalize(y - torch.mean(y),dim=0)) 
            + torch.nn.MSELoss()(output.squeeze(),y))

def consist_loss(s_list:list, W=None, target=0): # modify based on len of s
    assert isinstance(s_list,list), 'need a list, each element for one tissue!' 
    if len(s_list) == 0:
        return 0
    else:
        res_sum = 0
        for s in s_list:
            if target >= 0:
                W = torch.ones(s.shape[0],s.shape[0])
                D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
            else:  
                W = W.to(device)
                D = torch.diag(torch.sum(W,dim=1))
            L = D-W
            L = L.to(device)
            res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
            res_sum+=res
        return res_sum

def norm(x):
    return LA.norm(x, ord=2)

def W_consist(y):
    l = len(y)
    ww = torch.ones([l,l])
    for i in range(l):
        for j in range(i-1):
            ww[i][j] = torch.exp(2 * (1 - ((norm(y[i,:] - y[j,:])) / (norm(y[i,:]) +  norm(y[j,:]))) ) )
            ww[j][i] = ww[i][j]
    return ww

def ordinal_loss(z1,z00, zordinal):
    znum = len(zordinal)
    loss_z = torch.tensor(0.0,device=z00.device)
    bceloss =  torch.nn.BCELoss()
    sigmoid = torch.nn.Sigmoid()
    for k in range(znum-1):
        z0 = torch.ones_like(z00,device= z00.device)
        z0[z00 <= zordinal[k]] = 0
        loss_z += bceloss(sigmoid(z1[:,k]),z0.float())
    return loss_z


def dist_loss(s_list,ratio): 
    res_sum = 0
    for s in s_list:
        s = s.sort(dim=1).values
        res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)* (1 - ratio))]+EPS).mean()
        res_sum+=res
    return res_sum

def dist_loss_3dim(s,k): 
    s = s.sort(dim=1).values
    res =  -torch.log(s[:,:,-k:]+EPS).mean() -torch.log(1-s[:,:,:(s.size(1)-k)]+EPS).mean()
    return res

def zloss(model, mopt, data1, extra_output):
    if hasattr(mopt,'ztype') and mopt.ztype == 'cont':
        z1 = model.znet(extra_output['emb_out']).squeeze()
        z0 = data1.z[:,mopt.zidx].float().squeeze() 
        loss_z = torch.nn.MSELoss()(z1,z0)
    elif hasattr(mopt,'ztype') and mopt.ztype == 'ord':
        z1 = model.znet(extra_output['emb_out']).squeeze()
        z00 = data1.z[:,mopt.zidx].squeeze()
        loss_z = ordinal_loss(z1, z00, mopt.zordinal)
    else: 
        z1 = torch.nn.LogSoftmax(dim=1)(model.znet(extra_output['emb_out']))
        z0 = data1.z[:,mopt.zidx].long().squeeze() 
        loss_z = torch.nn.NLLLoss()(z1,z0)
    return loss_z

def serialized_zloss(model, opt, mopts, data, extras):
    alpha_z = 1.0 if not hasattr(mopts[0],'alpha_z') else mopts[0].alpha_z
    if hasattr(opt,'zordinal'):
        z1 = [model.znet(e['emb_out']).squeeze() for e in extras]
        z1 = torch.vstack(z1)
        z00 = [ d.z[:,m.zidx].squeeze() for d,m in zip(data,mopts)]
        z00 = torch.cat(z00)
        loss_z = ordinal_loss(z1,z00,mopts[0].zordinal)
    else:
        z1 = [model.znet(e['emb_out']).squeeze() for e in extras]
        z1 = torch.cat(z1)
        z0 = [d.z[:,m.zidx].float().squeeze() for d,m in zip(data,mopts)]
        z0 = torch.cat(z0) 
        loss_z = torch.nn.MSELoss()(z1,z0)
    return loss_z *alpha_z

def make_loss_fn(*lmbd):
    assert len(lmbd) == 12,"need of 10 lmbdas and 2 ratios"
    def _topk_loss(extra,device,lamb_w,lamb_s, ratio, mopt):
        s_list, w_list =[],[]
        loss =  torch.tensor(0.0, device=device)
        if lamb_w >0 or lamb_s >0:
            for k in range(mopt.gc):
                s_list.append(extra[f's{k+1}'])
                w_list.append(extra[f'w{k+1}'])
            for w,s in zip(w_list,s_list):
                if lamb_w >0:
                    loss += lamb_w* (torch.norm(w, p=2)-1) ** 2
                if lamb_s >0:
                    loss = loss + lamb_s* dist_loss([s], ratio)
        return loss

    def _loss_c_fn(opt, mopt, output, data1):
        nclass = opt.nclass if not hasattr(mopt,'nclass') else mopt.nclass
        target = data1.y.squeeze(1)
        if hasattr(opt,'no_balance') and opt.no_balance:
            wt=calc_nll_loss_wt(target,nclass,thre=4,alpha=0.5)    
        else:
            wt = torch.tensor([1 for clz in range(nclass)], dtype=torch.float, device = data1.y.device)
        if opt.target == -1:
            loss_c = torch.tensor(0.0, device=data1.y.device) if hasattr(opt,'vcdn_embedding') and opt.vcdn_embedding ==1  \
                     else lmbd[0]* (cont_correlation_loss(output, target))
        elif opt.target >= 0 :
            loss_c = lmbd[0]* (F.nll_loss(output, target, weight=wt) if opt.target ==0 \
                           else lmbd[0]* ordinal_loss(output,target, mopt.zordinal)) 
        return loss_c

    def _reg_loss_fn(opt,mopt,extra_output,data1,idx_data,WW_all):
        nclass = opt.nclass if not hasattr(mopt,'nclass') else mopt.nclass
        target = data1.y.squeeze(1)
        if opt.target == -1:
            if mopt.poolmethod=='topk' and lmbd[-2] < 1.0:
                loss_topk_consist = torch.tensor(0.0,device=data1.y.device) if lmbd[5] == 0 else (
                    lmbd[5]* consist_loss(extra_output['s1'],WW_all[idx_data,:][:,idx_data],opt.target))
            else:
                loss_topk_consist = torch.tensor(0.0,device=data1.y.device)
        elif opt.target >= 0 :
            loss_topk_consist = torch.tensor(0.0,device=data1.y.device)
            if mopt.poolmethod=='topk' and lmbd[-2] <1.0 and lmbd[5] != 0 and 's1' in extra_output:
                for i in range(nclass):
                    if len(target[target == i]) >0:
                        loss_topk_consist +=  consist_loss([extra_output['s1'][target == i]])
                loss_topk_consist = lmbd[5] * loss_topk_consist
        
        if mopt.poolmethod=='topk' and lmbd[-2] < 1 and (lmbd[1] >0 or lmbd[3]>0) and 's1' in extra_output:
            loss_topk = _topk_loss(extra_output, device=data1.y.device, lamb_w=lmbd[1], lamb_s=lmbd[3], ratio=lmbd[-2],mopt=mopt)
        else:
            loss_topk = 0.0
        
        if hasattr(opt,'memp') and opt.memp > 0 and 'loss3' in extra_output: 
            loss_mp = lmbd[6] * extra_output['loss3'] if lmbd[6] >0 else 0.0
            loss_mp2 = lmbd[7] * sum([dist_loss_3dim(extra_output[f'mp{k+1}'],1) for k in range(mopt.memp)]) if lmbd[7] >0 else 0.0 
        else:
            loss_mp  = 0.0
            loss_mp2 = 0.0
        bn_gamma_reg = 0 if lmbd[8] ==0 or extra_output['gamma'] is None     else lmbd[8] * calc_reg(extra_output['gamma'])  
        decorr_reg = 0.  if lmbd[9] ==0 or not 'intmempool' in extra_output  else lmbd[9] * R_decorr(extra_output['intmempool']) 
        losses = [loss_topk, loss_topk_consist, loss_mp2, bn_gamma_reg, decorr_reg]       
        loss = sum(losses)      
        return loss,(loss_mp,bn_gamma_reg,decorr_reg)
    return _loss_c_fn, _reg_loss_fn  

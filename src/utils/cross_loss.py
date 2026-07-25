import torch
from torch import linalg as LA
from .search_utils import remap_label
from .common_loss import cont_correlation_loss, serialized_zloss
def _columnvar_cov(v):
    m = torch.mean(v,0)
    v0 = v - m
    cov = torch.matmul(v0.T,v0)
    if v0.size(0) >1:
        cov = cov /(v0.size(0) -1)
    return cov

def clk_cross_loss(v,y,models,device):
    cross_loss = torch.nn.NLLLoss()
    loss_align = torch.tensor(0.0,device=device)
    v0, v1 = v
    y0, y1 = y
    y0x,_ = models[1].classif(v0,run='train')
    y1x,_ = models[0].classif(v1,run='train')
    loss_align = cross_loss(y0x,y0) + cross_loss(y1x, y1)
    return loss_align

def cross_kldiv(v,y,nclass):
    def _cross_kldiv(v):
        mean0 = torch.mean(v[0],dim=0)
        mean1 = torch.mean(v[1],dim=0)
        var0  = _columnvar_cov(v[0])
        var1  = _columnvar_cov(v[1])
        d = var0.size(0)
        var1inv = torch.inverse(var1)
        kldiv = torch.log(torch.abs(torch.det(var1) / torch.det(var0))) -d 
        kldiv += torch.trace(torch.matmul(var1inv,var0))
        meandiff = mean1-mean0
        kldiv += meandiff @ var1inv @ meandiff
        return kldiv/2.0

    kldiv = 0.0    
    for c in range(nclass):
        v0 = v[0][y[0] == c]
        v1 = v[1][y[1] == c]
        kldiv += _cross_kldiv((v0,v1))
    return kldiv

def filter_vy(v0,v1,y0,y1,nclass):
    w0,t0,w1,t1 =[],[],[],[]
    for i in range(nclass):
        if len(y0[y0 == i]) >0 and len(y1[y1 ==i])>0:
            w0.append(v0[y0==i])
            t0.append(y0[y0==i])
            w1.append(v1[y1==i])
            t1.append(y1[y1==i])
    v0 = torch.cat(w0)
    v1 = torch.cat(w1)
    y0 = torch.cat(t0)
    y1 = torch.cat(t1)
    return v0,v1,y0,y1

def cross_corr2(v,y,nclass=0,norm=False): 
    if norm:
        v0 = torch.nn.functional.normalize(v[0],dim=1)
        v1 = torch.nn.functional.normalize(v[1],dim=1)
    else:
        v0,v1 = v
    y0,y1 =y[0],y[1]
    if nclass >0:
        v0,v1,y0,y1 = filter_vy(v0,v1,y0,y1,nclass)
    cdist = torch.cdist(v0,v1) # cross distance
    diff = y0.view(-1,1) - y1.view(-1) #outer difference
    positives = torch.ones_like(diff,dtype=torch.float32)
    negatives = torch.ones_like(diff,dtype=torch.float32)
    positives[diff !=0] = 0
    negatives[diff ==0] = 0
    pcdist = cdist * positives 
    ncdist = cdist * negatives
    mpc = torch.sum(pcdist)/torch.sum(positives)
    mnc = torch.sum(ncdist)/torch.sum(negatives)
    return mpc/mnc

def cross_corr3(v,y,nclass=0,norm=False):
    if norm:
        v0 = torch.nn.functional.normalize(v[0],dim=1)
        v1 = torch.nn.functional.normalize(v[1],dim=1)
    else:
        v0,v1 = v
    y0,y1 = y[0],y[1]
    if nclass >0:
        v0,v1,y0,y1 = filter_vy(v0,v1,y0,y1,nclass)
    crossdot = torch.matmul(v0,torch.t(v1))
    diff = y0.view(-1,1) - y1.view(-1)
    positives = torch.ones_like(diff, dtype=torch.float32)
    positives[diff != 0] = 0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(crossdot,positives) # mean by default
    return loss

def align_corr(v, norm=True,abs=False):
    if norm:
        v0 = torch.nn.functional.normalize(v[0],dim=0)
        v1 = torch.nn.functional.normalize(v[1],dim=0)
    else:
        v0,v1 = v
    if abs:
        cor = torch.mean(torch.abs(torch.sum(v0*v1, dim =0)))
    else:
        cor = torch.mean(torch.sum(v0*v1, dim =0)) 
    return -cor
    
def align_dist(v, norm=False):
    if norm:
        v0 = torch.nn.functional.normalize(v[0],dim=0)
        v1 = torch.nn.functional.normalize(v[1],dim=0)
    else:
        v0,v1 = v
    dist = torch.mean(LA.norm(v0-v1,2,dim=0))
    return dist

def cross_correlation(intmempool, target, isRatio=True):
    d1,d2 = target
    manifold1, manifold2 = intmempool

    classinfo1 = torch.flatten(d1)
    classinfo2 = torch.flatten(d2)
        
    indices_classinfo10 = [index for index, element in enumerate(classinfo1) if element == 0] 
    indices_classinfo11 = [index for index, element in enumerate(classinfo1) if element == 1]
    indices_classinfo20 = [index for index, element in enumerate(classinfo2) if element == 0]
    indices_classinfo21 = [index for index, element in enumerate(classinfo2) if element == 1]

    indices_classinfo10 = torch.tensor(indices_classinfo10)
    indices_classinfo11 = torch.tensor(indices_classinfo11)
    indices_classinfo20 = torch.tensor(indices_classinfo20)
    indices_classinfo21 = torch.tensor(indices_classinfo21)

    man_10 = [manifold1[i] for i in indices_classinfo10]
    man_11 = [manifold1[i] for i in indices_classinfo11]
    man_20 = [manifold2[i] for i in indices_classinfo20]
    man_21 = [manifold2[i] for i in indices_classinfo21]

    man_10 = torch.stack(man_10)
    man_11 = torch.stack(man_11)
    man_20 = torch.stack(man_20)
    man_21 = torch.stack(man_21)

    loss_11 = torch.cdist(man_11, man_21, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    loss_00 = torch.cdist(man_10, man_20, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    loss_01 = torch.cdist(man_10, man_21, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    loss_10 = torch.cdist(man_11, man_20, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')

    d1 = torch.mean(loss_11)
    d2 = torch.mean(loss_00)
    d3 = torch.mean(loss_01)
    d4 = torch.mean(loss_10)

    num = torch.add(d1 , d2)
    den = torch.add(d3 , d4)
    
    loss5a = d1 + d2 - d3 - d4
        
    loss5b = torch.div(num , den)

    if not isRatio:
        return loss5a
    else:
        return loss5b

def center_point2(v,y,nclass):
    loss =0
    for klz in range(nclass):
        if len(y[0][y[0]== klz]) == 0 or len(y[1][y[1] == klz]) == 0:
            continue 
        loss += (torch.sqrt(torch.sum(torch.square(
                    torch.mean(v[0][y[0].view(-1) == klz],0) 
                  - torch.mean(v[1][y[1].view(-1) == klz],0))))
            )
    return loss

def center_point1(v, diffvar=False):
    return (torch.sqrt(torch.sum(torch.square(
                    torch.mean(v[0],0) 
                  - torch.mean(v[1],0))))
                  + (torch.sqrt(torch.sum(torch.square(
                    torch.log(torch.var(v[0],0)) 
                    -torch.log(torch.var(v[1],0)))))/v[0].shape[1] if diffvar else 0.0)
                  )
            
def pairwise_align_loss(models, data, extras, opt, mopts): 
    loss = torch.tensor(0.0,device=data[0].y.device)
    if opt.target == 0:
        cross_loss = torch.nn.NLLLoss()
    else:
        cross_loss = cont_correlation_loss
   
    no_species = len(models)
    ztarget_all = hasattr(opt,'ztarget_all') and opt.ztarget_all
    if ztarget_all: 
        loss_z = serialized_zloss(models[0], opt, mopts, data, extras)
        loss = loss + loss_z
    else:
        loss_z = torch.tensor(0.0, device=data[0].y.device)
    assert no_species >= 2,'implemented type 1 and 2' 
    if not hasattr(opt, 'align_pairs') and no_species == 2:
        opt.align_pairs =[(0,1)]
        opt.align_coeff = [mopts[0].alpha1]
    
    for (m0, m1), alpha_x in zip(opt.align_pairs, opt.align_coeff):
        v0 = extras[m0]['emb_out']
        v1 = extras[m1]['emb_out']
        y0 = data[m0].y
        y1 = data[m1].y
        
        if opt.target >=0 and hasattr(mopts[m0],'label_mapping'):
            y0 = remap_label(y0, mopts[m0].label_mapping)                
        if opt.target >=0 and hasattr(mopts[m1],'label_mapping'):
            y1 = remap_label(y1, mopts[m1].label_mapping)
        tnclass = torch.max(torch.cat([y0,y1])) + 1
        fc1_detached = hasattr(opt, 'fc1_detached') and opt.fc1_detached
        latent_anchor_m0 = hasattr(mopts[m0],'latent_anchor') and mopts[m0].latent_anchor
        latent_anchor_m1 = hasattr(mopts[m1],'latent_anchor') and mopts[m1].latent_anchor
        if latent_anchor_m0 and not fc1_detached: 
            v0 = v0.detach()
        if latent_anchor_m1 and not fc1_detached: 
            v1 = v1.detach()
       
        if True:
            fc1_ctr_backward = True
            if opt.fc1_ctr == 5 and opt.target == 0:
                loss_center_point= alpha_x * cross_corr2((v0,v1),(y0,y1),norm=True)
            elif opt.fc1_ctr == 4 and opt.target == 0: 
                loss_center_point= alpha_x * (cross_corr2((v0,v1),(y0, y1)) 
                                    + center_point2((v0,v1),(y0, y1), tnclass))
                                                
            elif opt.fc1_ctr == 3 and opt.target == 0:
                loss_center_point= alpha_x * cross_kldiv((v0,v1),(y0,y1),tnclass)
            elif opt.fc1_ctr == 2 and opt.target == 0: 
                loss_center_point= alpha_x * center_point2((v0,v1),(y0,y1),tnclass)
            elif opt.fc1_ctr == 1:  # for both cont and cat target
                loss_center_point= alpha_x * center_point1((v0,v1), opt.target != 0) #regardless of class
            else:
                loss_center_point = torch.tensor(0.0,device=v0.device)
                fc1_ctr_backward = False
        loss = loss + loss_center_point
        
        fc1_aliqn_backward = True
        if opt.fc1_align == 'common_regress' and opt.target == -1:
            mreg = m0 if hasattr(mopts[m0],'aux_regressor') else m1
            y_hat = torch.vstack([models[mreg].regressor(v0,data[m0].batch)[0],
                                models[mreg].regressor(v1,data[m1].batch)[0]])
            y0std = (y0-torch.mean(y0)) / torch.std(y0)
            y1std = (y1-torch.mean(y1)) / torch.std(y1)
            t_d   = torch.cat([y0std, y1std])
            loss_align = alpha_x * cross_loss(y_hat,t_d)
        elif opt.fc1_align == 'cross_regress' and opt.target == -1:
            y0x,_ = models[m1].regressor(v0,data[m0].batch)
            y1x,_ = models[m0].regressor(v1,data[m1].batch)
            loss_align = alpha_x * (cross_loss(y0x,y0) + cross_loss(y1x, y1))
        elif opt.fc1_align == 'clk' and opt.target == 0: 
            if False:
                loss_align = alpha_x * align_dist((v0,v1))
            else:    
                y0x,_ = models[m1].xclassif(v0,data[m0].batch)
                y1x,_ = models[m0].xclassif(v1,data[m1].batch)
                loss_align = alpha_x * (cross_loss(y0x,y0) + cross_loss(y1x, y1))
        elif opt.fc1_align == 'cor' and opt.target == 0:
            if False:
                loss_align = alpha_x * align_corr((v0,v1))
            else:    
                loss_align = alpha_x * cross_corr3((v0,v1),(y0, y1),tnclass)
        else:
            loss_align = torch.tensor(0.0,device=v0.device)
            fc1_aliqn_backward = False
        loss = loss + loss_align
    return loss, loss_z, loss_center_point, loss_align

def contrastive_loss_multiEdge(extra_output, data1, nclass): 
    y, k = data1.y, nclass
    fc1_ls = extra_output['in_fc1_ls']
    x,f_ls = fc1_ls[-1],fc1_ls[:-1],
    fmatrix   = torch.stack(f_ls,1)
    fpositive = fmatrix[list(range(len(y))), y.tolist()]
    fnegative = torch.stack([fmatrix[list(range(len(y))), torch.remainder(y-j, k).tolist()] 
                            for j in range(1,k)])
    sim    = torch.sum(x * fpositive, 1)/(LA.norm(x, 2, dim=1) * LA.norm(fpositive, 2, dim=1))
    dissim = (torch.sum(x.unsqueeze(0) * fnegative, -1) / 
                ( LA.norm(x, 2, dim=1).unsqueeze(0) * LA.norm(fnegative, 2, dim=-1) ))
    return  torch.log(torch.sum(torch.exp(dissim) / torch.exp(sim)))
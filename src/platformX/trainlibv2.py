import torch
import numpy as np
from  utils.search_utils import  calc_metrics
from utils.utils import predicate_xopt
from  utils.cross_loss import cross_corr3
EPS = 1e-15

from .MultiOmicsModel import  do_single_batch_multiomics
from .SingleModel import do_single_batch, IntResults

from torch.nn.modules.batchnorm import _BatchNorm
from collections import defaultdict
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
    model.apply(_enable)



def train_single(run,supermodel,loader,opt,mopt,epoch=0):  # mopt is mopts[0]
    device = next(supermodel.parameters()).device
    
    def predicate_single_model(model):
        return not hasattr(model, 'mopts')

    def optim2_zero_grad(model):
        if predicate_single_model(model):
            model.optimizer2.zero_grad()
        else:
            for m in model.models:
                m.optimizer2.zero_grad()

    if predicate_single_model(supermodel):
        if run == 'train':
            supermodel.train()
        else:
            supermodel.eval()
    else:
        for md in supermodel.models:
            if run == 'train':
                md.train()
            else:
                md.eval()

    losswt = supermodel.losswt if hasattr(supermodel,'losswt') and supermodel.losswt is not None else None
       
    def train_one_epoch(run,mopt):
        supermodel.saved_info = {'cpred':0, 'nout':0}
        if hasattr(supermodel,'models'):
            for m in supermodel.models:
                m.saved_info = {'cpred':0, 'nout':0}
        res_container = IntResults(opt)
        loss_all,lc_all = 0, 0
        cnt_b = 0
        if run == 'train' and hasattr(opt,'beta1'):
            optim2_zero_grad(supermodel)
        specific_sam_update_emb = opt.specific_sam_update_emb or not predicate_xopt(opt,'multimodal_data')
        def _mini_batch_loop(update_weights=True):
            retained_embeds = defaultdict(list)
            nonlocal cnt_b
            cnt_b = 0
            for data_batch in loader: 
                cnt_b +=1
                db_mopt = None
                model = supermodel 
                if isinstance(data_batch, dict) and data_batch['mopt'] is not None:
                    if len( data_batch['mopt']) == 1: 
                        model = data_batch['mopt'][0].model_
                    db_mopt = data_batch['mopt']
                if isinstance(data_batch, dict):
                    data_batch = data_batch['batch']
                def _forward(data_batch, _collect = True, losswt=None):
                    if predicate_single_model(model):
                        mopt = model.mopt 
                        model_name = mopt.tissue_name
                        WW_all = opt.WW_all if mopt.lamb5 !=0 and hasattr(opt,'WW_all') else None
                        data_batch = (WW_all, *data_batch)
                        fc1_detached = not specific_sam_update_emb #LY
                        output,extra_output,loss, lc = do_single_batch(model, data_batch, device, opt, run,fc1_detached=fc1_detached,lossr_only=True)
                        
                        _, data0, idx_data0, *_ = data_batch
                        emb_out =extra_output['emb_out']
                        loss_mp = extra_output['loss_mp']
                        if _collect:
                            if run == 'test':
                                res_container.collect_int_results(extra_output, mopt)
                            res_container.collect_results(data0,output, emb_out)
                        if predicate_xopt(opt,'multimodal_data') and not predicate_xopt(mopt,'samesplit'):
                            fac = losswt[model_name] if losswt is not None else 1.0
                            lc *= fac
                            loss_mp = extra_output['loss_mp'] * fac
                        loss = loss + lc    
                    else: 
                        mopts = db_mopt if db_mopt is not None else  model.mopts
                        model_name='multi'
                        output, extra_emb, loss, lc, extra_output2, data0, idx_data0, extra_output  = \
                        do_single_batch_multiomics(model,opt,device, epoch, data_batch, mopts=mopts, run=run,lossr_only=True)
                        
                        if opt.m_omics_correlation_mode == 'modality': 
                            t_emb_out = extra_emb['emb_out'] 
                            emb_out=[]
                            for m,exto in zip(mopts, extra_output):
                                emb_out.append( (m.tissue_name, exto['emb_out']) ) # embedding from ind model
                        else:  
                            t_emb_out = emb_out = extra_emb['emb_out'] # embedding from  transformer
                        if _collect:
                            if run == 'test':
                                res_container.collect_int_results(extra_output2[0], mopts[0])
                            res_container.collect_results(data0,output,t_emb_out)
                        loss_mp = torch.tensor(0.0,device=device)
                        for i in range(len(extra_output)):
                            loss_mp += extra_output[i]['loss_mp']
                        if not predicate_xopt(mopts[0],'samesplit'):
                            fac = losswt['multi'] if losswt is not None else 1.0
                            lc *= fac
                            loss_mp = loss_mp*fac
                        loss = loss + lc
                    model.saved_info['cpred'] += sum(output.cpu().argmax(axis=1) == data0.y.detach().cpu().squeeze(1))
                    model.saved_info['nout']  += len(output)
                    return loss, lc, idx_data0,loss_mp, (emb_out,data0.y)
                
                loss, lc, idx_data0, loss_mp, embed = _forward(data_batch,_collect=update_weights) #,losswt=losswt) # loss not include loss_mp if mempool_epoch_update
                
                if predicate_xopt(opt,'train_verbose'):
                    print('batch:',cnt_b,'-',model.mopt.tissue_name if not hasattr(model,'models') else 'multi-modal')
                
                if hasattr(model, 'epoch_lc_'):
                    model.epoch_lc_.append(lc.detach(),len(data_batch[0].samid))

                if run == 'train':
                    if update_weights:
                        model.zero_grad()
                        loss.backward()
                        model.update_weights()
                        if model.is_optimizer_SAM():
                            disable_running_stats(model)
                            loss, lc, idx_data0,_ = _forward(data_batch, _collect = False)
                            loss.backward()
                            model.update_weights(first_step = False)
                            enable_running_stats(model)
                        if hasattr(opt,'lr_scheduler') and opt.lr_scheduler == 'cosine':
                            model.step_scheduler()
                    else:
                        m_omics_correlation = hasattr(mopt,'align_coeff_multiomics2') and mopt.align_coeff_multiomics2 >0
                        need_align = m_omics_correlation and not predicate_xopt(mopt,'samesplit') and predicate_xopt(opt,'multimodal_data')
                        if need_align: 
                            if predicate_single_model(model):
                                if specific_sam_update_emb: 
                                    retained_embeds[model.mopt.tissue_name].append(embed)
                            else:
                                if opt.m_omics_correlation_mode == 'modality':
                                    for tn,em in embed[0]:
                                        retained_embeds[tn].append((em,embed[1]) )
                                elif opt.m_omics_correlation_all:
                                    if len(model.mopts) == 2:
                                        retained_embeds['two'].append(embed)
                                    else:
                                        retained_embeds['three'].append(embed)
                        if not predicate_single_model(model) or specific_sam_update_emb: #LY
                            retained_embeds['loss_mp'].append(loss_mp) 
                            retained_embeds['loss'].append(loss) 
                if run !='test':
                    nonlocal loss_all,lc_all
                    loss_all += loss.item() 
                    lc_all   += lc.item()            
            if run == 'train' and update_weights: 
                model.zero_grad()
            return retained_embeds
        
        retained_embeds = _mini_batch_loop(update_weights=True)
        if run == 'train':
            if run == 'train' and (not hasattr(opt,'lr_scheduler') or opt.lr_scheduler =='steplr'):
                supermodel.step_scheduler()
            loss2 = torch.tensor(0.0, device = device)
            loss_align = torch.tensor(0.0, device = device)   
            need_2nd_frd_1 = hasattr(mopt,'memp') and mopt.memp > 0 and predicate_xopt(opt,'mempool_epoch_update') # for memp
            
            m_omics_correlation = hasattr(mopt,'align_coeff_multiomics2') and mopt.align_coeff_multiomics2 >0
            need_2nd_frd_2 = m_omics_correlation and not predicate_xopt(mopt,'samesplit') and predicate_xopt(opt,'multimodal_data')
            
            if (need_2nd_frd_1 or need_2nd_frd_2):
                
                if (epoch % opt.n_epochs == 0 and epoch > 0) or hasattr(opt,'beta1'): # and predicate_xopt(opt,'mempool_epoch_update'):                 
                    optim2_zero_grad(supermodel)

                retained_embeds = _mini_batch_loop(update_weights=False)
                if need_2nd_frd_1:
                    loss2 = torch.sum(torch.stack(retained_embeds['loss_mp']))
                    
                del(retained_embeds['loss_mp'])
                del(retained_embeds['loss']) 
                if need_2nd_frd_2 and len(retained_embeds.values()) >1:
                    correlation_inputs = []
                    for rd in retained_embeds.values():
                        emb, tgt = list(zip(*rd))
                        if len(emb) > 0:
                            emb = torch.vstack(emb)
                            tgt = torch.vstack(tgt)
                            correlation_inputs.append((emb, tgt))
                    cnt =0 
                    
                    for vy0, vy1 in zip(correlation_inputs[:-1], correlation_inputs[1:]):
                        v, y = list(zip(vy0,vy1))
                        pairwise_coef = mopt.align_coeff_multiomics2
                        
                        loss_align = loss_align + cross_corr3(v,y) * pairwise_coef # cross_corr3 is mean among samples by default
                        
                        cnt +=1
                    loss_align = loss_align/cnt    
                    if predicate_xopt(opt,'train_verbose'):
                        print(f'loss_avgBatch:{loss_all/cnt_b}, lossc_avgBatch:{lc_all/cnt_b}, loss_align:{loss_align.item()}, loss_mp_all:{loss2.item()}')
                loss = loss2 + loss_align
                loss.backward()
                
                if predicate_single_model(supermodel):
                    supermodel.optimizer2.step()
                else:
                    for md in supermodel.models:
                        md.optimizer2.step()
                supermodel.step_scheduler2()

            loss_all = loss_all/cnt_b + loss2.item() + loss_align.item()
            lc_all = lc_all/cnt_b 
        return res_container, loss_all, lc_all 

    res_container, loss_all, lc_all = train_one_epoch(run,mopt)
    if run =='test':
        res_container.stack_int_results()  
    
    y_all, pred_y, samid_list, second_emb = res_container.get_stacked_results()

    acc = calc_metrics(opt, pred_y, y_all)  # returns AUC for binary classification

    raw_acc = np.mean(pred_y.argmax(axis=1) == y_all.squeeze(1)) if opt.target == 0 else acc

    res = {'acc':acc, 'raw_acc':raw_acc, 'l_all':loss_all, 'lcall':lc_all,
            'pred_y': pred_y,'y_all': y_all,
            'samid_list': samid_list, 'second_emb': second_emb,
        'emb_out':res_container.emb_out_list, 'fc2':res_container.fc2_list,
        'intgc':res_container.intgc_list, 'intpool':res_container.intpool_list, 
        'intmempool': res_container.intmempool_list}
    return  res


def train(run, model:list, loader:list, opt,mopt):
    if len(model) == 1:
        return train_single(run, model[0], loader[0], opt,mopt, epoch=train.epoch)
       
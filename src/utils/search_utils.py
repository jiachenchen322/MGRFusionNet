import torch
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.linear_model import LinearRegression
from shutil import rmtree
import h5py


def featurewise_int_rep(data, n_samples, n_rows, n_features):
    featurewise_data = np.zeros(data.shape)
    write_count = 0
    for i in range(n_features):
        for j in range(np.floor_divide(n_rows, n_samples)): 
            featurewise_data[:,write_count] = data[:,i+n_features*j]
            write_count += 1 
    print('featurewise shape',featurewise_data.shape)
    return featurewise_data

def generate_intermediate_header(n_samples, n_rows, n_features):
    header = 'samid'
    for j in range(n_features):
        for i in range(np.floor_divide(n_rows, n_samples)):
            header = header + ',c_' + str(i+1) + ':' + str(j+1)
    return header

def datasetmerge(dlist):
    x_arr = torch.stack((dlist[0].data.x, dlist[1].data.x), 0)
    ed_in_arr = torch.stack((dlist[0].data.edge_index, dlist[1].data.edge_index), 0)
    ed_att_arr = torch.stack((dlist[0].data.edge_attr, dlist[1].data.edge_attr), 0)
    pos_arr = torch.stack((dlist[0].data.pos, dlist[1].data.pos), 0)
    y = dlist[0].data.y
    data = Data(x=x_arr, edge_index = ed_in_arr, y=y, edge_attr=ed_att_arr, pos = pos_arr)
    return data 

def map_label(y, mapping):
    z = y.clone()
    for k,v in mapping.items():
        if k!=v: z[y==k] =v
    return z

def remap_label(y,mapping):
    z = y.clone()
    for k,v in mapping.items():
        if k!=v :z[y==v] =k
    return z

def target_mapping(dataset, tidx,permute=None,mopt=None):
    def _make_cat_vector(y):
        clzs= torch.unique(y) #mopt.nclass
        cnt = y.shape[0] // len(clzs)
        temp =[torch.ones(cnt,1)*clz for clz in clzs[:-1]]
        temp.append(torch.ones(y.shape[0]-cnt*(len(clzs)-1),1)*clzs[-1])
        temp = torch.cat(temp,0)
        return temp

    y = dataset.data.z
    temp = [y[:,i].reshape(-1,1) for i in tidx]
    y = torch.cat((temp), 1)
    if mopt is not None and hasattr(mopt,'label_mapping'):
        assert isinstance(mopt.label_mapping,dict), 'label_mapping requires dict!'
        y = map_label(y,mopt.label_mapping) 
    if permute is not None:
        if mopt is not None and mopt.targettype >=0:
            temp = _make_cat_vector(y)
            y = temp[permute].long()
        else:
            y = y[permute]
        if mopt is not None and hasattr(mopt,'ztarget'): #assume ztarget is a single target!
            zidx = np.where(dataset.data.cols[0]== mopt.ztarget)[0]
            if mopt.ztype != 'cont':
                temp = _make_cat_vector(dataset.data.z[:,zidx])
                dataset.data.z[:,zidx] = temp[permute].long()
            else:
                dataset.data.z[:,zidx.item()] = dataset.data.z[permute,zidx.item()]
    return y

def cv_mapping(dataset, cidx):
    cv = dataset.data.z   
    #cv = dataset.data.y
    temp = [cv[:,i].reshape(-1,1) for i in cidx]
    cv = torch.cat((temp), 1)
    return cv

def label_remapping(dataset, nclass): # oblete
    if nclass == 2:
        dataset.data.y[dataset.data.y == 2] = 0
        dataset.data.y[dataset.data.y == 3] = 1 
    elif nclass == 3:
        dataset.data.y[dataset.data.y == 2] = 0
        dataset.data.y[dataset.data.y == 3] = 2  
    return dataset.data.y

def load_fc1_score(temp_path, mopt, fc=1):
    INT_PATH = os.path.join(temp_path, 'int')
    int_csv_suf = mopt.tissue_name + '_' + mopt.target_ + '.csv' 
    fc1csv =os.path.join(INT_PATH , f'fc{fc}_' +int_csv_suf)
    fc1df = pd.read_csv(fc1csv)
    fc1arr = fc1df[fc1df.columns[1:]].to_numpy()
    return fc1arr

def load_intmempool1_pca(temp_path,mopt):
    INT_PATH = os.path.join(temp_path, 'int')
    int_csv_suf = mopt.tissue_name + '_' + mopt.target_ + '.csv' 
    mempoolcsv =os.path.join(INT_PATH , 'intmempool1_' +int_csv_suf)
    mempooldf = pd.read_csv(mempoolcsv)
    mp= mempooldf[mempooldf.columns[1:]].to_numpy()
    pca = PCA(n_components=2, svd_solver='full') # upto 10
    pca_out = pca.fit_transform(mp)
    return pca_out

def load_y_all(opt,mopt):
    EXP_PATH,BEST_PATH, PCA_PATH, LOAD_EXP_PATH, LOAD_BEST_PATH = make_path(3,opt,mopt)
    yallcsv = os.path.join(PCA_PATH ,'predicted.csv')
    yalldf = pd.read_csv(yallcsv)
    yall = yalldf['target'].to_numpy()
    ypred = yalldf['pred0'].to_numpy()
    return yall, yalldf['samid'].to_list(),ypred

def make_saveall_info(run,temp_path,opt,mopt, BEST_PATH=None, best_params=None, start_epoch=0, end_epoch=0, no_tissues=1):
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(temp_path)

    def _save_all_in_tr(model_dict, score_dict=None):
        loss_df = pd.DataFrame() 
        loss_df['tr_loss'] = model_dict['model'].history['loss']
        loss_df['val_loss'] = model_dict['model'].history['val_loss']
        #else:
        acc_df = pd.DataFrame(columns=['tr_acc', 'val_acc'])
        acc_df['tr_acc'] = model_dict['model'].history['metric']
        acc_df['val_acc'] = model_dict['model'].history['val_metric']
        
        loss_df.to_csv(os.path.join(temp_path,'loss.csv'), index=False)
        acc_df.to_csv(os.path.join(temp_path,'acc.csv'), index=False)
        with h5py.File(os.path.join(temp_path,'loss-acc.h5'),'w') as h5f:
            h5f.create_dataset('loss', data=loss_df.to_numpy())
            h5f.create_dataset('acc',  data=acc_df.to_numpy())
        
        with open(BEST_PATH, 'a') as f:
            s =  f"{model_dict['seq']}\t{best_params}\t{model_dict['best_epoch']}\t{model_dict['prev_best_loss']:1.3f}\t"
            s += f"{model_dict['best_train_loss']:1.3f}\t"
            s += f"{model_dict['prev_best_acc']:1.3f}\t{model_dict['best_train_acc']:1.3f}\t"
            s += f"{model_dict.get('prev_best_raw_acc', model_dict['prev_best_acc']):1.3f}\t{model_dict.get('best_train_raw_acc', model_dict['best_train_acc']):1.3f}\n"
            f.write(s)
        print('saving..', temp_path + '/model.pt')
        torch.save({
        'epoch': model_dict['best_epoch'],
        'model_state_dict': model_dict['best_model_wts'],
        'optimizer_state_dict': model_dict['best_optimizer'],
        }, os.path.join(temp_path, 'model.pt'))

        str_talos_seq =  os.path.basename(temp_path)
        conf_mat_fname = f'conf_mat_{str_talos_seq}_E{model_dict["best_epoch"]}.txt'
        with open( os.path.join(temp_path, conf_mat_fname),"w") as f:
            for k,v in model_dict['bestmetric_dict'].items():
                f.write(f'{k}:\n')
                f.write(f'{str(v)}\n')
    def _save_all_in_test(model_dict, score_dict):       
        if os.path.exists(temp_path):
            rmtree(temp_path)  
        INT_PATH = os.path.join(temp_path,'int')
        target_list = mopt[0].target_ if isinstance(mopt,list) else mopt.target_ 
        int_csv_suf = target_list + '.csv'  
        if not os.path.exists(INT_PATH):
            os.makedirs(INT_PATH)
        PCA_PATH = os.path.join(temp_path,'PCA')
        if os.path.exists(PCA_PATH):
            shutil.rmtree(PCA_PATH)
        os.makedirs(PCA_PATH)
        samid_list = score_dict['samid_list']
        def _save_featurewise(layername):
            n_sam = len(samid_list)
            dim0 = score_dict[f'{layername}'].shape[0]
            dim1 = score_dict[f'{layername}'].shape[1]
            layer_all = np.reshape(score_dict[f'{layername}'],(n_sam,-1))
            layer_header = generate_intermediate_header(len(samid_list), dim0,dim1)
            layer_featurewise = featurewise_int_rep(layer_all, n_sam, dim0,dim1)
            np.savetxt(os.path.join(INT_PATH, f'{layername}_{int_csv_suf}'),
                        np.hstack((samid_list, layer_featurewise)), 
                        delimiter=",", fmt=('%s'), comments='', header=layer_header)
       
        _save_featurewise('intgc')
        _save_featurewise('intpool')
        _save_featurewise('intmempool')

        if len(score_dict['y_all'].shape) == 1:
                score_dict['y_all'] = np.expand_dims(score_dict['y_all'], 1)
        if len(score_dict['y_pred_all'].shape) == 1:
            score_dict['y_pred_all'] = np.expand_dims(score_dict['y_pred_all'], 1)
        if opt.target >= 0:
            with open(os.path.join(PCA_PATH,"conf_mat.txt"),"w") as f:
                f.write(str(score_dict['cm']))
        elif opt.target == -1: # plot
            pred_plot2(score_dict['y_pred_all'], score_dict['y_all'], mopt.target_list, PCA_PATH, score_dict['tr_index'], score_dict['te_index'], mopt.cv_list) 
            reg = LinearRegression().fit(score_dict['y_pred_all'],score_dict['y_all'])
            y_pred_all2 = reg.predict(score_dict['y_pred_all'])
            
            header =','.join(['samid']+['pred'+str(i) for i in range(score_dict['y_pred_all'].shape[1])]+['target'])
            np.savetxt(os.path.join(PCA_PATH, 'lm-predicted.csv'), np.hstack((samid_list, y_pred_all2, score_dict['y_all'])), 
                delimiter=",", fmt=('%s,%s,%s'), comments='', header= header) 
            pred_plot2(y_pred_all2, score_dict['y_all'], mopt.target_list, PCA_PATH + 'lm-', score_dict['tr_index'], score_dict['te_index'], mopt.cv_list)
        
        header =','.join(['samid']+['pred'+str(i) for i in range(score_dict['y_pred_all'].shape[1])]+['target'])
        np.savetxt(os.path.join(PCA_PATH, 'predicted.csv'), 
                   np.hstack((samid_list, score_dict['y_pred_all'],
                              score_dict['y_all'] if len(score_dict['y_all'].shape)>1 else np.expand_dims(score_dict['y_all'], axis=1))),
                delimiter=",", fmt=(','.join(['%s']*(2+score_dict['y_pred_all'].shape[1]))),  comments='', header=header)
        
        if 'emb_out' in score_dict.keys() and score_dict['emb_out'] is not None:
            header = ','.join(['samid']+['col'+str(i) for i  in range(score_dict['emb_out'].shape[1])]) 
            np.savetxt(os.path.join(INT_PATH, 'fc1_' +int_csv_suf), np.hstack((samid_list,score_dict['emb_out'])), 
                    delimiter=",", fmt=(','.join(['%s']*(1+score_dict['emb_out'].shape[1]))),  comments='', header=header)
            if opt.target >= 0:  # save png files
                fc1 = score_dict['emb_out']
                if fc1.shape[1] == 3:
                    scatter_plot3(fc1[:,0],fc1[:,1],fc1[:,2],score_dict['y_all'], f'{PCA_PATH}/fc1_3.png')
                    scatter_plot3(fc1[:,1],fc1[:,2],fc1[:,0],score_dict['y_all'], f'{PCA_PATH}/fc1_3-120.png')
                elif fc1.shape[1] == 2:
                    scatter_plot2(fc1[:,0],fc1[:,1],score_dict['y_all'], f'{PCA_PATH}/fc1_2d.png')    
                else: # >3
                    pca2 = PCA(n_components=2, svd_solver='full') # upto 10
                    pca_out = pca2.fit_transform(fc1)
                    scatter_plot2(pca_out[:,0],pca_out[:,1],score_dict['y_all'], f'{PCA_PATH}/fc1_2PCs.png') 
                mp = score_dict['intmempool']
                pca3 = PCA(n_components=3, svd_solver='full') # upto 10
                pca_out = pca3.fit_transform(mp)
                scatter_plot3(pca_out[:,0],pca_out[:,1],pca_out[:,2],score_dict['y_all'], f'{PCA_PATH}/mempool-pca3d.png')
                scatter_plot3(pca_out[:,1],pca_out[:,2],pca_out[:,0],score_dict['y_all'], f'{PCA_PATH}/mempool-pca3d-120.png')
        
        if 'fc2' in score_dict.keys() and score_dict['fc2'] is not None: 
            header = ','.join(['samid']+['col'+str(i) for i  in range(score_dict['fc2'].shape[1])]) 
            np.savetxt(os.path.join(INT_PATH, 'fc2_' +int_csv_suf), np.hstack((samid_list,score_dict['fc2'])), 
                    delimiter=",", fmt=(','.join(['%s']*(1+score_dict['fc2'].shape[1]))),  comments='', header=header)
        
        if 'second_emb' in score_dict.keys() and score_dict['second_emb'] is not None: 
            layer_all = score_dict['second_emb']
            if isinstance(layer_all,list):
                layername = 'attn_wts'
            else:
                layername = 'multiomics_emb'
                layer_all = [layer_all]
            no_sam = samid_list.shape[0]
            for j in range(len(layer_all)):    
                if len(layer_all[j].shape)==2:
                    n_cols = layer_all[j].shape[1] 
                else:
                    n_cols = layer_all[j].shape[1] * layer_all[j].shape[2]
                header = ','.join(['samid']+['col'+str(i) for i  in range(n_cols)]) 
                np.savetxt(os.path.join(INT_PATH, f'{layername}_{j}_{int_csv_suf}'), 
                           np.hstack((samid_list,layer_all[j].reshape(no_sam,-1))), 
                            delimiter=",", fmt=(','.join(['%s']*(1+n_cols))), comments='', header=header)
    return _save_all_in_tr if run == 'train' else _save_all_in_test

def accumulate_across_batches(train, extra_output, opt,mopt, sn_arr_list, x3_list, intout_list):
    if x3_list == None:
        x3_list = []
    if intout_list == None:
        intpool1_list, intpool2_list, intmempool_list, intgc1_list, intgc2_list = [],[],[],[],[]
    else:
        intpool1_list, intpool2_list, intmempool_list, intgc1_list, intgc2_list = intout_list    
    if sn_arr_list == None:
        s1_list = []
        s2_list = []
        s3_list = []
        s1_arr_list = []
        s2_arr_list = []
        s3_arr_list = []
    else: 
        s1_arr_list,s2_arr_list,s3_arr_list, s1_list,s2_list,s3_list = sn_arr_list
    s1 = extra_output['s1'] 
    s2 = extra_output['s2']
    s3 = extra_output['s3']  
    if train != 'test':
        s1_list.append(s1.view(-1).detach().cpu().numpy())  
        s1_arr_list = np.hstack(s1_list) 
        
    else:
        s1_list.append(s1.detach().cpu().numpy())
    if True: 
        if train != 'test':
            s2_list.append(s2.view(-1).detach().cpu().numpy())
            s2_arr_list= np.hstack(s2_list) 
            s2_arr_list= np.hstack(s2_list) 
            s2_arr_list= np.hstack(s2_list) 
            
        else:
            s2_list.append(s2.detach().cpu().numpy())

    if True: #opt.memp == 1:
           
        if train != 'test':
            s3_list.append(s3.view(-1).detach().cpu().numpy()) # ZF added .view(-1) to match similar lines and resolve dimension mismatch
            s3_arr_list = np.hstack(s3_list)
        else:
            s3_list.append(s3.detach().cpu().numpy())
    if train == 'test':
        x3_list.append(extra_output['x3'])
        intpool1_list.append(extra_output['intpool1'])
        intpool2_list.append(extra_output['intpool2'])
        intmempool_list.append(extra_output['intmempool'])
        intgc1_list.append(extra_output['intgc1'])
        intgc2_list.append(extra_output['intgc2'])
    return [s1_arr_list,s2_arr_list,s3_arr_list, s1_list, s2_list,s3_list],x3_list,\
        [intpool1_list,intpool2_list,intmempool_list,intgc1_list,intgc2_list]

def calc_metrics(opt, output, target):
   
    assert len(output.shape) == len(target.shape)
    acc =0.0
    target = target.squeeze(1)
    if opt.target == -1:
        target = torch.tensor(target)
        ty1 = target.view(-1)
        acc = np.corrcoef(ty1, output, rowvar=False)[0,1]
    elif opt.target == 0:
        if hasattr(opt,'sequential'):
            y2 = np.concatenate([target for _ in range(2)]) # for seqEmbedding
        else:
            y2 = target
        try:
            probs = np.exp(output)[:, 1]  # positive class probability from log_softmax
            acc = roc_auc_score(y2, probs)
        except Exception:
            pred = output.argmax(axis=1)
            acc = np.mean(pred == y2)
    else: #opt.target == 1
        pred = np.ones_like(output)
        pred[output < 0.5] = 0
        pred = np.sum(np.cumprod(pred,axis=1), axis = 1)
        acc = np.mean(pred == target)
    return acc

###################### Network Training Function ####################################
def scatter_plot2(x1,x2, y, path):
    plt.close('all')
    fg = plt.figure()
    sc = plt.scatter(x1,x2, c=y,alpha=0.5)
    plt.legend(*sc.legend_elements(),loc ='best', title="Classes")
    plt.savefig(path)
    plt.close()

def scatter_plot3(x1,x2,x3, y, path):
    plt.close('all')
    fg = plt.figure()
    ax = plt.axes(projection='3d')
    sc = ax.scatter3D(x1,x2,x3, c=y)
    plt.legend(*sc.legend_elements(),loc ='best', title="Classes")
    plt.savefig(path)
    plt.close()

def loss_plot(start_epoch, end_epoch, loss, acc, tl, loss_plot_path):
    # plot_type='acc' or 'loss'
    val_loss, train_loss = loss['val_loss'],loss['tr_loss']
    labels = [int(i) + start_epoch for i in range(len(val_loss))]
    if start_epoch != 0:
        labels.insert(0,start_epoch-1)
    labels.append(labels[-1] + 1)
    # print(labels)
    plot_type = 'loss'
    fg = plt.figure(figsize=(10,12))
    #ax = fg.gca()
    ax = plt.subplot(211)
    ax.plot(val_loss, 'o', linestyle='-')
    ax.plot(train_loss, 'o', linestyle='-')
    ax.set_title(f'{tl}-{plot_type}_plot')
    ax.legend([f'val_{plot_type}', f'train_{plot_type}'])
    ax.set_xticklabels(labels)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=40,integer=True))

    plot_type = 'acc'
    val_acc, train_acc = acc['val_acc'], acc['tr_acc']
    ax = plt.subplot(212)
    ax.plot(val_acc, 'o', linestyle='-')
    ax.plot(train_acc, 'o', linestyle='-')
    # ax.set_yscale('log')
    ax.set_title(f'{tl}-{plot_type}_plot')
    ax.legend([f'val_{plot_type}', f'train_{plot_type}'])
    ax.set_xticklabels(labels)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=40,integer=True))

    plot_type = 'acc_loss'
    plt.savefig(os.path.join(loss_plot_path, f'{plot_type}_plot.png'))
    plt.close()

def get_pca_ev(inter_pred, tissue, samid_lst, param_dict, pca_path, root_prefix1): 
    
    dim3 = int(param_dict[param_dict.find('dim3') + 5])
    n_comp = min(inter_pred.shape[1], 4) # ZF changed first element to number of columns in inter_pred
    pca = PCA(n_components=n_comp, svd_solver='full') # upto 10
    pca_out = pca.fit_transform(inter_pred)
    # pca.fit(inter_pred)
    # pca_out = pca.transform(inter_pred)
    pca_ev = pca.explained_variance_ratio_
    if not os.path.exists(pca_path):
        os.makedirs(pca_path)
    np.savetxt(pca_path + 'PCA_ev.csv', pca_ev, delimiter=",")
    
    pca_results = pd.DataFrame(samid_lst, columns=['samid'])
    for i in range(n_comp):
        pca_results['PC' + str(i+1)] = pca_out[:,i]   
    # #reorder rows of pca_results according to gene's samid sequence      
    # gene = pd.read_csv(root_prefix1 + tissues[0] + '-gene.csv', sep=',')
    # pca_results.index = pca_results['samid']
    # pca_results = pca_results.reindex(gene['samid'].tolist()).reset_index(drop=True) 
    # #index of pca_results will cause corrwith to produce nan,so drop it after reordering.  
    pca_results.to_csv(pca_path + 'PCA.csv',index=False)
    #for tissue in tissues:
    gene = pd.read_csv(root_prefix1 + tissue + '-gene.csv', sep=',')
    for i in pca_ev:
        if i>0.1:
            idx = list(pca_ev).index(i) + 1
            pcorr = gene.iloc[:,1:].corrwith(pca_results.iloc[:,idx])
            pcorr = pcorr.to_frame().reset_index()
            pcorr.rename(columns={'index':'gene_id', 0:'pc_' + str(idx) + '_corr'}, inplace=True)
            pcorr.to_csv(pca_path + tissue + '_pc_' + str(idx) + '_corr.csv')
    #for i in range(len(tl)):
    #    pca_plot(pca_results['PC1'], y_all[:,i], tl[i], pca_path)
    ## pca_plot(pca_results['PC1'], df2[y[1]])

def general_plot(x0,y0, mainlabel, plot_path, tr_index,te_index):
    titles=(None, 'target','pred')
    if tr_index and te_index:
        _,axes = plt.subplots(1,2,figsize=(9, 4.5),
        tight_layout=True, sharey=True)
        ratio = 1.1
        xlim = (np.min(x0)*ratio,np.max(x0)*ratio)
        ylim = (np.min(y0)*ratio,np.max(y0)*ratio)
        for ax in axes:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        axes[0].plot(x0[tr_index],y0[tr_index],'o',c='r')
        #axes[0].legend(['train'])
        tcorr = np.corrcoef(x0[tr_index],y0[tr_index], rowvar=False)
        axes[0].set(title=f'train corr={tcorr[0,1]:.3f}',xlabel=titles[1],ylabel=titles[2])
        axes[0].grid()
        axes[0].label_outer()
        axes[1].plot(x0[te_index],y0[te_index],'o',c='g')
        #axes[1].legend(['val'])
        tcorr = np.corrcoef(x0[te_index],y0[te_index], rowvar=False)
        axes[1].set(title=f'val corr={tcorr[0,1]:.3f}', xlabel=titles[1],ylabel=titles[2])
        axes[1].grid()
        axes[1].label_outer()
        plt.savefig(plot_path+'-tiles.png')
        plt.close()
    
    fg = Figure()
    plt.plot(x0,y0,'o')
    plt.legend(['all'])
    plt.grid()
    tcorr = np.corrcoef(x0,y0, rowvar=False)
    plt.title(f'{mainlabel} corr= {tcorr[0,1]:.3f}')
    plt.xlabel(titles[1])
    plt.ylabel(titles[2])
    plt.savefig(plot_path+'.png')
    plt.close()

def pca_plot(pc, y, tl_name, pca_path):
    plt.figure(figsize=(20, 10))
    plt.title(str(pc.name) + ' vs ' + str(tl_name))
    plt.scatter(pc, y)
    plt.xlabel(str(pc.name))
    plt.ylabel(str(tl_name))
    # plt.savefig(pca_path + str(tl_name) + '.png')
    plt.savefig(pca_path + str(tl_name) + '_PC.png') 

def pred_plot(y_pred, y, tl, pca_path):
    for i in range(len(tl)):
        plt.figure(figsize=(20, 10))
        plt.title(tl[i] + '_pred' + ' vs ' + tl[i])
        #plt.scatter(y_pred, y)
        plt.scatter(y_pred[:,i], y[:,i])
        plt.xlabel(tl[i] + '_pred')
        plt.ylabel(tl[i])
        plt.savefig(pca_path + tl[i] + '_pred' +'.png') 

def pred_plot2(y_pred, y, tl, pca_path, tr_index,te_index, cv):
    for i in range(len(tl) - len(cv)):
        tpath = pca_path + tl[i] + '_pred' 
        general_plot(y[:,i:(i+1)], y_pred[:,i:(i+1)], tl[i]+' true vs pred', tpath, tr_index, te_index)


def check_condition(adi_dataset, liv_dataset, cv_list_1, cv_list_2, target_list):
    if type(adi_dataset.data.cols) != list:
        adi_dataset.data.cols = adi_dataset.data.cols.tolist()
    merge_anno_cols_list = []
    merge_anno_cols_1 = adi_dataset.data.cols
    ct_list_1 = target_list + cv_list_1
    merge_anno_cols_list.append(merge_anno_cols_1)
    if set(ct_list_1) <= set(merge_anno_cols_1):
        flag = True
    else:
        flag = False
    if liv_dataset != None:
        if type(liv_dataset.data.cols) != list:
            liv_dataset.data.cols = liv_dataset.data.cols.tolist()
        merge_anno_cols_2 = liv_dataset.data.cols
        ct_list_2 = target_list + cv_list_2
        merge_anno_cols_list.append(merge_anno_cols_2)
        if set(ct_list_2) <= set(merge_anno_cols_2):
            flag = True
        else:
            flag = False
    return flag, merge_anno_cols_list

def check_condition_2(dataset_list, cv_list, target_list):
    flag = True
    merge_anno_cols_list = []
    for dataset in dataset_list:
        if type(dataset.data.cols) != list:
            dataset.data.cols = dataset.data.cols.tolist()
        merge_anno_cols_1 = dataset.data.cols  # target names which match dataset.data.y
        merge_anno_cols_list.append(merge_anno_cols_1)
    for idx, sub_cv_list in enumerate(cv_list):  # diff tissue may have diff cov
        ct_list = target_list + sub_cv_list
        if set(ct_list) > set(merge_anno_cols_list[idx]):
            flag = False
    return flag, merge_anno_cols_list

def short_path(opt,mopt,situ=1,nt_suffix=None):
    nt = opt.nt if nt_suffix is None else  f'{opt.nt}{nt_suffix}'
    if not hasattr(mopt,'cv_list') or len(mopt.cv_list) == 0:
        cv = 'c0'
    else:
        cv = f'c{len(mopt.cv_list)}'
    if isinstance(mopt,list):
        tissue_name = '-'.join([mopt[i].tissue_name for i in range(len(mopt))])
        target_list = mopt[0].target_
        output_prefix = 'ck' + nt + tissue_name 
    else:
        tissue_name = mopt.tissue_name
        target_list = mopt.target_
        output_prefix = 'ck' + nt + tissue_name +'-'+ mopt.raw_graph_dirname
    
    EXP_PATH = output_prefix + '_s' + str(situ) + '_m' + str(opt.memp) + '_'  + \
            cv + '_'  + target_list + '/'
    if opt.from_s1 == 1:
        LOAD_EXP_PATH = output_prefix + '_s1_m' + str(opt.memp) + '_' + cv + '_' + target_list + '/'
    else:
        LOAD_EXP_PATH = output_prefix + '_s4_m' + str(opt.memp) + '_' + cv + '_' + target_list + '/'
    return EXP_PATH, LOAD_EXP_PATH

def make_path(situ,opt,mopt,save=False):
    EXP_PATH,LOAD_EXP_PATH = short_path(opt,mopt,situ)
    EXP_PATH = os.path.join(opt.project_name[:-3], EXP_PATH)
    LOAD_EXP_PATH = os.path.join(opt.project_name[:-3],LOAD_EXP_PATH)
    
    BEST_PATH = EXP_PATH + 'best_params.txt'
    LOAD_BEST_PATH = LOAD_EXP_PATH + 'best_params.txt'
    if save and not hasattr(opt,'talos_start'):
        if not os.path.exists(EXP_PATH):
            os.makedirs(EXP_PATH)
        if not os.path.exists(BEST_PATH):
            with open(BEST_PATH,'w') as f:
                f.write('seq \t Best params \t Best epoch \t Best val loss \t Best tr loss \t Best val auc \t Best tr auc \t Best val acc \t Best tr acc\n')
    if opt.whichbest == -1:
        PCA_PATH = EXP_PATH + 'PCA/'
    else:
        PCA_PATH = EXP_PATH + 'PCA/'+ str(opt.whichbest) + '/'
    return EXP_PATH, BEST_PATH, PCA_PATH, LOAD_EXP_PATH, LOAD_BEST_PATH

def tocpu(lst):
    return [x.cpu().detach() for x in lst]

def calc_confmat_from_res(res, opt, multi=False):
    y_pred_all = res['pred_y']  # no_sam by no_classes
    y_all = res['y_all']
    if not isinstance(y_all,list) :
        if hasattr(opt,'sequential'):
            y_all = np.vstack([y_all for _ in range(2)]) # for seqEmbedding   
        cm = confusion_matrix(y_all, y_pred_all.argmax(axis=1)) if opt.target ==0 else None
    else:
        cm = [confusion_matrix(y,py.argmax(axis=1)) for y,py in zip(y_all,y_pred_all) ] if opt.target ==0 else None
    return cm,y_pred_all,y_all

def get_params(opt,load_params):
    if hasattr(opt,'load_params_selector') and opt.load_params_selector == 'val_corr':
        pname = opt.load_params_selector
    else:    
        pname = 'val_loss'

    if pname == 'val_loss':
        loaded_bestparams = load_params.iloc[load_params[pname].argmin()]
    elif pname == 'val_corr':
        loaded_bestparams = load_params.iloc[load_params[pname].argmax()] # min_loss
    else: 
        loaded_bestparams = None

    return loaded_bestparams['params'], loaded_bestparams['seq']


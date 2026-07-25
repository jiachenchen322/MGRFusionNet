from captum.attr import IntegratedGradients
import torch
import torch_geometric
import numpy as np
if torch_geometric.__version__ < '2.0':
    from torch_geometric.data import DataLoader
else:
    from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from utils.utils import predicate_xopt
class Explainer:
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    def __init__(self, opt, model):
        self.model = model
        self.multiomics_fusion   = predicate_xopt(opt,'multimodal_data') 
        self.fusion_on_embedding = True #hasattr(opt,'vcdn_embedding') and opt.vcdn_embedding >0
        # assert hasattr(model,'explain_ref_x')

    def predict(self, data_lst, mopt_lst, extra_forward_args):
        # data_lst : list(Batch)
        # mopt_lst : list(InputClass)
        emb_ls = []
        res_ls = []
        loss = 0.0
        self.model.eval()
        for data, m in zip(data_lst,mopt_lst):
            model = m.model_
            data = data.to(self.device)
            # get extra arguments 
            if torch_geometric.__version__ < '2.4.0':
                extra_args = {k:getattr(data, k)
                        for k in extra_forward_args if k in data.keys}
            else:
                extra_args = {k:getattr(data, k)
                        for k in extra_forward_args if k in data.keys()}
            # call single model
            out = model(x=data.x, 
                        edge_index=data.edge_index, 
                        batch=data.batch,
                        edge_attr=data.edge_attr,
                        **extra_args)
            
            if len(mopt_lst) == 1:
                output = out[0]
                loss = F.nll_loss(out[0], data.y.squeeze(1), reduction='sum')
                break
            emb_ls.append(out[1]['emb_out'])
            res_ls.append(out[0])
        # integrated model output
        if len(mopt_lst) > 1:
            output, *_= (self.model(emb_ls) if self.fusion_on_embedding
                            else  self.model(res_ls))
            loss = F.nll_loss(output, data.y.squeeze(1), reduction='sum')
        return torch.exp(output), loss.cpu().detach().numpy()
        
    def integrated_gradients(
                            self,
                            clz,
                            data,
                            mopt,
                            n_steps=100, 
                            extra_forward_args=[],
                            ):
        # define the forward function 
        def forward_fn(mask, data, mopt, extra_forward_args):
            slice_start = 0
            for d in data: # modulate data with mask
                d = d.to(self.device)
                slice_end = slice_start + d.x.shape[0]
                d.x = (d.x.T* mask[slice_start:slice_end]).T
                slice_start = slice_end
            data= [next(iter(DataLoader([d], batch_size=1)))  for d in data]        
            return self.predict(data, mopt, extra_forward_args)[0]
          
        n_nodes_ls = [d.x.shape[0] for d in data]
        num_nodes = sum(n_nodes_ls)
        mask = torch.ones(num_nodes ).requires_grad_(True)
        mask = mask.to(self.device)

        if hasattr(self.model, 'explain_ref_x'):
            baseline = self.model.explain_ref_x[mopt[0].tissue_name].squeeze().to(self.device) if len(mopt) ==1 else \
                    torch.cat([self.model.explain_ref_x[m.tissue_name].squeeze() for m in mopt]).to(self.device)
        else:
            baseline = 0.0
       
        target_class = clz
        ig = IntegratedGradients(forward_fn,)
        additional_forward_args = (data, mopt, extra_forward_args)
        internal_batch_size = mask.shape[0]
        attributions = ig.attribute(
                                inputs=mask, 
                                target=target_class, 
                                additional_forward_args=additional_forward_args,
                                internal_batch_size=internal_batch_size,
                                n_steps=n_steps,
                                baselines= baseline,
                                    ).clone().detach().cpu()
       
        n_nodes_ls.insert(0,0)
        n_nodes_ls = np.cumsum(np.array(n_nodes_ls)) 
        node_mask = [attributions[s:e].numpy() for s,e in zip(n_nodes_ls[0:-1],n_nodes_ls[1:])]
        return node_mask

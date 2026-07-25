from torch_geometric.data import Data
from .graph_dataset import DomainGraphDataset

class MultiOmicsDataset(DomainGraphDataset):
    def __init__(self, opt, mopt, datasets, idx=None):
        self.mopt = mopt
        self.name = f'multiOmics-{len(datasets)}'
        self.datasets = datasets
        self.d_idx = idx
        ref_dataset = datasets[0], idx[0] if idx is not None else None
        super(MultiOmicsDataset,self).__init__(ref_dataset, opt, mopt.nclass, name=self.name, mopt=mopt)
    
    def get(self, index):
        idx = self.whole_set[index] 
        if self.d_idx is None:
            data = [self.datasets[i][idx][0] for i in range(len(self.datasets))]
        else:
            data = [ dt[d_idx[idx]][0] if d_idx is not None else dt[idx][0] 
                    for dt,d_idx in zip(self.datasets, self.d_idx) ]
        # samid,y = data[0].samid , data[0].y
        # for d in data[1:]:
        #     assert samid == d.samid and y == d.y
        data.append(idx) 
       
        all_data = data
        if self.mopt.support_attn == -1:
            for ds in self.datasets:
                edge_info = ds.kfolds[self.mopt.fold]['edge_info']
                data = ds[idx]
                datals = []
                if len(edge_info) == 2: #edge_index, edge_attr
                    for edge_idx, edge_att in zip(* edge_info):
                        d = Data(
                                x = data[0].x,
                                edge_index = edge_idx,
                                edge_attr  = edge_att,
                                )
                        datals.append(d)  
                all_data.extend(datals) 
        return all_data 

"""
Integrated Gradients analysis for MGRFusionNet.

This script estimates fused-model attributions for:
  - protein graph features at the node level
  - longitudinal inputs at the timepoint-by-feature level
"""
import sys, os, copy, argparse, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from tqdm import tqdm

from captum.attr import IntegratedGradients

import torch_geometric
if torch_geometric.__version__ < '2.0':
    from torch_geometric.data import DataLoader, Data
else:
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data

from model.construct_graph import read_data
from set_transformer.set_transformer import Encoder
from platformX.params import paramparser
from platformX.prepdata import create_models, fetch_datasets

# ─────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────
class LongitudinalLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=32, emb_dim=16):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=2, batch_first=True, dropout=0.2)
        self.project = nn.Sequential(nn.Linear(hidden_size, emb_dim), nn.ReLU())

    def forward(self, x, return_hidden=False):
        _, (h_n, _) = self.lstm(x)
        hidden = h_n[-1]
        emb    = self.project(hidden)
        if return_hidden:
            return emb, hidden
        return emb


class MultiModalNetLSTMAux(nn.Module):
    def __init__(self, prot_emb, input_size=7, hidden_size=32, emb_dim=16, nclass=2):
        super().__init__()
        self.prot_emb         = prot_emb
        self.long_lstm        = LongitudinalLSTM(input_size, hidden_size, emb_dim)
        self.encoder          = Encoder(num_layers=2, num_features=emb_dim,
                                        num_heads=1, dim_ff=64)
        self.classifier       = nn.Linear(emb_dim, nclass)
        self.aux_classifier_p = nn.Linear(emb_dim, nclass)
        self.aux_classifier_l = nn.Linear(emb_dim, nclass)
        self.warmup_head      = nn.Linear(hidden_size, nclass)

    def forward(self, batch, warmup=False):
        emb_p = self.prot_emb(batch.x, batch.edge_index,
                              batch.batch, batch.edge_attr, cov=None)
        emb_l, hidden_l = self.long_lstm(batch.long_seq, return_hidden=True)
        if warmup:
            return (F.log_softmax(self.warmup_head(hidden_l),       dim=-1),
                    F.log_softmax(self.aux_classifier_l(emb_l),     dim=-1))
        combined = torch.stack([emb_p, emb_l], dim=1)
        fused, _ = self.encoder(combined)
        z = fused[:, 0]
        return (F.log_softmax(self.classifier(z),            dim=-1),
                F.log_softmax(self.aux_classifier_p(emb_p), dim=-1),
                F.log_softmax(self.aux_classifier_l(emb_l), dim=-1))


# ─────────────────────────────────────────────
# Explainer
# ─────────────────────────────────────────────
class MultiModalExplainer:
    """Integrated Gradients explainer for the MGRFusionNet architecture.

    Computes per-gene (protein) and per-element (longitudinal) attributions
    through the *fused* classifier (Set Transformer CLS → Linear).
    """

    def __init__(self, model, device='cpu'):
        self.model  = model
        self.model.eval()
        self.device = device

    def _expand_graph_inputs(self, orig_x, edge_index, edge_attr, long_seq, n_graphs):
        """Repeat one graph sample into a valid PyG mini-batch of size n_graphs."""
        if n_graphs == 1:
            batch_idx = torch.zeros(orig_x.shape[0], dtype=torch.long,
                                    device=orig_x.device)
            return orig_x, edge_index, edge_attr, batch_idx, long_seq

        n_nodes = orig_x.shape[0]
        n_edges = edge_index.shape[1]

        batch_idx = torch.arange(n_graphs, device=orig_x.device).repeat_interleave(n_nodes)
        edge_offsets = (torch.arange(n_graphs, device=edge_index.device) * n_nodes)
        edge_offsets = edge_offsets.repeat_interleave(n_edges).unsqueeze(0)
        edge_index_rep = edge_index.repeat(1, n_graphs) + edge_offsets

        if edge_attr is None:
            edge_attr_rep = None
        else:
            edge_attr_rep = edge_attr.repeat(n_graphs, *([1] * (edge_attr.dim() - 1)))

        long_seq_rep = long_seq.repeat(n_graphs, 1, 1)
        return None, edge_index_rep, edge_attr_rep, batch_idx, long_seq_rep

    def _canonicalize_inputs(self, n_graphs, orig_x, edge_index, edge_attr, long_seq):
        """Undo Captum's tensor expansion of additional args back to one sample."""
        if edge_index.dim() > 2 and edge_index.shape[0] == n_graphs and n_graphs > 1:
            edge_index = edge_index[0]

        if n_graphs > 1 and orig_x.shape[0] % n_graphs == 0:
            n_nodes = orig_x.shape[0] // n_graphs
            orig_x = orig_x[:n_nodes]

        if long_seq.dim() == 3 and long_seq.shape[0] == n_graphs and n_graphs > 1:
            long_seq = long_seq[:1]

        if edge_attr is not None and edge_attr.dim() > 2 and edge_attr.shape[0] == n_graphs and n_graphs > 1:
            edge_attr = edge_attr[0]

        if edge_attr is not None and edge_attr.shape[0] > 0 and edge_attr.shape[0] % n_graphs == 0:
            base_edges = edge_attr.shape[0] // n_graphs
            if base_edges > 0 and n_graphs > 1:
                edge_attr = edge_attr[:base_edges]

        if edge_index.dim() == 2 and edge_index.shape[1] > 0 and edge_index.shape[1] % n_graphs == 0:
            base_edges = edge_index.shape[1] // n_graphs
            if base_edges > 0 and n_graphs > 1:
                edge_index = edge_index[:, :base_edges]

        return orig_x, edge_index, edge_attr, long_seq

    # ── forward wrappers (captum calls these repeatedly) ────────────

    def _forward_prot(self, node_mask, orig_x, edge_index, edge_attr,
                      batch_idx, long_seq):
        """Protein path: scale each gene-node's features by *node_mask*."""
        n_nodes = orig_x.shape[0]
        if node_mask.dim() == 1:
            node_mask = node_mask.view(-1, n_nodes)
        elif node_mask.dim() > 2:
            node_mask = node_mask.view(node_mask.shape[0], n_nodes)

        n_graphs = node_mask.shape[0]
        orig_x, edge_index, edge_attr, long_seq = self._canonicalize_inputs(
            n_graphs, orig_x, edge_index, edge_attr, long_seq
        )
        n_nodes = orig_x.shape[0]
        x = (orig_x.unsqueeze(0) * node_mask.unsqueeze(-1)).reshape(
            n_graphs * n_nodes, orig_x.shape[1]
        )
        _, edge_index_rep, edge_attr_rep, batch_idx_rep, long_seq_rep = (
            self._expand_graph_inputs(orig_x, edge_index, edge_attr, long_seq, n_graphs)
        )

        emb_p = self.model.prot_emb(x, edge_index_rep, batch_idx_rep,
                                     edge_attr_rep, cov=None)
        emb_l = self.model.long_lstm(long_seq_rep)
        combined = torch.stack([emb_p, emb_l], dim=1)
        fused, _ = self.model.encoder(combined)
        return self.model.classifier(fused[:, 0])       # logits [B, 2]

    def _forward_long(self, long_mask, orig_x, edge_index, edge_attr,
                      batch_idx, long_seq):
        """Longitudinal path: scale each element of the [2,7] input."""
        if long_mask.dim() == 1:
            long_mask = long_mask.view(-1, 2, 7)
        elif long_mask.dim() == 2:
            if long_mask.shape == (2, 7):
                long_mask = long_mask.unsqueeze(0)
            else:
                long_mask = long_mask.view(-1, 2, 7)
        elif long_mask.dim() > 3:
            long_mask = long_mask.view(-1, 2, 7)

        n_graphs = long_mask.shape[0]
        orig_x, edge_index, edge_attr, long_seq = self._canonicalize_inputs(
            n_graphs, orig_x, edge_index, edge_attr, long_seq
        )
        long_masked = long_seq.repeat(n_graphs, 1, 1) * long_mask
        x_rep, edge_index_rep, edge_attr_rep, batch_idx_rep, _ = (
            self._expand_graph_inputs(orig_x, edge_index, edge_attr, long_seq, n_graphs)
        )
        if x_rep is None:
            x_rep = orig_x.repeat(n_graphs, 1)

        emb_p = self.model.prot_emb(x_rep, edge_index_rep, batch_idx_rep,
                                     edge_attr_rep, cov=None)
        emb_l = self.model.long_lstm(long_masked)
        combined = torch.stack([emb_p, emb_l], dim=1)
        fused, _ = self.model.encoder(combined)
        return self.model.classifier(fused[:, 0])       # logits [B, 2]

    # ── public API ──────────────────────────────────────────────────

    def explain(self, sample, n_steps=200):
        """Return (attr_prot, attr_long, prob, correct) for one sample.

        attr_prot : ndarray [n_genes]      – per-gene attribution
        attr_long : ndarray [2, 7]         – per-feature-per-timepoint
        prob      : ndarray [2]            – class probabilities
        correct   : bool
        """
        loader = DataLoader([sample], batch_size=1)
        batch  = next(iter(loader)).to(self.device)

        # check prediction
        with torch.no_grad():
            out = self.model(batch)[0]
            prob = torch.exp(out).squeeze().cpu().numpy()
            pred = out.argmax(dim=1).item()
        target = int(sample.y.item())

        if pred != target:
            return None, None, prob, False

        # tensors shared by both IG calls
        ig_prot = IntegratedGradients(
            lambda node_mask: self._forward_prot(
                node_mask, batch.x, batch.edge_index, batch.edge_attr,
                batch.batch, batch.long_seq
            )
        )
        ig_long = IntegratedGradients(
            lambda long_mask: self._forward_long(
                long_mask, batch.x, batch.edge_index, batch.edge_attr,
                batch.batch, batch.long_seq
            )
        )

        # ── protein IG ──
        mask_p = torch.ones((1, batch.x.shape[0]), device=self.device,
                            requires_grad=True)
        attr_p = ig_prot.attribute(
            inputs=mask_p, target=target,
            n_steps=n_steps, baselines=0.0,
        ).detach().cpu().numpy().squeeze(0)

        # ── longitudinal IG ──
        mask_l = torch.ones((1, 2, 7), device=self.device, requires_grad=True)
        attr_l = ig_long.attribute(
            inputs=mask_l, target=target,
            n_steps=n_steps, baselines=0.0,
        ).detach().cpu().numpy().squeeze(0)

        return attr_p, attr_l, prob, True


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_file', required=True)
    ap.add_argument('--data_raw', required=True)
    ap.add_argument('--splits_path', required=True)
    ap.add_argument('--long_path', required=True)
    ap.add_argument('--multimodal_ckpts', nargs='+', required=True,
                    help='One multimodal checkpoint or one per fold.')
    ap.add_argument('--folds',   nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--n_steps', type=int,  default=200,
                    help='interpolation steps for IG (more = more accurate)')
    ap.add_argument('--split',   choices=['val', 'train'], default='val',
                    help='which split to explain')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--support_attn', type=int, default=0)
    ap.add_argument('--target_col', default='labels')
    ap.add_argument('--lstm_hidden', type=int, default=64)
    ap.add_argument('--out_dir', default='results/interpretability')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if len(args.multimodal_ckpts) == 1:
        mm_ckpts = {fold: args.multimodal_ckpts[0] for fold in args.folds}
    elif len(args.multimodal_ckpts) == len(args.folds):
        mm_ckpts = {fold: ckpt for fold, ckpt in zip(args.folds, args.multimodal_ckpts)}
    else:
        raise ValueError('Provide either one multimodal checkpoint or exactly one checkpoint per fold.')

    device = args.device

    # ── load raw data (shared across folds) ──
    long_raw = np.load(args.long_path).astype(np.float32)

    print('Loading graph data …')
    data_list, edge_info = read_data(
        args.data_raw, node_th=0, target=0,
        anno_1_cols=[args.target_col], anno_2=None,
        convmethod='GCNConv', bin_edge=0,
        fold=1, support_attn=args.support_attn, no_mostrelated=0)

    edge_index = edge_info[0][0]
    edge_attr  = edge_info[1][0]

    sample_base = {}
    for d in data_list:
        sid   = int(d.samid)
        label = int(d.z[0, 0].item())
        sample_base[sid] = (d.x, label)

    # ── GNN architecture template ──
    print('Setting up GNN architecture template …')
    opt_tmpl, mopts_tmpl, _, _, _ = paramparser(choice='gnn',
                                                    cfg_file=args.cfg_file)
    for m in mopts_tmpl:
        m.l1_reg = 0.001; m.clusters = 80
        m.weightdecay = 0.0; m.batchsize = 4
    opt_tmpl.batchsize = 4
    fetch_datasets(opt_tmpl, mopts_tmpl)
    _, _tmp_models = create_models(copy.deepcopy(opt_tmpl),
                                    copy.deepcopy(mopts_tmpl), device)
    prot_emb_dim = _tmp_models[0].embedding.get_outdim()
    n_genes      = data_list[0].x.shape[0]
    print(f'Protein embedding dim: {prot_emb_dim}  |  genes: {n_genes}')
    del _tmp_models

    # gene names are only usable if they align with the attribution width.
    gene_names = None
    if hasattr(data_list[0], 'cols') and data_list[0].cols is not None:
        cols = data_list[0].cols
        if isinstance(cols, (list, np.ndarray)) and len(cols) == n_genes:
            gene_names = [str(c) for c in cols]
        elif isinstance(cols, (list, np.ndarray)):
            print(f'Gene-name metadata length mismatch: got {len(cols)}, expected {n_genes}; using gene indices instead.')

    with h5py.File(args.splits_path, 'r') as fh:
        all_splits = {fold: (fh[f'{fold}/tr_index'][()],
                             fh[f'{fold}/te_index'][()])
                      for fold in args.folds}

    # ── per-fold IG computation ──
    all_results = {}

    for fold in args.folds:
        print(f'\n{"="*60}')
        print(f'  FOLD {fold} — Integrated Gradients  (split={args.split})')
        print(f'{"="*60}')

        tr_idx, val_idx = all_splits[fold]
        target_idx = val_idx if args.split == 'val' else tr_idx

        # Apply the same fold-specific normalization used during model training.
        tr_mean  = np.nanmean(long_raw[tr_idx], axis=0)
        long_imp = long_raw.copy()
        for pi in range(7):
            for ti in range(2):
                mask = np.isnan(long_imp[:, pi, ti])
                long_imp[mask, pi, ti] = tr_mean[pi, ti]
        long_seq = long_imp.transpose(0, 2, 1).astype(np.float32)
        flat_tr  = long_seq[tr_idx].reshape(-1, 7)
        mu  = flat_tr.mean(0); std = flat_tr.std(0) + 1e-8
        long_seq = (long_seq - mu) / std

        # build PyG dataset
        ds = []
        for sid in target_idx:
            if sid not in sample_base:
                continue
            x, label = sample_base[sid]
            ls = torch.from_numpy(long_seq[sid]).unsqueeze(0)    # [1,2,7]
            ds.append(Data(x=x, edge_index=edge_index,
                           edge_attr=edge_attr,
                           y=torch.tensor(label, dtype=torch.long),
                           long_seq=ls, samid=str(sid)))
        print(f'  Samples to explain: {len(ds)}')

        # ── build & load model ──
        opt_g  = copy.deepcopy(opt_tmpl)
        mopt_g = copy.deepcopy(mopts_tmpl)
        _, gnn_models = create_models(opt_g, mopt_g, device)
        prot_emb = copy.deepcopy(gnn_models[0].embedding)
        mm_model = MultiModalNetLSTMAux(
            prot_emb, input_size=7, hidden_size=args.lstm_hidden,
            emb_dim=prot_emb_dim).to(device)

        mm_ckpt = mm_ckpts[fold]
        print(f'  Loading checkpoint: {mm_ckpt}')
        mm_model.load_state_dict(
            torch.load(mm_ckpt, map_location=device))
        mm_model.eval()

        explainer = MultiModalExplainer(mm_model, device=device)

        # ── run IG per sample ──
        prot_attrs = {0: [], 1: []}
        long_attrs = {0: [], 1: []}
        samids     = {0: [], 1: []}
        probs_all  = {0: [], 1: []}
        n_correct  = 0

        for sample in tqdm(ds, desc=f'  Fold {fold} IG'):
            attr_p, attr_l, prob, correct = explainer.explain(
                sample, n_steps=args.n_steps)
            if not correct:
                continue
            clz = int(sample.y.item())
            prot_attrs[clz].append(attr_p)
            long_attrs[clz].append(attr_l)
            samids[clz].append(sample.samid)
            probs_all[clz].append(prob)
            n_correct += 1

        print(f'  Correctly classified: {n_correct}/{len(ds)}')
        all_results[fold] = dict(prot_attrs=prot_attrs,
                                  long_attrs=long_attrs,
                                  samids=samids, probs=probs_all)

    # ─────────────────────────────────────────
    # Save HDF5
    # ─────────────────────────────────────────
    OUT_H5  = os.path.join(args.out_dir, f'ig_mgrfusionnet_{args.split}.h5')
    OUT_TXT = os.path.join(args.out_dir, f'ig_mgrfusionnet_{args.split}_summary.txt')
    dt = h5py.string_dtype(encoding='ascii')

    with h5py.File(OUT_H5, 'w') as f:
        f.attrs['generated'] = datetime.datetime.now().isoformat()
        f.attrs['n_steps']   = args.n_steps
        f.attrs['split']     = args.split
        f.attrs['n_genes']   = n_genes
        if gene_names is not None:
            f.create_dataset('gene_names', data=[g.encode('ascii', 'ignore')
                                                  for g in gene_names],
                             dtype=dt)
        for fold, res in all_results.items():
            grp = f.create_group(f'fold{fold}')
            for clz in [0, 1]:
                if len(res['prot_attrs'][clz]) == 0:
                    continue
                cgrp = grp.create_group(f'class{clz}')
                cgrp.create_dataset('prot_attrs',
                    data=np.vstack(res['prot_attrs'][clz]))
                cgrp.create_dataset('long_attrs',
                    data=np.stack(res['long_attrs'][clz]))
                cgrp.create_dataset('probs',
                    data=np.vstack(res['probs'][clz]))
                sids = res['samids'][clz]
                cgrp.create_dataset('samids', (len(sids),), dtype=dt,
                    data=[s.encode('ascii', 'ignore') for s in sids])
    print(f'\nHDF5 results  → {OUT_H5}')

    # ─────────────────────────────────────────
    # Text summary
    # ─────────────────────────────────────────
    lines = []
    def w(s=''):
        lines.append(s)

    w(f'Integrated Gradients — MGRFusionNet  ({args.split} set)')
    w(f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    w(f'n_steps={args.n_steps}  device={device}')
    w('=' * 60)

    class_names = {0: 'class_0', 1: 'class_1'}

    for fold, res in all_results.items():
        w(f'\n{"─"*60}')
        w(f'FOLD {fold}')
        w(f'{"─"*60}')

        for clz in [0, 1]:
            pa = res['prot_attrs'][clz]
            la = res['long_attrs'][clz]
            if len(pa) == 0:
                w(f'\n  Class {clz} ({class_names[clz]}): no correct predictions')
                continue

            pa = np.vstack(pa)          # [N, n_genes]
            la = np.stack(la)           # [N, 2, 7]

            w(f'\n  Class {clz} ({class_names[clz]})  N={pa.shape[0]}')

            # top-20 genes by mean |IG|
            mean_abs = np.abs(pa).mean(axis=0)
            top_idx  = np.argsort(mean_abs)[::-1][:20]
            w(f'\n  Top-20 genes by mean |IG|:')
            w(f'  {"Rank":>4}  {"Gene":>20}  {"mean|IG|":>10}  '
              f'{"std|IG|":>10}  {"mean(IG)":>10}')
            for rank, gi in enumerate(top_idx, 1):
                name = gene_names[gi] if gene_names else f'gene_{gi}'
                w(f'  {rank:>4}  {name:>20}  {mean_abs[gi]:>10.6f}  '
                  f'{np.abs(pa[:, gi]).std():>10.6f}  '
                  f'{pa[:, gi].mean():>10.6f}')

            # longitudinal attributions
            mean_l = np.abs(la).mean(axis=0)   # [2, 7]
            w(f'\n  Longitudinal mean |IG|  (rows=timepoints, cols=features):')
            header = '          ' + '  '.join(f'{"F"+str(j):>8}' for j in range(7))
            w(header)
            for t in range(2):
                row = f'  T{t}      ' + '  '.join(f'{mean_l[t,j]:>8.6f}'
                                                    for j in range(7))
                w(row)

    # aggregated across folds
    w(f'\n{"="*60}')
    w('AGGREGATED ACROSS FOLDS')
    w(f'{"="*60}')
    for clz in [0, 1]:
        all_pa = []
        all_la = []
        for res in all_results.values():
            all_pa.extend(res['prot_attrs'][clz])
            all_la.extend(res['long_attrs'][clz])
        if len(all_pa) == 0:
            continue
        pa = np.vstack(all_pa)
        la = np.stack(all_la)

        w(f'\n  Class {clz} ({class_names[clz]})  N={pa.shape[0]}')

        mean_abs = np.abs(pa).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:20]
        w(f'\n  Top-20 genes by mean |IG| (all folds pooled):')
        w(f'  {"Rank":>4}  {"Gene":>20}  {"mean|IG|":>10}  '
          f'{"std|IG|":>10}  {"mean(IG)":>10}')
        for rank, gi in enumerate(top_idx, 1):
            name = gene_names[gi] if gene_names else f'gene_{gi}'
            w(f'  {rank:>4}  {name:>20}  {mean_abs[gi]:>10.6f}  '
              f'{np.abs(pa[:, gi]).std():>10.6f}  '
              f'{pa[:, gi].mean():>10.6f}')

        mean_l = np.abs(la).mean(axis=0)
        w(f'\n  Longitudinal mean |IG|  (rows=timepoints, cols=features):')
        header = '          ' + '  '.join(f'{"F"+str(j):>8}' for j in range(7))
        w(header)
        for t in range(2):
            row = f'  T{t}      ' + '  '.join(f'{mean_l[t,j]:>8.6f}'
                                                for j in range(7))
            w(row)
        # marginal summaries
        per_T = mean_l.mean(axis=1)   # [2]  avg across features
        per_F = mean_l.mean(axis=0)   # [7]  avg across timepoints
        w(f'\n  Per-timepoint mean |IG|  (avg across features):')
        for t in range(2):
            w(f'    T{t}: {per_T[t]:.6f}')
        w(f'\n  Per-feature mean |IG|  (avg across timepoints):')
        for j in range(7):
            w(f'    F{j}: {per_F[j]:.6f}')

    txt = '\n'.join(lines)
    with open(OUT_TXT, 'w') as f:
        f.write(txt)
    print(f'Text summary  → {OUT_TXT}')
    print('\n' + txt)


if __name__ == '__main__':
    main()

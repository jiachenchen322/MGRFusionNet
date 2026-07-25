"""
Cross-modal interaction analysis for MGRFusionNet.

This script estimates second-order finite-difference interaction scores
between selected protein features and longitudinal phenotypes through
the fused classifier.
"""
import sys, os, copy, argparse, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from tqdm import tqdm

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

PHENO_NAMES = [f'feature_{i}' for i in range(7)]

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
# Interaction computation
# ─────────────────────────────────────────────
def forward_masked(model, x_obs, edge_index, edge_attr, long_obs,
                   prot_mask, long_mask, device):
    """Single forward pass with masked inputs. Returns P(class 1) scalar.

    prot_mask : [n_genes]  — 1.0 for observed, 0.0 for baseline
    long_mask : [7]        — 1.0 for observed, 0.0 for baseline (applied to both timepoints)
    """
    x = x_obs * prot_mask.unsqueeze(-1)                       # [n_genes, feat]
    long_in = long_obs * long_mask.unsqueeze(0).unsqueeze(0)  # [1, 2, 7]

    # build a single-sample PyG batch
    batch_idx = torch.zeros(x.shape[0], dtype=torch.long, device=device)

    emb_p = model.prot_emb(x, edge_index, batch_idx, edge_attr, cov=None)
    emb_l = model.long_lstm(long_in)
    combined = torch.stack([emb_p, emb_l], dim=1)
    fused, _ = model.encoder(combined)
    logits = model.classifier(fused[:, 0])
    prob = torch.softmax(logits, dim=-1)
    return prob[0, 1].item()   # P(class 1)


def compute_interactions(model, sample, prot_indices, pheno_indices, device):
    """Compute Delta_{j,k} for all (j,k) pairs for one sample.

    Returns ndarray [n_prot, n_pheno].
    """
    loader = DataLoader([sample], batch_size=1)
    batch  = next(iter(loader)).to(device)

    x_obs      = batch.x                # [n_genes, feat]
    edge_index = batch.edge_index
    edge_attr  = batch.edge_attr
    long_obs   = batch.long_seq          # [1, 2, 7]
    n_genes    = x_obs.shape[0]

    n_prot  = len(prot_indices)
    n_pheno = len(pheno_indices)

    # precompute: f(z^(0), X^(0))
    prot_zero = torch.zeros(n_genes, device=device)
    long_zero = torch.zeros(7, device=device)
    f_00 = forward_masked(model, x_obs, edge_index, edge_attr, long_obs,
                          prot_zero, long_zero, device)

    # precompute: f(z^(0), X^[k]) for each k
    f_0k = np.zeros(n_pheno)
    for ki, k in enumerate(pheno_indices):
        lm = torch.zeros(7, device=device)
        lm[k] = 1.0
        f_0k[ki] = forward_masked(model, x_obs, edge_index, edge_attr, long_obs,
                                  prot_zero, lm, device)

    # precompute: f(z^[j], X^(0)) for each j
    f_j0 = np.zeros(n_prot)
    for ji, j in enumerate(prot_indices):
        pm = torch.zeros(n_genes, device=device)
        pm[j] = 1.0
        f_j0[ji] = forward_masked(model, x_obs, edge_index, edge_attr, long_obs,
                                  pm, long_zero, device)

    # compute: f(z^[j], X^[k]) for each (j, k)
    delta = np.zeros((n_prot, n_pheno))
    for ji, j in enumerate(prot_indices):
        pm = torch.zeros(n_genes, device=device)
        pm[j] = 1.0
        for ki, k in enumerate(pheno_indices):
            lm = torch.zeros(7, device=device)
            lm[k] = 1.0
            f_jk = forward_masked(model, x_obs, edge_index, edge_attr, long_obs,
                                  pm, lm, device)
            delta[ji, ki] = f_jk - f_j0[ji] - f_0k[ki] + f_00

    return delta


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_file', required=True)
    ap.add_argument('--data_raw', required=True)
    ap.add_argument('--splits_path', required=True)
    ap.add_argument('--long_path', required=True)
    ap.add_argument('--multimodal_ckpts', nargs='+', required=True)
    ap.add_argument('--ig_h5', required=True,
                    help='IG HDF5 output produced by explain_integrated_gradients.py')
    ap.add_argument('--folds',      nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--n_prot',     type=int, default=10)
    ap.add_argument('--n_pheno',    type=int, default=3)
    ap.add_argument('--out_prefix', default='results/interpretability/cross_modal_interaction')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--support_attn', type=int, default=0)
    ap.add_argument('--target_col', default='labels')
    ap.add_argument('--lstm_hidden', type=int, default=64)
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if len(args.multimodal_ckpts) == 1:
        mm_ckpts = {fold: args.multimodal_ckpts[0] for fold in args.folds}
    elif len(args.multimodal_ckpts) == len(args.folds):
        mm_ckpts = {fold: ckpt for fold, ckpt in zip(args.folds, args.multimodal_ckpts)}
    else:
        raise ValueError('Provide either one multimodal checkpoint or exactly one checkpoint per fold.')

    device = args.device

    # ── determine top proteins & phenotypes from IG results ──
    print('Reading IG results to determine top features …')
    prot_ig_list = []
    long_ig_list = []
    with h5py.File(args.ig_h5, 'r') as f:
        for key in sorted(f.keys()):
            if not key.startswith('fold'):
                continue
            if 'class1' in f[key]:
                prot_ig_list.append(f[key]['class1/prot_attrs'][()])
                long_ig_list.append(f[key]['class1/long_attrs'][()])

    prot_ig = np.abs(np.vstack(prot_ig_list)).mean(axis=0)       # [n_genes]
    long_ig = np.abs(np.concatenate(long_ig_list)).mean(axis=0)  # [2, 7]
    per_pheno_ig = long_ig.mean(axis=0)                          # [7]

    top_prot_idx  = np.argsort(prot_ig)[::-1][:args.n_prot].tolist()
    top_pheno_idx = np.argsort(per_pheno_ig)[::-1][:args.n_pheno].tolist()

    prot_names  = [f'gene_{idx}' for idx in top_prot_idx]
    pheno_names = [PHENO_NAMES[k] for k in top_pheno_idx]

    print(f'  Top-{args.n_prot} proteins:   {prot_names}')
    print(f'  Top-{args.n_pheno} phenotypes: {pheno_names}')

    # ── load raw data ──
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

    print('Setting up GNN architecture template …')
    opt_tmpl, mopts_tmpl, _, _, _ = paramparser(choice='gnn', cfg_file=args.cfg_file)
    for m in mopts_tmpl:
        m.l1_reg = 0.001; m.clusters = 80; m.weightdecay = 0.0; m.batchsize = 4
    opt_tmpl.batchsize = 4
    fetch_datasets(opt_tmpl, mopts_tmpl)
    _, _tmp_models = create_models(copy.deepcopy(opt_tmpl),
                                    copy.deepcopy(mopts_tmpl), device)
    prot_emb_dim = _tmp_models[0].embedding.get_outdim()
    del _tmp_models

    with h5py.File(args.splits_path, 'r') as fh:
        all_splits = {fold: (fh[f'{fold}/tr_index'][()],
                             fh[f'{fold}/te_index'][()])
                      for fold in args.folds}

    # ── per-fold computation ──
    all_deltas = []   # list of [n_prot, n_pheno] arrays

    for fold in args.folds:
        print(f'\n{"="*60}')
        print(f'  FOLD {fold} — Cross-modal interactions')
        print(f'{"="*60}')

        tr_idx, val_idx = all_splits[fold]

        # normalise longitudinal data
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

        # build class-1 val dataset
        ds = []
        for sid in val_idx:
            if sid not in sample_base:
                continue
            x, label = sample_base[sid]
            if label != 1:
                continue
            ls = torch.from_numpy(long_seq[sid]).unsqueeze(0)
            ds.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                           y=torch.tensor(label, dtype=torch.long),
                           long_seq=ls, samid=str(sid)))
        print(f'  Class-1 val samples: {len(ds)}')

        # build & load model
        opt_g  = copy.deepcopy(opt_tmpl)
        mopt_g = copy.deepcopy(mopts_tmpl)
        _, gnn_models = create_models(opt_g, mopt_g, device)
        prot_emb = copy.deepcopy(gnn_models[0].embedding)
        mm_model = MultiModalNetLSTMAux(
            prot_emb, input_size=7, hidden_size=args.lstm_hidden,
            emb_dim=prot_emb_dim).to(device)
        mm_model.load_state_dict(
            torch.load(mm_ckpts[fold], map_location=device))
        mm_model.eval()

        # check correct predictions & compute interactions
        fold_deltas = []
        n_correct = 0
        for sample in tqdm(ds, desc=f'  Fold {fold} interaction'):
            loader = DataLoader([sample], batch_size=1)
            batch  = next(iter(loader)).to(device)
            with torch.no_grad():
                out = mm_model(batch)[0]
                pred = out.argmax(dim=1).item()
            if pred != 1:
                continue
            n_correct += 1

            with torch.no_grad():
                delta = compute_interactions(
                    mm_model, sample, top_prot_idx, top_pheno_idx, device)
            fold_deltas.append(delta)

        print(f'  Correctly classified: {n_correct}/{len(ds)}')
        if len(fold_deltas) > 0:
            all_deltas.extend(fold_deltas)

    # ── aggregate ──
    delta_stack = np.stack(all_deltas)          # [N, n_prot, n_pheno]
    mean_delta  = delta_stack.mean(axis=0)      # [n_prot, n_pheno]
    std_delta   = delta_stack.std(axis=0)

    # ── save results ──
    out_h5  = args.out_prefix + '.h5'
    out_txt = args.out_prefix + '_summary.txt'

    with h5py.File(out_h5, 'w') as f:
        f.attrs['generated'] = datetime.datetime.now().isoformat()
        f.attrs['n_samples'] = len(all_deltas)
        dt = h5py.string_dtype(encoding='ascii')
        f.create_dataset('prot_names', data=[n.encode() for n in prot_names], dtype=dt)
        f.create_dataset('pheno_names', data=[n.encode() for n in pheno_names], dtype=dt)
        f.create_dataset('prot_indices', data=top_prot_idx)
        f.create_dataset('pheno_indices', data=top_pheno_idx)
        f.create_dataset('mean_delta', data=mean_delta)
        f.create_dataset('std_delta', data=std_delta)
        f.create_dataset('all_deltas', data=delta_stack)
    print(f'\nHDF5 → {out_h5}')

    # ── save CSV (10 x 3) ──
    import pandas as pd
    out_csv = args.out_prefix + '.csv'
    df = pd.DataFrame(mean_delta, index=prot_names, columns=pheno_names)
    df.index.name = 'Protein'
    df.to_csv(out_csv)
    print(f'CSV  → {out_csv}')

    # ── text summary ──
    lines = []
    def w(s=''):
        lines.append(s)

    w('Cross-modal Interaction Scores (Delta_{j,k})')
    w(f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    w(f'Folds: {args.folds}   N(class-1 correct): {len(all_deltas)}')
    w('=' * 70)

    # header
    col_w = max(len(n) for n in pheno_names) + 2
    header = ' ' * 12 + ''.join(f'{n:>{col_w}}' for n in pheno_names)
    w(f'\n  Mean Delta (deviation from additivity):')
    w(f'  {header}')
    w(f'  {"-" * len(header)}')
    for ji, pn in enumerate(prot_names):
        row = f'  {pn:>10}  ' + ''.join(f'{mean_delta[ji, ki]:>{col_w}.6f}'
                                         for ki in range(len(pheno_names)))
        w(row)

    w(f'\n  Std Delta:')
    w(f'  {header}')
    w(f'  {"-" * len(header)}')
    for ji, pn in enumerate(prot_names):
        row = f'  {pn:>10}  ' + ''.join(f'{std_delta[ji, ki]:>{col_w}.6f}'
                                         for ki in range(len(pheno_names)))
        w(row)

    # highlight strongest interactions
    w(f'\n  Top interactions by |mean Delta|:')
    flat = np.abs(mean_delta).flatten()
    rank_idx = np.argsort(flat)[::-1]
    w(f'  {"Rank":>4}  {"Protein":>10}  {"Phenotype":>{col_w}}  '
      f'{"mean_Delta":>12}  {"std_Delta":>12}')
    w(f'  {"-" * (4 + 2 + 10 + 2 + col_w + 2 + 12 + 2 + 12)}')
    for rank, idx in enumerate(rank_idx, 1):
        ji = idx // len(pheno_names)
        ki = idx % len(pheno_names)
        w(f'  {rank:>4}  {prot_names[ji]:>10}  {pheno_names[ki]:>{col_w}}  '
          f'{mean_delta[ji, ki]:>12.6f}  {std_delta[ji, ki]:>12.6f}')

    txt = '\n'.join(lines)
    with open(out_txt, 'w') as f:
        f.write(txt)
    print(f'Text → {out_txt}')
    print('\n' + txt)


if __name__ == '__main__':
    main()

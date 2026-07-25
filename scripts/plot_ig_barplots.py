"""
Plot Integrated Gradients bar plots from MGRFusionNet HDF5 results.

Produces a 3-panel figure:
  (a) Top-20 proteins by mean |IG|
  (b) Per-feature (phenotype) mean |IG|
  (c) Per-timepoint mean |IG|

Run:
    python scripts/plot_ig_barplots.py --h5 results/interpretability/ig_mgrfusionnet_val.h5
"""
import argparse
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'


def load_and_aggregate(h5_path, target_class=1):
    """Load HDF5, pool all folds for target_class only, return (prot_attrs, long_attrs, gene_names)."""
    prot_list = []
    long_list = []
    gene_names = None

    with h5py.File(h5_path, 'r') as f:
        if 'gene_names' in f:
            gene_names = [s.decode() if isinstance(s, bytes) else s
                          for s in f['gene_names'][()]]
        for key in sorted(f.keys()):
            if not key.startswith('fold'):
                continue
            grp = f[key]
            ckey = f'class{target_class}'
            if ckey not in grp:
                continue
            cgrp = grp[ckey]
            prot_list.append(cgrp['prot_attrs'][()])
            long_list.append(cgrp['long_attrs'][()])

    prot_all = np.vstack(prot_list)   # [N_total, n_genes]
    long_all = np.concatenate(long_list, axis=0)  # [N_total, 2, 7]
    n_genes = prot_all.shape[1]

    if gene_names is None or len(gene_names) != n_genes:
        gene_names = [f'gene_{i}' for i in range(n_genes)]

    return prot_all, long_all, gene_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='results/interpretability/ig_mgrfusionnet_val.h5')
    ap.add_argument('--top_k', type=int, default=20)
    ap.add_argument('--out', default='results/interpretability/ig_barplots.pdf')
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    prot_all, long_all, gene_names = load_and_aggregate(args.h5)

    # ── protein: top-k by mean |IG| ──
    mean_abs_prot = np.abs(prot_all).mean(axis=0)
    top_idx = np.argsort(mean_abs_prot)[::-1][:args.top_k]
    top_names = [gene_names[i] for i in top_idx]
    top_vals  = mean_abs_prot[top_idx]

    # ── longitudinal ──
    mean_abs_long = np.abs(long_all).mean(axis=0)  # [2, 7]
    per_F = mean_abs_long.mean(axis=0)  # [7]
    per_T = mean_abs_long.mean(axis=1)  # [2]

    feat_names = [f'Feature {i}' for i in range(len(per_F))]
    time_names = [f'Time {i}' for i in range(len(per_T))]

    # ── plot ──
    bar_color = '#B0C4DE'

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1, 0.7], wspace=0.35)

    # (a) Top-k proteins — rank 1 at top
    ax_a = fig.add_subplot(gs[0])
    y_pos = np.arange(len(top_names))
    ax_a.barh(y_pos, top_vals[::-1], color=bar_color, edgecolor='none')
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(top_names[::-1])
    ax_a.set_xlabel('|IG|')
    ax_a.set_title(f'(a) Top {args.top_k} proteins', loc='left', fontweight='bold')
    ax_a.set_xticklabels([])
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # (b) Phenotypes — sorted by |IG| descending, rank 1 at top
    ax_b = fig.add_subplot(gs[1])
    f_order = np.argsort(per_F)          # ascending
    y_pos_f = np.arange(len(feat_names))
    ax_b.barh(y_pos_f, per_F[f_order], color=bar_color, edgecolor='none')
    ax_b.set_yticks(y_pos_f)
    ax_b.set_yticklabels([feat_names[i] for i in f_order])
    ax_b.set_xlabel('|IG|')
    ax_b.set_title('(b) Phenotypes', loc='left', fontweight='bold')
    ax_b.set_xticklabels([])
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # (c) Time points — sorted by |IG| descending, rank 1 at top
    ax_c = fig.add_subplot(gs[2])
    t_order = np.argsort(per_T)          # ascending
    y_pos_t = np.arange(len(time_names))
    ax_c.barh(y_pos_t, per_T[t_order], color=bar_color, edgecolor='none')
    ax_c.set_yticks(y_pos_t)
    ax_c.set_yticklabels([time_names[i] for i in t_order])
    ax_c.set_xlabel('|IG|')
    ax_c.set_title('(c) Time points', loc='left', fontweight='bold')
    ax_c.set_xticklabels([])
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    plt.savefig(args.out, bbox_inches='tight', dpi=600)
    print(f'Saved → {args.out}')
    plt.close()


if __name__ == '__main__':
    main()

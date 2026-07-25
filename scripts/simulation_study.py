"""
Simulation study for MGRFusionNet
==================================

Generates synthetic multimodal data (graph + longitudinal sequence) from a
shared latent structure and compares:
  1. Fusion model  (GCN + LSTM + Set Transformer)
  2. GCN-only      (graph modality alone)
  3. LSTM-only      (longitudinal modality alone)

Two settings:
  - low-dim:  N=1500, P=100,  60/20/20 split
  - high-dim: N=1024, P=2000, 80/10/10 split

Run:
    python scripts/simulation_study.py
"""
import sys, os, copy, argparse, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch_geometric
if torch_geometric.__version__ < '2.0':
    from torch_geometric.data import DataLoader, Data
else:
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from set_transformer.set_transformer import Encoder

# ─────────────────────────────────────────────
# 1.  Synthetic data generation
# ─────────────────────────────────────────────

def generate_synthetic_data(N, P, V=7, T=10, G=20, d_g=5, d_r=5, d_s=3,
                            K_edges=20, sigma_noise=0.3, seed=42):
    """Generate synthetic multimodal data following the simulation design.

    Returns
    -------
    graph_data : list[Data]   – one PyG Data per sample (x, edge_index, edge_attr, y, long_seq)
    info       : dict         – ground-truth latent variables for inspection
    """
    rng = np.random.RandomState(seed)

    # ── latent vectors ──
    z_g = rng.randn(N, d_g).astype(np.float32)   # molecular-specific
    z_r = rng.randn(N, d_r).astype(np.float32)   # sequence-specific
    z_s = rng.randn(N, d_s).astype(np.float32)   # shared / systemic

    # ── graph topology from pathway memberships ──
    # M: [P, G] binary membership, each protein in 2-4 pathways
    M = np.zeros((P, G), dtype=np.float32)
    for p in range(P):
        n_path = rng.randint(2, 5)  # 2, 3, or 4
        pathways = rng.choice(G, size=n_path, replace=False)
        M[p, pathways] = 1.0

    # adjacency: connect if shared >= 2 pathways
    shared = M @ M.T                             # [P, P]
    np.fill_diagonal(shared, 0)
    adj_full = (shared >= 2).astype(np.float32)

    # sparsify: keep top-K edges per node
    adj_sparse = np.zeros_like(adj_full)
    for p in range(P):
        neighbors = np.where(adj_full[p] > 0)[0]
        if len(neighbors) <= K_edges:
            adj_sparse[p, neighbors] = 1.0
        else:
            scores = shared[p, neighbors]
            top_k = neighbors[np.argsort(scores)[::-1][:K_edges]]
            adj_sparse[p, top_k] = 1.0
    # symmetrise
    adj_sparse = np.maximum(adj_sparse, adj_sparse.T)
    edge_src, edge_dst = np.where(adj_sparse > 0)
    edge_index = torch.tensor(np.stack([edge_src, edge_dst]), dtype=torch.long)
    edge_attr  = torch.ones(edge_index.shape[1], 1, dtype=torch.float32)

    # ── graph-side node features ──
    # B: [d_s+d_g, G]
    B = rng.randn(d_s + d_g, G).astype(np.float32) * 0.4
    z_graph = np.concatenate([z_s, z_g], axis=1)     # [N, d_s+d_g]
    U = np.tanh(z_graph @ B)                          # [N, G] pathway activity
    X = U @ M.T + rng.randn(N, P).astype(np.float32) * sigma_noise  # [N, P]

    # ── sequence-side longitudinal features ──
    C = rng.randn(d_s + d_r, V).astype(np.float32) * 0.3
    z_seq = np.concatenate([z_s, z_r], axis=1)        # [N, d_s+d_r]
    A_amp = np.maximum(z_seq @ C, 0)                  # [N, V]  ReLU

    t_grid = np.arange(T, dtype=np.float32) / (T - 1)  # [T]
    wave   = t_grid + 0.3 * t_grid ** 2                 # [T]

    S = np.zeros((N, T, V), dtype=np.float32)
    for n in range(N):
        for k in range(T):
            S[n, k, :] = A_amp[n] * wave[k] + rng.randn(V).astype(np.float32) * 0.25

    # ── binary outcome ──
    # graph-side risk
    strong_paths = [2, 7, 14, 18]
    weak_paths   = [3, 6, 11, 15, 19]
    # clamp to valid range
    strong_paths = [p for p in strong_paths if p < G]
    weak_paths   = [p for p in weak_paths if p < G]

    eta_g = 1.5 * U[:, strong_paths].sum(axis=1) + 0.8 * U[:, weak_paths].sum(axis=1)

    # sequence-side risk
    omega = np.arange(1, T + 1, dtype=np.float32)
    omega = omega / omega.sum()                         # [T]
    s_bar = np.einsum('k,nkv->nv', omega, S)            # [N, V]

    w_seq = rng.randn(V).astype(np.float32) * 0.8
    eta_r = np.tanh(s_bar @ w_seq)                      # [N]

    # standardise
    eta_g_std = (eta_g - eta_g.mean()) / (eta_g.std() + 1e-8)
    eta_r_std = (eta_r - eta_r.mean()) / (eta_r.std() + 1e-8)

    # mixing weight
    alpha = rng.beta(2, 2, size=N).astype(np.float32)
    eps_y = rng.randn(N).astype(np.float32) * 0.4

    eta = (1 - alpha) * eta_g_std + alpha * eta_r_std + eps_y
    pi  = 1.0 / (1.0 + np.exp(-eta))

    # threshold at 0.7 quantile
    q70 = np.quantile(pi, 0.7)
    y   = (pi > q70).astype(np.int64)

    print(f'  Data: N={N}, P={P}, V={V}, T={T}, G={G}')
    print(f'  Edges: {edge_index.shape[1]}  |  Class balance: '
          f'{(y==0).sum()} neg / {(y==1).sum()} pos')

    # ── pack into PyG Data objects ──
    data_list = []
    for n in range(N):
        x_n = torch.tensor(X[n], dtype=torch.float32).unsqueeze(-1)  # [P, 1]
        seq_n = torch.tensor(S[n], dtype=torch.float32).unsqueeze(0) # [1, T, V]
        data_list.append(Data(
            x=x_n,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(y[n], dtype=torch.long),
            long_seq=seq_n,
        ))

    info = dict(M=M, B=B, C=C, U=U, X=X, S=S, y=y, pi=pi,
                eta_g=eta_g, eta_r=eta_r, alpha=alpha, w_seq=w_seq,
                edge_index=edge_index)

    return data_list, info


def split_data(data_list, ratios, seed=42):
    """Split data_list into subsets according to ratios."""
    rng = np.random.RandomState(seed)
    N = len(data_list)
    idx = rng.permutation(N)
    cuts = np.cumsum([int(r * N) for r in ratios[:-1]])
    splits = np.split(idx, cuts)
    return [[data_list[i] for i in s] for s in splits]


# ─────────────────────────────────────────────
# 2.  Model definitions
# ─────────────────────────────────────────────

class GCNEncoder(nn.Module):
    """Two-layer GCN → global pool → projection."""
    def __init__(self, in_dim, hidden_dim=64, emb_dim=32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.project = nn.Sequential(
            nn.Linear(hidden_dim * 2, emb_dim),   # max + mean pool
            nn.ReLU(),
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = torch.cat([global_max_pool(h, batch),
                       global_mean_pool(h, batch)], dim=1)
        return self.project(h)


class SeqLSTM(nn.Module):
    """LSTM encoder for longitudinal sequences."""
    def __init__(self, input_size=7, hidden_size=64, emb_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, dropout=0.2)
        self.project = nn.Sequential(nn.Linear(hidden_size, emb_dim), nn.ReLU())

    def forward(self, x):
        # x: [B, T, V]
        _, (h_n, _) = self.lstm(x)
        return self.project(h_n[-1])


class FusionModel(nn.Module):
    """GCN + LSTM + Set Transformer fusion."""
    def __init__(self, in_dim, V=7, gcn_hidden=64, lstm_hidden=64,
                 emb_dim=32, nclass=2):
        super().__init__()
        self.gcn  = GCNEncoder(in_dim, gcn_hidden, emb_dim)
        self.lstm = SeqLSTM(V, lstm_hidden, emb_dim)
        self.encoder = Encoder(num_layers=2, num_features=emb_dim,
                               num_heads=1, dim_ff=64)
        self.classifier = nn.Linear(emb_dim, nclass)

    def forward(self, batch):
        emb_g = self.gcn(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        emb_r = self.lstm(batch.long_seq)
        combined = torch.stack([emb_g, emb_r], dim=1)   # [B, 2, emb]
        fused, _ = self.encoder(combined)
        z = fused[:, 0]                                  # CLS token
        return F.log_softmax(self.classifier(z), dim=-1)


class GCNOnlyModel(nn.Module):
    """GCN-only ablation."""
    def __init__(self, in_dim, gcn_hidden=64, emb_dim=32, nclass=2):
        super().__init__()
        self.gcn = GCNEncoder(in_dim, gcn_hidden, emb_dim)
        self.classifier = nn.Linear(emb_dim, nclass)

    def forward(self, batch):
        emb = self.gcn(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        return F.log_softmax(self.classifier(emb), dim=-1)


class LSTMOnlyModel(nn.Module):
    """LSTM-only ablation."""
    def __init__(self, V=7, lstm_hidden=64, emb_dim=32, nclass=2):
        super().__init__()
        self.lstm = SeqLSTM(V, lstm_hidden, emb_dim)
        self.classifier = nn.Linear(emb_dim, nclass)

    def forward(self, batch):
        emb = self.lstm(batch.long_seq)
        return F.log_softmax(self.classifier(emb), dim=-1)


# ─────────────────────────────────────────────
# 3.  Training & evaluation
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, class_weight, device):
    model.train()
    total_loss = 0.0
    w = torch.tensor(class_weight, dtype=torch.float32, device=device)
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = F.nll_loss(out, batch.y, weight=w)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        probs = torch.exp(out)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch.y.cpu().numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = (probs > 0.5).astype(int)
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return auc, acc, f1, probs, labels


def run_experiment(setting_name, N, P, split_ratios, V=7, in_dim=1,
                   n_epochs=80, lr=1e-3, batch_size=32, device='cpu',
                   n_repeats=5, base_seed=42):
    """Train Fusion, GCN-only, LSTM-only and return results."""

    results = {name: {'auc': [], 'acc': [], 'f1': []}
               for name in ['Fusion', 'GCN-only', 'LSTM-only']}

    for rep in range(n_repeats):
        print(f'\n  ── Repeat {rep+1}/{n_repeats} ──')

        torch.manual_seed(base_seed + rep)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed + rep)

        # generate fresh data each repeat
        data_list, info = generate_synthetic_data(N=N, P=P, seed=base_seed + rep)

        all_y = np.array([d.y.item() for d in data_list])
        n_neg, n_pos = (all_y == 0).sum(), (all_y == 1).sum()
        cw = [1.0, n_neg / max(n_pos, 1)]
        print(f'  Class weight: [1.0, {cw[1]:.2f}]')

        train_set, val_set, test_set = split_data(data_list, split_ratios,
                                                   seed=base_seed + rep)
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

        models = {
            'Fusion':    FusionModel(in_dim, V=V).to(device),
            'GCN-only':  GCNOnlyModel(in_dim).to(device),
            'LSTM-only': LSTMOnlyModel(V=V).to(device),
        }

        for name, model in models.items():
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs)

            best_val_auc = 0.0
            best_state   = None

            for epoch in range(1, n_epochs + 1):
                tr_loss = train_epoch(model, train_loader, optimizer, cw, device)
                scheduler.step()
                val_auc, _, _, _, _ = evaluate(model, val_loader, device)
                if epoch % 10 == 0 or epoch == 1:
                    print(f'      {name:>10} ep {epoch:>3}  '
                          f'loss={tr_loss:.4f}  val_auc={val_auc:.4f}')
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = copy.deepcopy(model.state_dict())

            # evaluate best checkpoint on test
            model.load_state_dict(best_state)
            test_auc, test_acc, test_f1, _, _ = evaluate(model, test_loader, device)
            results[name]['auc'].append(test_auc)
            results[name]['acc'].append(test_acc)
            results[name]['f1'].append(test_f1)
            print(f'    {name:>10}  AUC={test_auc:.4f}  Acc={test_acc:.4f}  '
                  f'F1={test_f1:.4f}  (val_best={best_val_auc:.4f})')

    return results


# ─────────────────────────────────────────────
# 4.  Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', choices=['low', 'high', 'both'], default='both')
    ap.add_argument('--preset', choices=['quick', 'standard'], default='standard',
                    help='Use `quick` for a lightweight public example or `standard` for the manuscript-scale simulation.')
    ap.add_argument('--n_repeats', type=int, default=5)
    ap.add_argument('--n_epochs',  type=int, default=80)
    ap.add_argument('--seed',      type=int, default=42)
    ap.add_argument('--device',    default='cpu')
    ap.add_argument('--out',       default='results/simulation/simulation_results.txt')
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.preset == 'quick':
        if args.n_repeats == 5:
            args.n_repeats = 2
        if args.n_epochs == 80:
            args.n_epochs = 20

    settings = {}
    if args.preset == 'quick':
        if args.setting in ('low', 'both'):
            settings['low'] = dict(N=256, P=60, split=(0.6, 0.2, 0.2))
        if args.setting in ('high', 'both'):
            settings['high'] = dict(N=320, P=200, split=(0.8, 0.1, 0.1))
    else:
        if args.setting in ('low', 'both'):
            settings['low'] = dict(N=1500, P=100, split=(0.6, 0.2, 0.2))
        if args.setting in ('high', 'both'):
            settings['high'] = dict(N=1024, P=2000, split=(0.8, 0.1, 0.1))

    all_results = {}

    for sname, cfg in settings.items():
        print(f'\n{"="*70}')
        print(f'  Setting: {sname}-dim   N={cfg["N"]}  P={cfg["P"]}')
        print(f'{"="*70}')

        res = run_experiment(
            setting_name=sname,
            N=cfg['N'], P=cfg['P'],
            split_ratios=cfg['split'],
            V=7, in_dim=1,
            n_epochs=args.n_epochs,
            batch_size=32,
            device=args.device,
            n_repeats=args.n_repeats,
            base_seed=args.seed,
        )
        all_results[sname] = res

    # ── write results ──
    lines = []
    def w(s=''):
        lines.append(s)

    w('Simulation Study Results — MGRFusionNet')
    w(f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    w(f'Preset: {args.preset}   Repeats: {args.n_repeats}   Epochs: {args.n_epochs}')
    w('=' * 70)

    for sname, res in all_results.items():
        cfg = settings[sname]
        w(f'\n{"─"*70}')
        w(f'Setting: {sname}-dim   N={cfg["N"]}  P={cfg["P"]}  '
          f'Split={cfg["split"]}')
        w(f'{"─"*70}')
        w(f'  {"Model":>10}  {"AUC":>14}  {"Acc":>14}  {"F1":>14}')
        w(f'  {"-"*56}')
        for name in ['Fusion', 'GCN-only', 'LSTM-only']:
            r = res[name]
            auc_m, auc_s = np.mean(r['auc']), np.std(r['auc'])
            acc_m, acc_s = np.mean(r['acc']), np.std(r['acc'])
            f1_m,  f1_s  = np.mean(r['f1']),  np.std(r['f1'])
            w(f'  {name:>10}  {auc_m:.4f}±{auc_s:.4f}  '
              f'{acc_m:.4f}±{acc_s:.4f}  {f1_m:.4f}±{f1_s:.4f}')

    txt = '\n'.join(lines)
    with open(args.out, 'w') as f:
        f.write(txt)
    print(f'\nResults → {args.out}')
    print('\n' + txt)


if __name__ == '__main__':
    main()

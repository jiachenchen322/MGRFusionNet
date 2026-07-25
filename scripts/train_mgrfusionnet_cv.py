"""
Train MGRFusionNet with fold-wise cross-validation.

Inputs required by this script:
1. A protein-encoder configuration compatible with `src/main_train.py`
2. Pretrained protein-encoder checkpoints, one per fold or one shared checkpoint
3. User-supplied graph and longitudinal data
"""
import sys, os, copy, datetime, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py

import torch_geometric
if torch_geometric.__version__ < '2.0':
    from torch_geometric.data import DataLoader, Data
else:
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data

from sklearn.metrics import (roc_auc_score, accuracy_score,
                              classification_report, confusion_matrix, roc_curve)

from model.construct_graph import read_data
from set_transformer.set_transformer import Encoder
from platformX.params import paramparser
from platformX.prepdata import create_models, fetch_datasets

ap = argparse.ArgumentParser()
ap.add_argument('--cfg_file', required=True,
                help='Protein-encoder config file used to rebuild the encoder architecture.')
ap.add_argument('--gnn_ckpts', nargs='+', required=True,
                help='Protein-encoder checkpoints. Provide either one checkpoint or one per fold.')
ap.add_argument('--data_raw', required=True,
                help='Path to the controlled-access graph raw directory expected by read_data().')
ap.add_argument('--splits_path', required=True,
                help='HDF5 file containing fold split indices.')
ap.add_argument('--long_path', required=True,
                help='NumPy file containing longitudinal data.')
ap.add_argument('--out_dir', default='results/mgrfusionnet_cv')
ap.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2])
ap.add_argument('--device', default='cpu')
ap.add_argument('--batch_size', type=int, default=4)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--weight_decay', type=float, default=0.0)
ap.add_argument('--l1_reg', type=float, default=1e-3)
ap.add_argument('--class_weight', type=float, default=3.7)
ap.add_argument('--aux_weight', type=float, default=0.5)
ap.add_argument('--lstm_hidden', type=int, default=64)
ap.add_argument('--support_attn', type=int, default=0)
ap.add_argument('--target_col', default='labels')
args = ap.parse_args()

FOLDS        = args.folds
BATCH_SIZE   = args.batch_size
LR           = args.lr
WEIGHT_DECAY = args.weight_decay
L1_REG       = args.l1_reg
DEVICE       = args.device
CLASS_WEIGHT = args.class_weight
AUX_WEIGHT   = args.aux_weight
LSTM_HIDDEN  = args.lstm_hidden

CFG_FILE    = args.cfg_file
DATA_RAW    = args.data_raw
SPLITS_PATH = args.splits_path
LONG_PATH   = args.long_path
OUT_DIR     = args.out_dir

os.makedirs(OUT_DIR, exist_ok=True)

if len(args.gnn_ckpts) == 1:
    GNN_CKPTS = {fold: args.gnn_ckpts[0] for fold in FOLDS}
elif len(args.gnn_ckpts) == len(FOLDS):
    GNN_CKPTS = {fold: ckpt for fold, ckpt in zip(FOLDS, args.gnn_ckpts)}
else:
    raise ValueError('Provide either one `--gnn_ckpts` path or exactly one checkpoint per fold.')

# ─────────────────────────────────────────────
# Load raw data (once, shared across folds)
# ─────────────────────────────────────────────
long_raw = np.load(LONG_PATH).astype(np.float32)

print('Loading graph data …')
data_list, edge_info = read_data(
    DATA_RAW, node_th=0, target=0,
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
opt_tmpl, mopts_tmpl, _, _, _ = paramparser(choice='gnn', cfg_file=CFG_FILE)
for m in mopts_tmpl:
    m.l1_reg = 0.001; m.clusters = 80; m.weightdecay = 0.0; m.batchsize = 4
opt_tmpl.batchsize = 4
fetch_datasets(opt_tmpl, mopts_tmpl)
_, _tmp_models = create_models(copy.deepcopy(opt_tmpl), copy.deepcopy(mopts_tmpl), DEVICE)
prot_emb_dim = _tmp_models[0].embedding.get_outdim()
print(f'Protein embedding dim: {prot_emb_dim}')
del _tmp_models

with h5py.File(SPLITS_PATH, 'r') as fh:
    all_splits = {fold: (fh[f'{fold}/tr_index'][()],
                         fh[f'{fold}/te_index'][()])
                  for fold in FOLDS}

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
        hidden = h_n[-1]                  # (batch, hidden_size)
        emb    = self.project(hidden)     # (batch, emb_dim)
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
        z = fused[:, 0]                # class token (Set Transformer CLS position)
        return (F.log_softmax(self.classifier(z),            dim=-1),
                F.log_softmax(self.aux_classifier_p(emb_p), dim=-1),
                F.log_softmax(self.aux_classifier_l(emb_l), dim=-1))


def make_run_epoch(model):
    """Create a training/evaluation closure for the staged optimization schedule."""
    def run_epoch(loader, optimizer, train=True, stage='full'):
        model.train() if train else model.eval()
        if not next(model.prot_emb.parameters()).requires_grad:
            model.prot_emb.eval()
        all_probs, all_labels = [], []
        total_loss = 0.0
        w = torch.tensor([1.0, CLASS_WEIGHT], device=DEVICE)
        with torch.set_grad_enabled(train):
            for batch in loader:
                batch = batch.to(DEVICE)

                if stage == '1a':
                    out_warmup, out_l = model(batch, warmup=True)
                    loss = (F.nll_loss(out_warmup, batch.y, weight=w)
                            + F.nll_loss(out_l,     batch.y, weight=w))
                    probs_for_auc = torch.exp(out_warmup)[:, 1]
                else:
                    out_main, out_p, out_l = model(batch)

                    if stage == '1b':
                        loss = (F.nll_loss(out_main, batch.y, weight=w)
                                + AUX_WEIGHT * F.nll_loss(out_p, batch.y, weight=w))
                        probs_for_auc = torch.exp(out_main)[:, 1]
                    else:  # full (Stage 2)
                        loss = (F.nll_loss(out_main, batch.y, weight=w)
                                + AUX_WEIGHT * (F.nll_loss(out_p, batch.y, weight=w)
                                              + F.nll_loss(out_l, batch.y, weight=w)))
                        probs_for_auc = torch.exp(out_main)[:, 1]

                if train and L1_REG > 0:
                    l1 = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
                    loss = loss + L1_REG * l1
                if train:
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item() * batch.num_graphs
                all_probs.append(probs_for_auc.detach().cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())
        probs  = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        auc    = roc_auc_score(labels, probs)
        preds  = (probs > 0.5).astype(int)
        acc    = accuracy_score(labels, preds)
        return auc, acc, total_loss / len(loader.dataset), probs, labels, preds
    return run_epoch


def print_header(title):
    print(f'\n── {title} ──')
    print(f'{"Epoch":>5}  {"trAUC":>6} {"trAcc":>6}  {"vAUC":>6} {"vAcc":>6}  {"loss":>7}')
    print('─' * 48)


# ─────────────────────────────────────────────
# 3-Fold CV loop
# ─────────────────────────────────────────────
fold_results = []

for fold in FOLDS:
    print(f'\n{"="*60}')
    print(f'  FOLD {fold}')
    print(f'{"="*60}')

    tr_idx, val_idx = all_splits[fold]

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

    def make_mm_dataset(idx):
        out = []
        for sid in idx:
            if sid not in sample_base: continue
            x, label = sample_base[sid]
            ls = torch.from_numpy(long_seq[sid]).unsqueeze(0)
            out.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=torch.tensor(label, dtype=torch.long),
                            long_seq=ls))
        return out

    mm_tr_set   = make_mm_dataset(tr_idx)
    mm_val_set  = make_mm_dataset(val_idx)
    print(f'  Train: {len(mm_tr_set)}  Val: {len(mm_val_set)}')

    mm_tr_loader   = DataLoader(mm_tr_set,   batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    mm_val_loader  = DataLoader(mm_val_set,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # ══════════════════════════════════════════
    # Load pre-trained GNN checkpoint
    # ══════════════════════════════════════════
    ckpt_path = GNN_CKPTS[fold]
    print(f'\n  Loading GNN checkpoint: {ckpt_path}')
    opt_g  = copy.deepcopy(opt_tmpl)
    mopt_g = copy.deepcopy(mopts_tmpl)
    _, gnn_models = create_models(opt_g, mopt_g, DEVICE)
    gnn = gnn_models[0]
    ck = torch.load(ckpt_path, map_location=DEVICE)
    gnn.load_state_dict(ck['model_state_dict'])
    print(f'  GNN fold {fold} loaded (epoch {ck.get("epoch", "?")})')

    # ══════════════════════════════════════════
    # Build multimodal model
    # ══════════════════════════════════════════
    prot_emb_copy = copy.deepcopy(gnn.embedding)
    mm_model = MultiModalNetLSTMAux(prot_emb_copy,
                                    input_size=7, hidden_size=LSTM_HIDDEN,
                                    emb_dim=prot_emb_dim).to(DEVICE)
    run_epoch = make_run_epoch(mm_model)

    best_vauc  = 0.0
    best_state = None

    # ──────────────────────────────────────────
    # Stage 1a: initialize the longitudinal branch before joint fusion training.
    # ──────────────────────────────────────────
    print_header(f'Fold {fold} | Stage 1a (LSTM warm-up, 25ep)')
    for p in mm_model.parameters():                  p.requires_grad = False
    for p in mm_model.long_lstm.parameters():        p.requires_grad = True
    for p in mm_model.aux_classifier_l.parameters(): p.requires_grad = True
    for p in mm_model.warmup_head.parameters():      p.requires_grad = True

    opt1a = torch.optim.Adam(filter(lambda p: p.requires_grad, mm_model.parameters()),
                              lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(1, 26):
        tr_auc, tr_acc, tr_loss, _, _, _ = run_epoch(mm_tr_loader,  opt1a, train=True,  stage='1a')
        va_auc, va_acc, _, _, _, _       = run_epoch(mm_val_loader, opt1a, train=False, stage='1a')
        if epoch % 5 == 0 or epoch == 1:
            print(f'{epoch:>5}  {tr_auc:>6.3f} {tr_acc:>6.3f}  '
                  f'{va_auc:>6.3f} {va_acc:>6.3f}  {tr_loss:>7.4f}')

    # ──────────────────────────────────────────
    # Stage 1b: fit the fusion layers while keeping unimodal encoders fixed.
    # ──────────────────────────────────────────
    print_header(f'Fold {fold} | Stage 1b (fusion learning, 15ep)')
    for p in mm_model.parameters():               p.requires_grad = False
    for p in mm_model.encoder.parameters():       p.requires_grad = True
    for p in mm_model.classifier.parameters():    p.requires_grad = True
    for p in mm_model.aux_classifier_p.parameters(): p.requires_grad = True

    opt1b = torch.optim.Adam(filter(lambda p: p.requires_grad, mm_model.parameters()),
                              lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(1, 16):
        tr_auc, tr_acc, tr_loss, _, _, _ = run_epoch(mm_tr_loader,  opt1b, train=True,  stage='1b')
        va_auc, va_acc, _, _, _, _       = run_epoch(mm_val_loader, opt1b, train=False, stage='1b')
        if va_auc > best_vauc:
            best_vauc  = va_auc
            best_state = copy.deepcopy(mm_model.state_dict())
        if epoch % 5 == 0 or epoch == 1:
            flag = ' *' if va_auc == best_vauc else ''
            print(f'{epoch:>5}  {tr_auc:>6.3f} {tr_acc:>6.3f}  '
                  f'{va_auc:>6.3f} {va_acc:>6.3f}  {tr_loss:>7.4f}{flag}')

    # ──────────────────────────────────────────
    # Stage 2: jointly fine-tune all components with differential learning rates.
    # ──────────────────────────────────────────
    print_header(f'Fold {fold} | Stage 2 (all unfrozen, diff-LR, 40ep)')
    for p in mm_model.parameters(): p.requires_grad = True
    opt2 = torch.optim.Adam([
        {'params': mm_model.prot_emb.parameters(),           'lr': LR * 0.01},   # 1e-5
        {'params': mm_model.long_lstm.parameters(),          'lr': LR * 0.3},    # 3e-4
        {'params': mm_model.encoder.parameters(),            'lr': LR * 0.3},    # 3e-4
        {'params': mm_model.classifier.parameters(),         'lr': LR * 0.3},    # 3e-4
        {'params': mm_model.aux_classifier_p.parameters(),   'lr': LR * 0.3},    # 3e-4
        {'params': mm_model.aux_classifier_l.parameters(),   'lr': LR * 0.3},    # 3e-4
    ], weight_decay=WEIGHT_DECAY)
    for epoch in range(1, 41):
        tr_auc, tr_acc, tr_loss, _, _, _ = run_epoch(mm_tr_loader,  opt2, train=True,  stage='full')
        va_auc, va_acc, _, _, _, _       = run_epoch(mm_val_loader, opt2, train=False, stage='full')
        if va_auc > best_vauc:
            best_vauc  = va_auc
            best_state = copy.deepcopy(mm_model.state_dict())
        if epoch % 5 == 0 or epoch == 1:
            flag = ' *' if va_auc == best_vauc else ''
            print(f'{epoch:>5}  {tr_auc:>6.3f} {tr_acc:>6.3f}  '
                  f'{va_auc:>6.3f} {va_acc:>6.3f}  {tr_loss:>7.4f}{flag}')

    # Evaluate best checkpoint
    save_path = os.path.join(OUT_DIR, f'best_mgrfusionnet_fold{fold}.pt')
    torch.save(best_state, save_path)
    mm_model.load_state_dict(best_state)
    f_tr_auc, f_tr_acc, _, tr_probs, tr_labels, tr_preds = run_epoch(mm_tr_loader,  opt2, train=False, stage='full')
    f_va_auc, f_va_acc, _, va_probs, va_labels, va_preds = run_epoch(mm_val_loader, opt2, train=False, stage='full')

    fpr, tpr, thresholds = roc_curve(va_labels, va_probs)
    best_thresh  = float(thresholds[np.argmax(tpr - fpr)])
    va_preds_opt = (va_probs >= best_thresh).astype(int)
    tr_preds_opt = (tr_probs >= best_thresh).astype(int)

    print(f'\n  Fold {fold} summary: MM best Val AUC={best_vauc:.4f}  Youden={best_thresh:.3f}')

    fold_results.append({
        'fold': fold,
        'best_vauc': best_vauc,
        'tr_auc': f_tr_auc, 'tr_acc': f_tr_acc,
        'va_auc': f_va_auc, 'va_acc': f_va_acc,
        'best_thresh': best_thresh,
        'tr_probs': tr_probs, 'tr_labels': tr_labels,
        'tr_preds': tr_preds, 'tr_preds_opt': tr_preds_opt,
        'va_probs': va_probs, 'va_labels': va_labels,
        'va_preds': va_preds, 'va_preds_opt': va_preds_opt,
    })

# ─────────────────────────────────────────────
# Aggregate across folds
# ─────────────────────────────────────────────
all_va_probs  = np.concatenate([r['va_probs']  for r in fold_results])
all_va_labels = np.concatenate([r['va_labels'] for r in fold_results])
agg_auc     = roc_auc_score(all_va_labels, all_va_probs)
agg_preds05 = (all_va_probs > 0.5).astype(int)
agg_acc05   = accuracy_score(all_va_labels, agg_preds05)
mean_vauc   = np.mean([r['best_vauc'] for r in fold_results])
std_vauc    = np.std( [r['best_vauc'] for r in fold_results])

fpr_agg, tpr_agg, thr_agg = roc_curve(all_va_labels, all_va_probs)
best_thresh_agg = float(thr_agg[np.argmax(tpr_agg - fpr_agg)])
agg_preds_opt   = (all_va_probs >= best_thresh_agg).astype(int)

print(f'\n{"="*60}')
print('Cross-Validation Summary')
print('  Per-fold MM  val AUC: ' +
      '  '.join(f"fold{r['fold']}={r['best_vauc']:.4f}" for r in fold_results))
print(f'  MM Mean ± Std: {mean_vauc:.4f} ± {std_vauc:.4f}')
print(f'  Aggregated Val AUC (all folds pooled): {agg_auc:.4f}')
print(f'  Aggregated Val Acc  (thr=0.5):         {agg_acc05:.4f}')
print(f'{"="*60}')

# ─────────────────────────────────────────────
# Save detailed results
# ─────────────────────────────────────────────
OUT_TXT = os.path.join(OUT_DIR, 'mgrfusionnet_cv_results.txt')
with open(OUT_TXT, 'w') as f:
    f.write('MGRFusionNet Cross-Validation Results\n')
    f.write(f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    f.write(f'BatchSize: {BATCH_SIZE}  LR: {LR}  L1: {L1_REG}  '
            f'ClassWeight: {CLASS_WEIGHT}  AuxWeight: {AUX_WEIGHT}\n')
    f.write(f'LSTM hidden_size: {LSTM_HIDDEN}  num_layers: 2  dropout: 0.2\n')
    f.write('Training schedule: Stage 1a warm-up, Stage 1b fusion learning, Stage 2 joint fine-tuning\n')
    f.write('Protein encoder checkpoints:\n')
    for fold in FOLDS:
        f.write(f'  fold{fold}={GNN_CKPTS[fold]}\n')
    f.write('=' * 60 + '\n\n')

    for r in fold_results:
        fold   = r['fold']
        thresh = r['best_thresh']
        f.write(f'{"─"*60}\n')
        f.write(f'FOLD {fold}  |  MM best val AUC: {r["best_vauc"]:.4f}\n')
        f.write(f'           Youden threshold: {thresh:.4f}\n\n')

        for split_name, labels, preds05, preds_opt, probs, auc in [
            ('TRAIN', r['tr_labels'], r['tr_preds'], r['tr_preds_opt'], r['tr_probs'], r['tr_auc']),
            ('VAL',   r['va_labels'], r['va_preds'], r['va_preds_opt'], r['va_probs'], r['va_auc']),
        ]:
            for thresh_name, preds in [('threshold=0.5', preds05),
                                        (f'threshold={thresh:.3f} (Youden)', preds_opt)]:
                acc = accuracy_score(labels, preds)
                f.write(f'  ── {split_name} [{thresh_name}] ──\n')
                f.write(f'  N={len(labels)}  AUC={auc:.4f}  Acc={acc:.4f}\n\n')
                f.write('  Confusion Matrix (rows=true, cols=pred):\n')
                f.write(f'    {confusion_matrix(labels, preds)}\n\n')
                f.write('  Classification Report:\n')
                report = classification_report(labels, preds,
                                               target_names=['class_0', 'class_1'], digits=4)
                for line in report.splitlines():
                    f.write(f'    {line}\n')
                f.write('\n')

        for split_name, labels, preds_opt, probs in [
            ('TRAIN', r['tr_labels'], r['tr_preds_opt'], r['tr_probs']),
            ('VAL',   r['va_labels'], r['va_preds_opt'], r['va_probs']),
        ]:
            f.write(f'  ── FOLD {fold} {split_name} per-sample (threshold={thresh:.3f}) ──\n')
            f.write(f'  {"idx":>6}  {"true":>5}  {"pred":>5}  {"prob":>8}\n')
            for i, (y, yh, p) in enumerate(zip(labels, preds_opt, probs)):
                f.write(f'  {i:>6}  {int(y):>5}  {int(yh):>5}  {p:>8.4f}\n')
            f.write('\n')

    f.write(f'{"="*60}\n')
    f.write('AGGREGATED (all 3 folds pooled)\n\n')
    f.write('Per-fold MM  val AUC: ' +
            '  '.join(f"fold{r['fold']}={r['best_vauc']:.4f}" for r in fold_results) + '\n')
    f.write(f'MM Mean ± Std Val AUC: {mean_vauc:.4f} ± {std_vauc:.4f}\n\n')

    for thresh_name, preds in [('threshold=0.5', agg_preds05),
                                (f'threshold={best_thresh_agg:.3f} (Youden)', agg_preds_opt)]:
        acc = accuracy_score(all_va_labels, preds)
        f.write(f'  ── ALL VAL [{thresh_name}] ──\n')
        f.write(f'  N={len(all_va_labels)}  AUC={agg_auc:.4f}  Acc={acc:.4f}\n\n')
        f.write('  Confusion Matrix:\n')
        f.write(f'    {confusion_matrix(all_va_labels, preds)}\n\n')
        f.write('  Classification Report:\n')
        report = classification_report(all_va_labels, preds,
                                       target_names=['class_0', 'class_1'], digits=4)
        for line in report.splitlines():
            f.write(f'    {line}\n')
        f.write('\n')

print(f'\nDetailed results → {OUT_TXT}')
print(f'Per-fold models  → {OUT_DIR}/best_mgrfusionnet_fold*.pt')

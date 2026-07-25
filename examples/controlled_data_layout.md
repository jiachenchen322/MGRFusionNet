# Controlled Data Layout

This repository does not distribute real-world study data. To run the real-data workflow, place controlled-access assets in a local directory of your choice and provide the corresponding paths explicitly to the scripts.

One possible layout is:

```text
controlled_data/
├── graph_raw/
│   ├── edge_info
│   ├── etc/
│   ├── all_samples_0.h5
│   ├── all_samples_targets.csv
│   └── splits-index.h5
├── longitudinal/
│   └── longitudinal.npy
├── metadata/
│   └── gene_header_with_symbols.csv
└── checkpoints/
    ├── protein_encoder_fold0.pt
    ├── protein_encoder_fold1.pt
    └── protein_encoder_fold2.pt
```

Notes:

- `graph_raw/` should match the expectations of `read_data()` in `src/model/construct_graph.py`.
- `splits-index.h5` should contain a `tr_index` and a `te_index` dataset per partition.
  `train_mgrfusionnet_cv.py` fits on `tr_index` and uses `te_index` to select the
  checkpoint and monitor training, so the indices it reads are a training/validation pair.
  The held-out test set on which the reported performance is measured is kept separate from
  this file and is not read by any script in this repository.
- The longitudinal NumPy tensor should align sample-wise with the split indices.
- The multimodal scripts do not assume any hardcoded dataset name, cohort label, or disease label.

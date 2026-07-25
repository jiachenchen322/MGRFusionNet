# MGRFusionNet

<p align="center">
  <img src="Fig1.png" alt="MGRFusionNet Architecture" width="800"/>
</p>

A multimodal learning framework that fuses graph-structured molecular measurements with longitudinal clinical trajectories for disease classification and interpretability analysis.

## Repository Structure

```text
MGRFusionNet/
├── configs/                  # Configuration templates
├── examples/                 # Data layout guides and runnable examples
├── scripts/
│   ├── train_mgrfusionnet_cv.py          # Multimodal fusion training
│   ├── explain_integrated_gradients.py   # Integrated Gradients interpretability
│   ├── cross_modal_interaction.py        # Cross-modal interaction scores
│   ├── plot_ig_barplots.py               # IG visualization
│   └── simulation_study.py               # Synthetic multimodal benchmark
└── src/
    ├── main_train.py         # Protein encoder training
    ├── model/                # GNN architecture and graph construction
    ├── platformX/            # Training library and multi-omics model
    ├── set_transformer/      # Set Transformer for modality fusion
    ├── explanation_code/     # Integrated Gradients for the protein encoder
    └── utils/                # Loss functions, metrics, and utilities
```

## Installation

Python 3.10+ recommended.

To run the simulation study only:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-sim.txt
```

To run the real-data workflow, install the full set instead:

```bash
pip install -r requirements.txt
```

`torch-scatter` and `torch-sparse` are needed only by the real-data workflow. If pip
cannot find a wheel for your torch build, see the
[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Quick Start (Simulation)

Run a self-contained synthetic benchmark without any real data:

```bash
bash examples/run_simulated_example.sh
```

This takes a few minutes on CPU and needs only `requirements-sim.txt`.
See [examples/simulated_example.md](examples/simulated_example.md) for details.

## Real-Data Workflow (Reference Implementation)

This section documents the model architecture and the training procedure as implemented.
It is a reference implementation, not a runnable reproduction: the controlled-access data
are not distributed here, and no pretrained weights are included, so the commands below
cannot be executed as given. See
[examples/controlled_data_layout.md](examples/controlled_data_layout.md) for the data
format each script expects.

The evaluation protocol used to obtain the performance figures in the paper — including
how the training, validation and held-out test partitions were constructed — is defined in
the paper's Methods section. These scripts do not implement held-out test-set evaluation,
and the summary statistics they print are training-time diagnostics rather than the
reported performance of the method.

### 1. Train the protein encoder

```bash
python3 src/main_train.py --config_path configs/protein_encoder_template.ini
```

### 2. Train the fusion model

```bash
python3 scripts/train_mgrfusionnet_cv.py \
  --cfg_file path/to/config.ini \
  --gnn_ckpts path/to/fold0.pt path/to/fold1.pt path/to/fold2.pt \
  --data_raw path/to/graph_raw \
  --splits_path path/to/splits.h5 \
  --long_path path/to/longitudinal.npy \
  --out_dir results/mgrfusionnet_cv
```

The script iterates over the partitions supplied in `--splits_path`, training one fusion
model per partition. Within each partition it uses the validation indices to select the
checkpoint and to monitor training; the held-out test set is not among the partitions it
reads. Any figures it prints are therefore diagnostics of the fitting procedure, and are
not the performance reported for the method.

### 3. Interpretability analysis

```bash
# Integrated Gradients
python3 scripts/explain_integrated_gradients.py \
  --cfg_file path/to/config.ini \
  --data_raw path/to/graph_raw \
  --splits_path path/to/splits.h5 \
  --long_path path/to/longitudinal.npy \
  --multimodal_ckpts results/mgrfusionnet_cv/best_mgrfusionnet_fold*.pt \
  --out_dir results/interpretability

# Cross-modal interaction scores
python3 scripts/cross_modal_interaction.py \
  --cfg_file path/to/config.ini \
  --data_raw path/to/graph_raw \
  --splits_path path/to/splits.h5 \
  --long_path path/to/longitudinal.npy \
  --multimodal_ckpts results/mgrfusionnet_cv/best_mgrfusionnet_fold*.pt \
  --ig_h5 results/interpretability/ig_mgrfusionnet_val.h5 \
  --out_prefix results/interpretability/cross_modal_interaction

# Visualization
python3 scripts/plot_ig_barplots.py \
  --h5 results/interpretability/ig_mgrfusionnet_val.h5 \
  --out results/interpretability/ig_barplots.pdf
```

## Data Availability

The real-data analysis depends on controlled-access human-subject data. No controlled data
or pretrained weights are distributed here. Running the real-data scripts requires the user
to supply graph raw data, partition-index files, longitudinal tensors, and protein-encoder
checkpoints compatible with this codebase.

The composition of the dataset, the construction of the training, validation and held-out
test partitions, and the performance obtained on the held-out test set are reported in the
paper. This repository does not reproduce that evaluation.

## Citation

If you use this repository in academic work, please cite our paper once bibliographic details are available.

## License

Released under the MIT License. See [LICENSE](LICENSE).

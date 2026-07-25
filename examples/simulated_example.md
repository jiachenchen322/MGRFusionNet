# Simulated Example

This repository includes a fully public simulation workflow that can be used as a runnable example without any controlled-access data.

## Evaluation Protocol

Each repeat draws a fresh synthetic dataset and partitions it disjointly into training,
validation and test sets in a 0.6 / 0.2 / 0.2 ratio. Models are fitted on the training set,
the checkpoint is selected by validation AUC, and the selected checkpoint is then evaluated
once on the held-out test set. The AUC, accuracy and F1 reported below are test-set figures;
the validation set is used only for model selection and never contributes to the reported
numbers. Results are averaged over repeats, each with its own data draw and seed, so the
reported spread reflects variability across independent datasets rather than across folds
of a single one.

## Recommended Quick Start

From the repository root:

```bash
bash examples/run_simulated_example.sh
```

Equivalent entry points:

```bash
./run_example.sh
```

```bash
make example
```

This command runs a lightweight low-dimensional simulation using a reduced problem size and shorter training schedule. It is intended as an installation and workflow check rather than as a manuscript-scale benchmark.

Expected output:

- `results/simulation/quickstart_results.txt`

The launcher script automatically uses `.venv/bin/python` when a repository-local virtual environment is available; otherwise it falls back to `python3`.

## Manual Command

```bash
python3 scripts/simulation_study.py \
  --preset quick \
  --setting low \
  --out results/simulation/quickstart_results.txt
```

## Full Simulation

To run the larger simulation settings used for the main study:

```bash
python3 scripts/simulation_study.py \
  --preset standard \
  --setting both \
  --out results/simulation/simulation_results.txt
```

To reproduce the standard low-dimensional example only:

```bash
python3 scripts/simulation_study.py \
  --preset standard \
  --setting low \
  --out results/simulation/standard_low_results.txt
```

## Bundled Reference Output

The canonical bundled reference output is:

- `examples/example_outputs/standard_low_results_example.txt`

Standard low-dimensional reference summary:

```text
Simulation Study Results — MGRFusionNet
Preset: standard   Repeats: 5   Epochs: 80

Setting: low-dim   N=1500  P=100  Split=(0.6, 0.2, 0.2)
       Model             AUC             Acc              F1
      Fusion  0.8163±0.0615  0.7367±0.0599  0.7120±0.0550
    GCN-only  0.6747±0.0924  0.6200±0.0285  0.5848±0.0362
   LSTM-only  0.7689±0.0644  0.6873±0.0652  0.6679±0.0586
```

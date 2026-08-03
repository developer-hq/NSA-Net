# NSA-Net

NSA-Net predicts whether two molecules can form a nano self-assembly under specified experimental conditions. The project combines molecular language-model features, molecular graphs, Mordred physicochemical descriptors, and experimental conditions. It provides a complete workflow from Excel data preparation to model training, result aggregation, and prediction.

Three model configurations are supported:

- `NSA-Net-C`: condition-aware model with experimental-condition modulation of cross-molecular structural interactions.
- `NSA-Net-S`: semantic-structural model without experimental-condition modulation.
- `NSA-Net-G`: graph-physicochemical model without the semantic branch or experimental-condition modulation.

## Project Structure

```text
NSA-Net/
|-- src/nsanet/
|   |-- config.py              # NSA-Net-C, NSA-Net-S, and NSA-Net-G configurations
|   |-- model.py               # NSA-Net model
|   |-- data.py                # Dataset and DataLoader
|   |-- graph_features.py      # Molecular graph and atom features
|   |-- mordred_features.py    # Mordred physicochemical descriptors
|   `-- metrics.py             # Evaluation metrics
|-- scripts/
|   |-- prepare_dataset.py     # Convert Excel data into model inputs
|   |-- precompute_mordred.py  # Compute Mordred descriptors
|   |-- train.py               # Train NSA-Net
|   |-- predict.py             # Run prediction from a checkpoint
|   |-- collect_results.py     # Aggregate multi-seed results
|   `-- build_grouped_benchmark50.py  # Build the grouped benchmark
|-- models/
|   |-- NSA-Net-C.pth          # Condition-aware checkpoint
|   `-- NSA-Net-S.pth          # Condition-free semantic-structural checkpoint
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.10 or later
- Linux
- CUDA-capable GPU recommended; CPU execution is also supported

Install the dependencies:

```bash
cd /home/srq/NSA-Net
pip install -r requirements.txt
```

The main dependencies are PyTorch, Transformers, RDKit, Mordred, NumPy, Pandas, scikit-learn, and openpyxl.

## Usage

### 1. Prepare the dataset

By default, the input Excel workbook contains a training sheet named `final_train` and a held-out sheet named `val`. Each sheet contains two molecular SMILES, molar ratios, concentration, pH, temperature, reaction time, and a binary label.

```bash
cd /home/srq/NSA-Net

python scripts/prepare_dataset.py \
  --input /path/to/benchmark.xlsx \
  --output /path/to/dataset \
  --train-sheet final_train \
  --val-sheet val \
  --label-col label
```

The processed splits are written to `/path/to/dataset/train/` and `/path/to/dataset/test/`.

### 2. Compute Mordred descriptors

```bash
python scripts/precompute_mordred.py \
  --dataset-root /path/to/dataset
```

### 3. Train the models

Train NSA-Net-C:

```bash
python scripts/train.py \
  --experiment NSA-Net-C \
  --dataset-root /path/to/dataset
```

Train NSA-Net-S:

```bash
python scripts/train.py \
  --experiment NSA-Net-S \
  --dataset-root /path/to/dataset
```

Train NSA-Net-G:

```bash
python scripts/train.py \
  --experiment NSA-Net-G \
  --dataset-root /path/to/dataset
```

Train all configurations:

```bash
python scripts/train.py \
  --experiment all \
  --dataset-root /path/to/dataset
```

The default training configuration uses five random seeds and 65 epochs. Per-seed results, checkpoints, and predictions are saved under `results/seeds/`, `results/models/`, and `results/predictions/`.

### 4. Aggregate results

```bash
python scripts/collect_results.py
```

The summary is written to `Experiment.csv` in the repository root.

### 5. Run prediction

Use the NSA-Net-C checkpoint:

```bash
python scripts/predict.py \
  --checkpoint models/NSA-Net-C.pth \
  --dataset-root /path/to/dataset/test \
  --output-dir /path/to/predictions
```

For NSA-Net-S, use `models/NSA-Net-S.pth`. An NSA-Net-G checkpoint can be used after it is placed in `models/` with an `NSA-Net-G` filename. Keep the sibling `train/` directory because prediction reconstructs normalization statistics from the training split.

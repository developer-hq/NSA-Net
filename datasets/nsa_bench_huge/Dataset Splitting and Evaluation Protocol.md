# Dataset Splitting and Evaluation Protocol

## 1. Task Definition

This document describes the binary classification training, validation, and testing workflow used for the current data files. The model takes two chemical molecules and their formulation or reaction conditions as input, produces a continuous prediction score, and converts that score into a binary prediction using a fixed threshold.

- Label column: `label`
- Positive class: `1`
- Negative class: `0`
- Binary classification threshold: `0.5`
- Model score: predicted positive-class probability or an equivalent continuous score
- Data encoding: `UTF-8 with BOM`; read using `utf-8-sig`

## 2. Data Sizes

The three current data files are listed below.

| Split | File | Rows | Positive samples | Positive rate |
|---|---|---:|---:|---:|
| Training set | `data/train400.csv` | 400 | 168 | 42.0% |
| Historical validation set eval | `data/eval50.csv` | 50 | 20 | 40.0% |
| Test set | `data/test50.csv` | 50 | 21 | 42.0% |

The training set is used to fit model parameters. The historical validation set eval is used to select the best model or best epoch. The test set is used only for the final benchmark and is not used for training, hyperparameter tuning, early stopping, or threshold selection.

## 3. Data Splitting Method

### 3.1 Training Set

The training set contains 400 rows, including 168 positive samples. The training data comes from supplementary open experimental data after label repair and preserves the original experimental conditions and source fields.

The training set is the only set that may be used to fit model parameters and determine data preprocessing parameters.

The preprocessing rules are as follows:

- Invalid numeric values are first converted to missing values.
- Missing-value medians are calculated using only the final training set.
- The validation and test sets use the fixed training-set medians and do not refit them.
- Any feature standardization, normalization, or statistical fitting must be performed only on the training set.
- The six condition columns use normalized values, and `chem_B_ratio` has already been divided by `chem_A_ratio`.

### 3.2 Historical Validation Set eval

The validation set is denoted eval and contains 50 rows, including 20 positive samples.

eval is a fixed historical validation set obtained from the `val` sheet of `benchmark.xlsx`. It was not randomly resampled within the current experimental workflow. This validation set is used to record historical results and serve as the model selection set, so it must be explicitly labeled as development-only.

The rules for using eval are as follows:

1. eval is fixed before the experiment and is not redivided based on model results.
2. eval is used only to select the best model or best epoch.
3. eval is not used for final performance reporting.
4. eval has previously been inspected and therefore must not be described as a fully independent blind test set.
5. The test set is not evaluated before model selection is complete.

### 3.3 Test Set Is a Cold-Start Set

The test set is a cold-start set. It contains 50 rows, 21 positive samples, and 29 unique molecule combinations.

The test set uses a cold-start setting, defined as follows:

1. Molecule combinations in the test set do not appear in the training or validation sets.
2. All experimental conditions for the same molecule combination must be placed in the same split.
3. Different concentrations, ratios, temperatures, pH values, or reaction times for the same molecule combination must not be divided across splits.
4. There is no molecule-level cold-start restriction, meaning that an individual molecule in the test set may already have appeared in the training or validation set.

For the current files:

- The test set contains 29 unique molecule combinations.
- The molecule-combination overlap between test and train is 0.
- The molecule-combination overlap between test and eval is 0.
- Based on exact matching of the original SMILES strings, 48 of the test rows contain at least one molecule that already appears in train or eval.
- Therefore, the current test set satisfies the cold-start setting.

The test set is fully frozen after the model is determined and is used only for the final benchmark. The test set must not be used for:

- model training;
- hyperparameter search;
- early stopping;
- threshold tuning;
- model selection;
- feature selection or data preprocessing fitting.

### 3.4 Splitting Procedure

The main settings of the current splitting protocol are as follows:

- Random seed: `3407`
- Data pool: the first 450 rows of the repaired source data
- The training and test sets are drawn from this data pool
- eval uses the fixed historical 50-row validation set
- Test-set selection preserves whole molecule-combination groups so that one molecule combination is not divided across splits
- Test-set selection maintains a positive-negative ratio close to that of the data pool
- Test-set selection does not use any model scores
- The last 50 rows of the repaired source data are not used for the current split in order to preserve the fixed historical eval50

## 4. Evaluation Method

### 4.1 Basic Metrics

After the model produces continuous scores, the following basic metrics are used:

- `AP`: Average Precision, used as the primary metric.
- `ROC-AUC`: used to measure ranking performance.
- `Accuracy`, `Precision`, `Recall`, and `F1`: auxiliary classification metrics at the threshold of `0.5`.

Evaluation is based on standard binary classification metrics and does not introduce unnecessary additional evaluation procedures.

### 4.2 Selecting the Best Model on eval

Each model configuration, random seed, or training epoch is evaluated on eval. The model selection rules are as follows:

1. First select the model or epoch with the highest eval `AP`.
2. If `AP` is tied, select the model or epoch with the higher eval `ROC-AUC`.
3. If the tie remains, select the earlier epoch.
4. After the best model is determined, its parameters and threshold are fully frozen.

This process uses only train and eval and never uses test.

### 4.3 Benchmarking on test

Only after the final model and threshold are determined is the final evaluation run on test. The test results are reported as the benchmark results.

Reporting rules are as follows:

- Report the frozen model's `AP` and `ROC-AUC` on test.
- Also report `Accuracy`, `Precision`, `Recall`, and `F1` at the threshold of `0.5`.
- If multiple random seeds are used, report the mean and standard deviation.
- Test results must not be used to reselect models, adjust thresholds, or modify preprocessing.

nsa_bench_huge will be released tomorrow.

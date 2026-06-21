# Data Directory Guide

This directory contains the dataset used for anomaly detection experiments.

Large datasets are not included in this repository and must be stored locally.

---

## Directory Structure

data/

├── raw/
│   ├── train/
│   ├── test/
│   ├── val/
│   └── val_labels/
│
└── processed/

---

## Raw Data

The `raw/` directory contains the original sensor data files.

raw/

├── train/       (53 runs)
├── test/        (28 runs)
├── val/         (10 runs)
└── val_labels/  (10 label files)

### Description

- train/: Training runs used for feature engineering and model training.
- test/: Unlabeled test runs used for final evaluation and submission generation.
- val/: Validation runs used for threshold optimization and model evaluation.
- val_labels/: Ground-truth anomaly labels corresponding to the validation runs.

---

## Processed Data

The `processed/` directory contains datasets generated from the raw sensor data.

Examples include:

- Selected sensor datasets
- Engineered feature datasets
- Model-ready feature matrices

These files are generated through the project pipeline and are not stored in GitHub.

---

## Reproducing Processed Data

1. Place the original dataset files inside the appropriate `raw/` subdirectories.
2. Run the feature engineering pipeline.
3. Run the sensor selection pipeline.
4. Execute model training notebooks or pipelines.

The required processed datasets will be generated automatically.
# Dataset Guide

This directory contains all data required for the time-series anomaly detection project.

The dataset consists of multivariate sensor time-series recordings used for training, validation, and testing of anomaly detection models.

---

## Overview

The data is organized into raw inputs and processed outputs generated during feature engineering.

- **Raw data**: Original sensor recordings (not modified)
- **Processed data**: Feature-engineered datasets used for model training
- **Artifacts**: Intermediate outputs generated during analysis and experimentation

---

## Directory Structure

```text
data/
├── raw/
│   ├── train/
│   ├── test/
│   ├── val/
│   └── val_labels/
│
├── processed/
└── artifacts/
```

---

## Raw Data

The `raw/` directory contains the original multivariate sensor time-series data.

```text
raw/
├── train/       (53 runs)
├── test/        (28 runs)
├── val/         (10 runs)
└── val_labels/  (10 files)
```

### Description

- **train/** → Used for model training and feature engineering  
- **test/** → Unlabeled data used for final evaluation  
- **val/** → Used for validation and threshold tuning  
- **val_labels/** → Ground-truth labels for validation set  

---



- Raw datasets are not tracked in GitHub due to size constraints.
- All datasets follow a reproducible pipeline-based workflow.
- Ensure correct folder structure before running scripts.
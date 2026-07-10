# Time-series anomaly detection

## Overview

This project detects anomalies in multivariate sensor time-series data from an industrial process, using two complementary approaches:

- **CNN autoencoder** — a dilated (TCN-style) 1D convolutional autoencoder that learns to reconstruct normal sensor windows. Anomaly score is a **Mahalanobis distance** over per-sensor reconstruction error (not a flat average), so a genuine deviation on one sensor isn't diluted by another sensor's ordinary noise.
- **Isolation Forest** — a statistical outlier detector, run both on tabular lag/rolling features (standalone baseline) and on per-window summary statistics (used inside the hybrid).
- **Hybrid fusion** — both models' scores are normalized and combined with a weight chosen on the validation set (the sweep can also select "pure CNN, no fusion" if that scores best — see Results).

There are no anomaly labels for training data — only a small labeled validation set and an externally-scored test set — so every model here learns what *normal* looks like and flags departures from it, rather than learning to classify anomalies directly.

**A full narrative writeup — data, EDA, model theory, and all four rounds of debugging with diagrams — is in [`reports/project_textbook.pdf`](reports/project_textbook.pdf).** This README is the quick-reference version.

---

## Repository structure

```text
ts-anomaly-detection/
├── data/
│   ├── raw/                  # train/val/val_labels/test CSV runs (not tracked in git)
│   ├── processed/            # sensor feature matrix + selected_sensors.json
│   ├── artifacts/            # exploratory analysis outputs (KDE groupings, etc.)
│   └── README.md
│
├── notebooks/
│   ├── 01_sensor_selection.ipynb
│   ├── 02_anomaly_detection_isolation_forest.ipynb   # tabular Isolation Forest baseline
│   ├── 03_cnn_baseline.ipynb                          # first CNN autoencoder pass
│   ├── 04_cnn_tuning.ipynb                            # window/stride/epoch experiments
│   ├── 05_hybrid_model.ipynb                          # thin runner for src/pipelines/run_hybrid.py
│   └── archive/               # earlier EDA/feature-engineering drafts, kept for reference
│
├── src/
│   ├── config.py               # all hyperparameters and paths in one place
│   ├── core/
│   │   ├── data_loader.py       # CSV loading/cleaning helpers (sensor-selection path)
│   │   ├── feature_pipeline.py  # lag/rolling/EWMA/interaction features for Isolation Forest
│   │   ├── feature_selection.py # sensor scoring + correlation filtering
│   │   └── windowing.py         # shared sliding-window utilities (both model branches)
│   ├── models/
│   │   ├── cnn/                 # dilated ConvAutoencoder1D, training loop (w/ checkpoint selection), scoring
│   │   └── isolation_forest/    # per-window features, training, scoring
│   ├── evaluation/
│   │   └── metrics.py           # F-beta threshold selection, Mahalanobis scoring, precision/recall/F1/ROC-AUC
│   └── pipelines/
│       ├── run_isolation_forest.py   # standalone tabular Isolation Forest baseline
│       ├── run_cnn.py                # standalone CNN autoencoder
│       └── run_hybrid.py             # CNN + window Isolation Forest, fused
│
├── scripts/
│   └── run_hybrid.slurm       # SLURM batch script for GPU training (RPTU Elwetritsch cluster)
│
├── outputs/
│   ├── isolation_forest/{metrics,predictions}/
│   ├── cnn/metrics.json, predictions.csv
│   ├── hybrid/metrics.json
│   └── hybrid_predictions.csv        # final run_id, timestep, prediction submission
│
├── reports/
│   ├── project_textbook.pdf         # full narrative: data, EDA, theory, all 4 rounds, diagrams
│   ├── project_summary_report.pdf   # condensed work-log summary
│   └── hybrid_model_postmortem.pdf  # root-cause analysis of why early rounds plateaued
│
├── plots/                     # figures referenced by the notebooks
├── HYBRID_MODEL_README.md     # architecture deep-dive for the hybrid model
├── requirements.txt
└── README.md
```

---

## Installation

```bash
python -m venv ts-anomaly-env
source ts-anomaly-env/bin/activate   # Windows: ts-anomaly-env\Scripts\activate
pip install -r requirements.txt
```

Training runs on CPU (no CUDA required); a GPU will be used automatically if available. `scripts/run_hybrid.slurm` is provided for running on a SLURM-managed GPU cluster.

---

## Dataset requirements

Place raw CSV runs under `data/raw/`:

```text
data/raw/
├── train/       (28 runs, unlabeled)
├── val/         (10 runs)
├── val_labels/  (10 files, per-timestep 0/1 labels)
└── test/        (53 runs, unlabeled — scored externally, e.g. Codabench)
```

Raw and processed data are not tracked in git due to size. See `data/README.md` for the full preprocessing/regeneration workflow.

---

## Training pipeline

Run from the repository root (each script reads paths/hyperparameters from `src/config.py`):

```bash
python -m src.pipelines.run_isolation_forest   # tabular baseline -> outputs/isolation_forest/
python -m src.pipelines.run_cnn                # CNN autoencoder  -> outputs/cnn/
python -m src.pipelines.run_hybrid             # CNN + window IF, fused -> outputs/hybrid/, outputs/hybrid_predictions.csv
```

All three load raw data, fit a `StandardScaler` on train only, build sliding windows **per source run** (windows never cross run boundaries), train, score, select a threshold, and write metrics + predictions.

The CNN trains for up to 50 epochs, checking validation every 5 epochs and keeping the best-scoring checkpoint (training loss falls smoothly with no plateau, so the last epoch is not always the best one). `run_hybrid.py` additionally sweeps the Isolation Forest `contamination` parameter and the CNN/ISO fusion weight (including a "pure CNN, no fusion" option) against validation.

Notebooks 01-04 document the exploratory path (sensor selection, feature engineering, CNN architecture/window-size tuning); notebook 05 simply calls `run_hybrid.py` and displays the result.

---

## Evaluation pipeline

`src/evaluation/metrics.py`:
- `find_best_threshold` sweeps `sklearn.metrics.precision_recall_curve` on the validation set and picks the threshold maximizing **F-beta** (`beta=0.5` by default — weights precision over recall, since plain F1 consistently over-predicted anomalies given how overlapped the score distributions are).
- `fit_mahalanobis` / `mahalanobis_scores` fit a Gaussian error model on confirmed-normal validation windows' **per-sensor** CNN reconstruction error, and score new windows by Mahalanobis distance — replacing a flat mean-squared-error that was diluting a genuine spike on one sensor with another sensor's ordinary noise.
- `compute_metrics` reports precision, recall, F1, ROC-AUC, and predicted anomaly rate at a given threshold.

The CNN and hybrid pipelines score validation/test at a dense stride (`config.EVAL_STEP`) and average overlapping window scores per raw timestep before thresholding, so predictions map one-to-one onto the actual per-timestep submission format.

**`data/raw/test/` has no local labels** (it's scored externally), so every metric below is a **validation-set** metric except where a real portal score is explicitly noted.

---

## Results

| Stage | CNN F1 | Isolation Forest F1 | Hybrid F1 |
|---|---|---|---|
| Original baseline (window-level, not directly comparable) | 0.60 | 0.40 | 0.62 |
| After bug fixes (per-timestep) | 0.408 | 0.491 | 0.494 |
| + CNN checkpoint selection | 0.458 | 0.491 | 0.494 |
| + Dilated CNN architecture | 0.484 | 0.491 | 0.494 |
| **+ Mahalanobis scoring + F-beta threshold** | **0.581** | 0.489 | **0.581** |

**Current best model** (validation): F1 = 0.581, precision = 0.601, recall = 0.562, ROC-AUC = 0.768, predicted anomaly rate = 0.224 (true rate ≈ 0.24). **Confirmed on the real competition portal: F1 = 0.59** — close to the validation estimate, confirming the evaluation methodology is trustworthy.

The fusion weight sweep currently selects **CNN weight = 1.0** (pure CNN, no Isolation Forest contribution) — after three rounds of CNN-only improvement left the fused hybrid F1 flat at 0.494, the Mahalanobis + F-beta fix finally moved the number that mattered, and the CNN alone now outperforms any blend with Isolation Forest. This is stated plainly rather than dressed up: right now, "the hybrid model" and "the CNN model" are the same model.

See `reports/project_textbook.pdf` for the full explanation of each fix and why earlier rounds (checkpoint selection, dilated architecture) improved the CNN without moving the hybrid score at all.

---

## Reproducing the experiments

```bash
source ts-anomaly-env/bin/activate
python -m src.pipelines.run_isolation_forest
python -m src.pipelines.run_cnn
python -m src.pipelines.run_hybrid
```

Random seeds are fixed (`config.RANDOM_SEED = 42`). CNN training takes roughly 30 minutes on CPU at 50 epochs, faster on GPU (`scripts/run_hybrid.slurm`).

---

## Future improvements

- **Give Isolation Forest a reason to rejoin the ensemble** — richer per-window features (cross-sensor interaction terms, frequency-domain features) so it can contribute again now that pure CNN outperforms any blend with its current features.
- **Contamination-robust training** — train isn't guaranteed anomaly-free; consider iteratively down-weighting the highest-error windows during training.
- **Learned fusion** — replace the fixed-grid weight sweep with a logistic regression over `[cnn_score, iso_score]`.
- **Held-out tuning split** — window size, stride, architecture, and threshold have all been selected by watching the same 10-run validation set; a proper k-fold or separate tuning split would give a more honest read on generalization before submission.

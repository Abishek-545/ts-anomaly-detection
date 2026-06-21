# Time Series Anomaly Detection

## Overview

This project focuses on anomaly detection in multivariate sensor time-series data using both traditional machine learning and deep learning approaches.

The objective is to identify abnormal system behavior from sensor measurements by combining engineered statistical features with temporal pattern learning.

**Application Domain:**
[Add project-specific domain information here, e.g., industrial process monitoring, chemical manufacturing systems, predictive maintenance, or sensor-based process control.]

The project currently includes:

* Sensor selection and feature selection workflow.
* Feature engineering pipeline for multivariate sensor data.
* Isolation Forest baseline anomaly detection model.
* CNN-based anomaly detection pipeline (in progress).
* Modular project structure for reproducible experimentation and evaluation.

---

## Project Structure

```text
ts-anomaly-detection/

├── data/
│   ├── raw/
│   ├── processed/
│   ├── artifacts/
│   └── README.md
│
├── notebooks/
│   ├── archive/
│   ├── 01_sensor_selection.ipynb
│   └── 02_anomaly_detection_isolation_forest.ipynb
│
├── src/
│   ├── core/
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
│   │   └── feature_selection.py
│   │
│   ├── models/
│   │   ├── isolation_forest/
│   │   └── cnn/
│   │
│   └── pipelines/
│
├── outputs/
│   ├── isolation_forest/
│   │   ├── metrics/
│   │   ├── predictions/
│   │   └── reports/
│   │
│   └── cnn/
│       ├── metrics/
│       ├── predictions/
│       └── reports/
│
├── plots/
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset consists of multivariate sensor time-series recordings collected from an industrial process.

Raw and processed datasets are intentionally excluded from this repository due to size constraints.

Refer to `data/README.md` for dataset placement instructions and preprocessing workflow.

```
```

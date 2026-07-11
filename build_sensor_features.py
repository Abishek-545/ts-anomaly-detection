"""
Builds data/processed/df_sensor_features.csv from raw training data.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew
from scipy.signal import find_peaks

TRAIN_DIR = "data/raw/train"
OUT_PATH = "data/processed/df_sensor_features.csv"


def compute_stats_for_sensor(concatenated):
    x = concatenated.astype(float)
    n = len(x)
    std = np.std(x)
    prominence = max(std * 0.5, 1e-6)
    peaks, _ = find_peaks(x, prominence=prominence)
    t = np.arange(n)
    trend_strength = abs(np.corrcoef(t, x)[0, 1]) if std > 1e-9 else 0.0
    return {
        "mean": np.mean(x),
        "skew": scipy_skew(x) if std > 1e-9 else 0.0,
        "num_peaks": len(peaks),
        "nunique": pd.Series(x).nunique(),
        "trend_strength": trend_strength,
        "std": std,
    }


def main():
    files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {TRAIN_DIR}")

    sample = pd.read_csv(files[0])
    sensors = list(sample.columns)
    all_runs = [pd.read_csv(f) for f in files]

    records = []
    for s in sensors:
        concatenated = np.concatenate([df[s].values for df in all_runs])
        stats = compute_stats_for_sensor(concatenated)
        stats["sensor"] = s
        records.append(stats)

    df = pd.DataFrame(records)

    from sklearn.preprocessing import MinMaxScaler
    proxy_cols = ["mean", "skew", "num_peaks", "nunique", "trend_strength"]
    scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(df[proxy_cols].abs()),
        columns=proxy_cols,
    )
    df["sensor_score"] = scaled.mean(axis=1)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}  shape={df.shape}")


if __name__ == "__main__":
    main()

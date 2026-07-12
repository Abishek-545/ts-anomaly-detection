"""
Diagnose the CNN anomaly detector's failure modes on validation.

Trains the Round-4 CNN (or reuses a saved checkpoint), scores validation at
per-timestep resolution, and writes diagnostic plots to plots/diagnostics/:

  1. val_score_timelines.png  - per-run score over time, true anomaly regions
     shaded, chosen threshold drawn. Shows WHERE we miss / over-flag.
  2. score_distributions.png  - histogram of scores for normal vs anomalous
     timesteps. The overlap IS the ROC-AUC ceiling, made visible.
  3. pr_curve.png             - precision-recall curve with the F0.5-optimal and
     F1-optimal thresholds both marked (tests lever D1: does the portal's F1
     want a different threshold than our F0.5?).
  4. per_sensor_error.png     - mean per-sensor reconstruction error, normal vs
     anomalous windows. Shows which sensors carry the signal.

Also prints a per-run F1 breakdown and a false-negative / false-positive tally.

Run:  python scripts/diagnose_cnn.py            (trains, ~30 min CPU / fast GPU)
      python scripts/diagnose_cnn.py --load     (reuse outputs/cnn/model.pt)
"""

import json
import os
import sys

# Make `src` importable when run as `python scripts/diagnose_cnn.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from src import config
from src.core.windowing import create_windows_per_file, load_labels_per_file, load_split_per_file
from src.evaluation.metrics import find_best_threshold
from src.models.cnn.predict import get_reconstruction_errors
from src.models.cnn.model import ConvAutoencoder1D
from src.models.cnn.train import train_autoencoder
from src.pipelines.run_cnn import fit_mahalanobis_from_validation, score_split_densely

PLOT_DIR = config.OUTPUT_DIR.parent / "plots" / "diagnostics"
MODEL_PATH = config.OUTPUT_DIR / "cnn" / "model.pt"


def main():
    load = "--load" in sys.argv
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("Loading data...")
    train_files = load_split_per_file(config.TRAIN_PATH)
    val_files = load_split_per_file(config.VAL_PATH)
    val_labels = load_labels_per_file(config.VAL_LABELS_PATH)
    y_val = np.concatenate(val_labels)
    sensor_names = list(val_files[0].columns)

    scaler = StandardScaler()
    scaler.fit(np.concatenate([df.values for df in train_files], axis=0))
    train_scaled = [scaler.transform(df.values) for df in train_files]
    X_train, _ = create_windows_per_file(train_scaled, window_size=config.WINDOW_SIZE, step=config.STEP)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_train.shape[2]

    if load and MODEL_PATH.exists():
        print("Loading saved model:", MODEL_PATH)
        model = ConvAutoencoder1D(n_features).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        def eval_fn(m):
            mu, prec = fit_mahalanobis_from_validation(m, val_files, val_labels, scaler, device)
            s = score_split_densely(m, val_files, scaler, device, mu, prec)
            _, fb = find_best_threshold(y_val, s, beta=config.THRESHOLD_BETA)
            return fb

        print(f"Training CNN ({config.EPOCHS} epochs)...")
        model = train_autoencoder(
            X_train, n_features=n_features, device=device, epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE, eval_every=config.CNN_EVAL_EVERY, eval_fn=eval_fn,
        )
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print("Saved model:", MODEL_PATH)

    print("Fitting Mahalanobis model + scoring validation densely...")
    mu, precision = fit_mahalanobis_from_validation(model, val_files, val_labels, scaler, device)
    # per-run scores so we can plot each run separately
    per_run_scores = []
    for df in val_files:
        scaled = scaler.transform(df.values)
        windows, _ = create_windows_per_file([scaled], window_size=config.WINDOW_SIZE, step=config.EVAL_STEP)
        from src.evaluation.metrics import mahalanobis_scores
        from src.core.windowing import expand_window_scores_to_timesteps
        we = get_reconstruction_errors(model, windows, device, batch_size=config.BATCH_SIZE, per_sensor=True)
        ws = mahalanobis_scores(we, mu, precision)
        per_run_scores.append(expand_window_scores_to_timesteps(ws, len(df), config.WINDOW_SIZE, config.EVAL_STEP))
    val_scores = np.concatenate(per_run_scores)

    beta = config.THRESHOLD_BETA
    thr_b, _ = find_best_threshold(y_val, val_scores, beta=beta)
    thr_f1, _ = find_best_threshold(y_val, val_scores, beta=1.0)

    # ---------- Plot 1: per-run timelines ----------
    n = len(val_files)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(14, 2.2 * ((n + 1) // 2)))
    axes = axes.flatten()
    for i, (sc, lab) in enumerate(zip(per_run_scores, val_labels)):
        ax = axes[i]
        ax.plot(sc, lw=0.6, color="#2E6DA4", label="score")
        ax.fill_between(range(len(lab)), 0, sc.max(), where=(lab > 0), color="#E08A8A", alpha=0.35,
                        label="true anomaly")
        ax.axhline(thr_b, color="#3F7D3F", lw=0.8, ls="--", label=f"thr F{beta}")
        ax.set_title(f"val run {i + 1}  (anom rate {lab.mean():.2f})", fontsize=8)
        ax.tick_params(labelsize=6)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=6, loc="upper right")
    fig.suptitle("Mahalanobis score vs true anomalies, per validation run", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "val_score_timelines.png", dpi=110)
    plt.close(fig)

    # ---------- Plot 2: score distributions ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    normal = val_scores[y_val == 0]
    anom = val_scores[y_val == 1]
    hi = np.percentile(val_scores, 99)
    bins = np.linspace(val_scores.min(), hi, 80)
    ax.hist(normal, bins=bins, alpha=0.6, density=True, color="#4C8DBF", label="normal timesteps")
    ax.hist(anom, bins=bins, alpha=0.6, density=True, color="#D46A6A", label="anomalous timesteps")
    ax.axvline(thr_b, color="#3F7D3F", ls="--", label=f"threshold (F{beta})")
    ax.axvline(thr_f1, color="#B8860B", ls=":", label="threshold (F1)")
    ax.set_xlabel("Mahalanobis score"); ax.set_ylabel("density")
    ax.set_title("Score overlap = the ROC-AUC ceiling (less overlap -> higher achievable F1)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "score_distributions.png", dpi=120); plt.close(fig)

    # ---------- Plot 3: PR curve ----------
    prec, rec, thr = precision_recall_curve(y_val, val_scores)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(rec, prec, color="#2E6DA4")
    for t, c, lbl in [(thr_b, "#3F7D3F", f"F{beta}"), (thr_f1, "#B8860B", "F1")]:
        idx = np.searchsorted(thr, t)
        idx = min(idx, len(prec) - 1)
        ax.scatter(rec[idx], prec[idx], color=c, zorder=5, label=f"{lbl} thr: P={prec[idx]:.2f} R={rec[idx]:.2f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision-Recall curve (F0.5 vs F1 operating points)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "pr_curve.png", dpi=120); plt.close(fig)

    # ---------- Plot 4: per-sensor error, normal vs anomalous windows ----------
    errs, wlabs = [], []
    for df, lab in zip(val_files, val_labels):
        scaled = scaler.transform(df.values)
        w, wl = create_windows_per_file([scaled], [lab], window_size=config.WINDOW_SIZE, step=config.STEP)
        errs.append(get_reconstruction_errors(model, w, device, batch_size=config.BATCH_SIZE, per_sensor=True))
        wlabs.append(wl)
    errs = np.concatenate(errs); wlabs = np.concatenate(wlabs)
    mean_norm = errs[wlabs == 0].mean(axis=0)
    mean_anom = errs[wlabs == 1].mean(axis=0)
    order = np.argsort(-(mean_anom - mean_norm))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(sensor_names))
    ax.bar(x - 0.2, mean_norm[order], width=0.4, label="normal windows", color="#4C8DBF")
    ax.bar(x + 0.2, mean_anom[order], width=0.4, label="anomalous windows", color="#D46A6A")
    ax.set_xticks(x); ax.set_xticklabels([sensor_names[k] for k in order], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("mean reconstruction error"); ax.legend(fontsize=8)
    ax.set_title("Per-sensor reconstruction error: which sensors carry the anomaly signal")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "per_sensor_error.png", dpi=120); plt.close(fig)

    # ---------- Text summary ----------
    pred_b = (val_scores > thr_b).astype(int)
    summary = {
        "overall": {
            "f1_at_Fbeta_thr": round(f1_score(y_val, pred_b), 4),
            "f1_at_F1_thr": round(f1_score(y_val, (val_scores > thr_f1).astype(int)), 4),
            "precision_Fbeta": round(precision_score(y_val, pred_b), 4),
            "recall_Fbeta": round(recall_score(y_val, pred_b), 4),
            "false_negatives": int(((pred_b == 0) & (y_val == 1)).sum()),
            "false_positives": int(((pred_b == 1) & (y_val == 0)).sum()),
            "true_anomaly_timesteps": int((y_val == 1).sum()),
        },
        "per_run_f1_at_Fbeta": [],
        "top_signal_sensors": [sensor_names[k] for k in order[:5]],
    }
    for i, (sc, lab) in enumerate(zip(per_run_scores, val_labels)):
        p = (sc > thr_b).astype(int)
        summary["per_run_f1_at_Fbeta"].append({
            "run": i + 1, "anom_rate": round(float(lab.mean()), 3),
            "f1": round(float(f1_score(lab, p, zero_division=0)), 3),
            "recall": round(float(recall_score(lab, p, zero_division=0)), 3),
            "precision": round(float(precision_score(lab, p, zero_division=0)), 3),
        })
    with open(PLOT_DIR / "diagnostic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("\nWrote plots + summary to", PLOT_DIR)


if __name__ == "__main__":
    main()

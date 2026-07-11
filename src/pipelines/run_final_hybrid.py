"""
Final, most-defensible hybrid: CNN + LOF only (Isolation Forest consistently
found to get ~0% weight across every prior experiment - dropped to reduce
unnecessary search space and overfitting risk).

Two things specifically designed to avoid the overfitting-to-small-validation-
set pattern seen in every earlier fusion attempt:
  1. Only ONE free parameter (cnn_weight, since lof_weight = 1 - cnn_weight).
  2. The final deployed weight is the AVERAGE of the weight chosen
     independently in each of the 5 grouped-CV folds, not a fresh
     re-optimization on all of validation.
"""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src import config
from src.core.feature_pipeline import create_features
from src.core.windowing import (
    create_windows_per_file,
    load_labels_per_file,
    load_split_per_file,
)
from src.evaluation.metrics import compute_metrics, find_best_threshold
from src.models.cnn.train import train_autoencoder
from src.pipelines.run_hybrid import normalize, score_cnn_densely
from src.pipelines.run_isolation_forest import load_val_labels

OUTPUT_DIR = config.OUTPUT_DIR / "final_hybrid"
SUBMISSION_PATH = config.OUTPUT_DIR / "final_hybrid_predictions.csv"
LOF_N_NEIGHBORS = 200
WEIGHT_GRID = np.arange(0, 1.001, 0.05)


def main():
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("Loading data...")
    train_files = load_split_per_file(config.TRAIN_PATH)
    val_files = load_split_per_file(config.VAL_PATH)
    test_files = load_split_per_file(config.TEST_PATH)
    val_labels = load_labels_per_file(config.VAL_LABELS_PATH)
    y_val = np.concatenate(val_labels)
    val_run_ids = np.concatenate([np.full(len(df), i) for i, df in enumerate(val_files)])

    scaler = StandardScaler()
    scaler.fit(np.concatenate([df.values for df in train_files], axis=0))
    train_scaled = [scaler.transform(df.values) for df in train_files]

    X_train, _ = create_windows_per_file(train_scaled, window_size=config.WINDOW_SIZE, step=config.STEP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def eval_fn(model):
        val_scores = score_cnn_densely(model, val_files, scaler, device)
        _, f1 = find_best_threshold(y_val, val_scores)
        return f1

    print("Training CNN (plain MSE, early stopping + weight decay)...")
    cnn = train_autoencoder(
        X_train, n_features=X_train.shape[2], device=device,
        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
        eval_every=config.CNN_EVAL_EVERY, eval_fn=eval_fn,
        robust_downweight=False,
    )

    print("Scoring CNN at per-timestep resolution...")
    cnn_val = score_cnn_densely(cnn, val_files, scaler, device)
    cnn_test = score_cnn_densely(cnn, test_files, scaler, device)

    print("Building tabular features + training LOF...")
    X_train_tab = create_features(str(config.TRAIN_PATH), str(config.SELECTED_SENSORS_PATH))
    X_val_tab = create_features(str(config.VAL_PATH), str(config.SELECTED_SENSORS_PATH))
    X_test_tab = create_features(str(config.TEST_PATH), str(config.SELECTED_SENSORS_PATH))

    lof = LocalOutlierFactor(n_neighbors=LOF_N_NEIGHBORS, novelty=True, n_jobs=-1)
    lof.fit(X_train_tab)
    lof_val = -lof.score_samples(X_val_tab)
    lof_test = -lof.score_samples(X_test_tab)

    y_val_tab = load_val_labels()
    assert len(cnn_val) == len(y_val), "window-branch length mismatch"
    if len(lof_val) != len(cnn_val) or len(y_val_tab) != len(y_val):
        raise ValueError(
            f"Length mismatch: window-branch val={len(cnn_val)}, tabular LOF val={len(lof_val)}, "
            f"window labels={len(y_val)}, tabular labels={len(y_val_tab)}."
        )

    print("Normalizing score arrays...")
    cnn_val_n, cnn_test_n = normalize(cnn_val, cnn_test)
    lof_val_n, lof_test_n = normalize(lof_val, lof_test)

    def blend(w_cnn, cnn_s, lof_s):
        return w_cnn * cnn_s + (1 - w_cnn) * lof_s

    print("Run-grouped 5-fold CV: finding the best cnn_weight independently per fold...")
    gkf = GroupKFold(n_splits=5)
    fold_best_weights = []
    oof_scores = np.zeros(len(y_val))

    for tr_idx, te_idx in gkf.split(cnn_val_n.reshape(-1, 1), y_val, groups=val_run_ids):
        best_train_f1, best_w = -1, None
        for w_cnn in WEIGHT_GRID:
            b = blend(w_cnn, cnn_val_n[tr_idx], lof_val_n[tr_idx])
            _, f1_train = find_best_threshold(y_val[tr_idx], b)
            if f1_train > best_train_f1:
                best_train_f1, best_w = f1_train, w_cnn
        fold_best_weights.append(best_w)
        oof_scores[te_idx] = blend(best_w, cnn_val_n[te_idx], lof_val_n[te_idx])

    print(f"Per-fold chosen cnn_weight: {[round(w, 2) for w in fold_best_weights]}")
    print(f"  mean={np.mean(fold_best_weights):.3f}  std={np.std(fold_best_weights):.3f}")

    oof_threshold, _ = find_best_threshold(y_val, oof_scores)
    oof_metrics = compute_metrics(y_val, oof_scores, oof_threshold)
    print("Honest (out-of-fold, run-grouped) hybrid validation metrics:")
    print(json.dumps(oof_metrics, indent=2))

    final_w_cnn = float(np.mean(fold_best_weights))
    print(f"\nFinal deployed weight (mean across folds): cnn={final_w_cnn:.3f}, lof={1-final_w_cnn:.3f}")

    final_val_scores = blend(final_w_cnn, cnn_val_n, lof_val_n)
    final_threshold, _ = find_best_threshold(y_val, final_val_scores)
    final_metrics = compute_metrics(y_val, final_val_scores, final_threshold)
    print("Metrics using the averaged weight, evaluated on all of validation:")
    print(json.dumps(final_metrics, indent=2))

    final_test_scores = blend(final_w_cnn, cnn_test_n, lof_test_n)
    test_pred = (final_test_scores > final_threshold).astype(int)

    cnn_threshold, _ = find_best_threshold(y_val, cnn_val_n)
    lof_threshold, _ = find_best_threshold(y_val, lof_val_n)

    all_metrics = {
        "cnn_only": compute_metrics(y_val, cnn_val_n, cnn_threshold),
        "lof_only": compute_metrics(y_val, lof_val_n, lof_threshold),
        "hybrid_honest_oof": oof_metrics,
        "hybrid_averaged_weight_on_full_val": final_metrics,
        "final_weight_cnn": final_w_cnn,
        "final_weight_lof": 1 - final_w_cnn,
        "per_fold_weights": [round(float(w), 3) for w in fold_best_weights],
    }
    print(json.dumps(all_metrics, indent=2))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("Building submission...")
    test_file_paths = sorted((config.TEST_PATH).glob("*.csv"))
    rows = []
    idx = 0
    for run_id, path in enumerate(test_file_paths, start=1):
        n_timesteps = len(pd.read_csv(path))
        for t in range(n_timesteps):
            rows.append((run_id, t, int(test_pred[idx])))
            idx += 1

    submission = pd.DataFrame(rows, columns=["run_id", "timestep", "prediction"])
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Saved:", SUBMISSION_PATH, submission.shape)

    return all_metrics


if __name__ == "__main__":
    main()

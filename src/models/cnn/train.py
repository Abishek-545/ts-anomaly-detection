"""Training loop for the CNN autoencoder."""
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.cnn.model import ConvAutoencoder1D


def train_autoencoder(
    X_train, n_features, device, epochs, batch_size, lr=1e-3, verbose=True,
    eval_every=None, eval_fn=None, weight_decay=1e-4, patience=3,
    robust_downweight=False, downweight_percentile=90, downweight_factor=0.2,
):
    """Train a ConvAutoencoder1D on normal (unlabeled) windows.

    `robust_downweight` (per Xu et al. 2018's M-ELBO idea, adapted from VAE
    to a plain autoencoder): within each batch, the top
    `downweight_percentile`% highest-loss samples are down-weighted by
    `downweight_factor` before the loss is averaged for backprop -
    highest-loss training samples are disproportionately likely to be
    undetected anomalies contaminating the "normal" training set.

    Returns the trained model (best checkpoint if eval_fn is used).
    """
    from src.models.cnn.model import SensorDataset

    train_loader = DataLoader(SensorDataset(X_train), batch_size=batch_size, shuffle=True)
    model = ConvAutoencoder1D(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn_none = nn.MSELoss(reduction="none")

    best_f1 = -1.0
    best_state = None
    checks_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)

            per_sample_loss = loss_fn_none(out, batch).mean(dim=(1, 2))

            if robust_downweight and len(per_sample_loss) > 1:
                with torch.no_grad():
                    thresh = torch.quantile(per_sample_loss, downweight_percentile / 100.0)
                    weights = torch.where(
                        per_sample_loss > thresh,
                        torch.tensor(downweight_factor, device=device),
                        torch.tensor(1.0, device=device),
                    )
                loss = (per_sample_loss * weights).mean()
            else:
                loss = per_sample_loss.mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} loss: {total_loss / len(train_loader):.6f}")

        if eval_fn is not None and eval_every and (epoch + 1) % eval_every == 0:
            val_f1 = eval_fn(model)
            if verbose:
                print(f"  -> validation F1 at epoch {epoch + 1}: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
                if checks_without_improvement >= patience:
                    if verbose:
                        print(
                            f"  -> early stopping: no improvement for "
                            f"{patience} checks ({patience * eval_every} epochs)"
                        )
                    break

    if best_state is not None:
        if verbose:
            print(f"Restoring best checkpoint (validation F1={best_f1:.4f})")
        model.load_state_dict(best_state)

    return model

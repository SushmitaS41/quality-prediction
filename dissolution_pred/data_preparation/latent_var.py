"""Temporal autoencoder for building per-batch latent vectors from sensor data.

Preserves temporal structure by resampling each batch's multivariate time
series to a fixed length, then training an LSTM autoencoder.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ---------------------------------------------------------------------------
# 1. Resample each batch to a fixed-length time series
# ---------------------------------------------------------------------------

def resample_batches(
    batch_sensor_df: pd.DataFrame,
    seq_len: int = 128,
    meta_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Resample every batch to a fixed number of timesteps via linear interpolation.

    Parameters
    ----------
    batch_sensor_df : pd.DataFrame
        Wide-format: `fpbatch`, `timestamp`, <sensor1>, <sensor2>, ...
    seq_len : int
        Fixed number of timesteps each batch is resampled to.
    meta_cols : list[str], optional
        Non-sensor columns. Defaults to ["fpbatch", "timestamp"].

    Returns
    -------
    X : np.ndarray, shape (n_batches, seq_len, n_sensors)
    fpbatch_ids : list[str]
        Batch identifiers in the same order as X.
    sensor_cols : list[str]
        Sensor column names.
    """
    if meta_cols is None:
        meta_cols = ["fpbatch", "timestamp"]

    sensor_cols = [c for c in batch_sensor_df.columns if c not in meta_cols]
    batches = []
    fpbatch_ids = []

    for fpbatch, grp in batch_sensor_df.groupby("fpbatch", sort=False):
        grp = grp.sort_values("timestamp")
        vals = grp[sensor_cols].values.astype(np.float64)  # (T, n_sensors)
        T = len(vals)

        if T == 0:
            continue

        # Linear interpolation to fixed seq_len
        if T == 1:
            resampled = np.tile(vals, (seq_len, 1))
        else:
            orig_idx = np.linspace(0, 1, T)
            new_idx = np.linspace(0, 1, seq_len)
            resampled = np.column_stack([
                np.interp(new_idx, orig_idx, vals[:, j])
                for j in range(vals.shape[1])
            ])  # (seq_len, n_sensors)

        batches.append(resampled)
        fpbatch_ids.append(fpbatch)

    X = np.stack(batches, axis=0)  # (n_batches, seq_len, n_sensors)
    return X, fpbatch_ids, sensor_cols


# ---------------------------------------------------------------------------
# 2. LSTM Autoencoder model
# ---------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    def __init__(self, n_sensors: int, hidden_dim: int, latent_dim: int, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_sensors, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_sensors)
        _, (h_n, _) = self.lstm(x)       # h_n: (n_layers, batch, hidden)
        h_last = h_n[-1]                  # (batch, hidden)
        return self.fc(h_last)            # (batch, latent_dim)


class TemporalDecoder(nn.Module):
    def __init__(self, n_sensors: int, hidden_dim: int, latent_dim: int,
                 seq_len: int, n_layers: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, n_sensors)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, latent_dim)
        h = self.fc(z)                                  # (batch, hidden)
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)   # (batch, seq_len, hidden)
        out, _ = self.lstm(h)                            # (batch, seq_len, hidden)
        return self.output(out)                          # (batch, seq_len, n_sensors)


class TemporalAutoencoder(nn.Module):
    def __init__(self, n_sensors: int, hidden_dim: int, latent_dim: int,
                 seq_len: int, n_layers: int = 1):
        super().__init__()
        self.encoder = TemporalEncoder(n_sensors, hidden_dim, latent_dim, n_layers)
        self.decoder = TemporalDecoder(n_sensors, hidden_dim, latent_dim, seq_len, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

def train_autoencoder(
    batch_sensor_df: pd.DataFrame,
    latent_dim: int = 8,
    hidden_dim: int = 64,
    n_layers: int = 1,
    seq_len: int = 128,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 10,
    output_path: str | None = None,
) -> tuple[pd.DataFrame, TemporalAutoencoder, StandardScaler]:
    """Train an LSTM autoencoder and return per-batch latent vectors.

    Pipeline:
    1. Resample each batch's time series to `seq_len` timesteps.
    2. Scale each sensor channel with StandardScaler.
    3. Train LSTM autoencoder with early stopping.
    4. Encode all batches → one latent vector per batch.

    Parameters
    ----------
    batch_sensor_df : pd.DataFrame
        Wide-format batch-sensor data (fpbatch, timestamp, sensor cols).
    latent_dim : int
        Dimensionality of the latent vector per batch.
    hidden_dim : int
        LSTM hidden state size.
    n_layers : int
        Number of LSTM layers.
    seq_len : int
        Fixed number of timesteps to resample each batch to.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate.
    val_split : float
        Fraction of batches used for validation.
    patience : int
        Early stopping patience.
    output_path : str, optional
        If provided, saves the latent DataFrame as CSV.

    Returns
    -------
    latent_df : pd.DataFrame
        Columns: `fpbatch`, `latent_0`, `latent_1`, ...
    model : TemporalAutoencoder
    scaler : StandardScaler
    """

    # Step 1: Resample all batches to fixed length
    X_raw, fpbatch_ids, sensor_cols = resample_batches(batch_sensor_df, seq_len=seq_len)
    n_batches, _, n_sensors = X_raw.shape
    print(f"Resampled: {n_batches} batches × {seq_len} timesteps × {n_sensors} sensors")

    # Step 2: Scale per sensor channel (fit on flattened data)
    X_flat = X_raw.reshape(-1, n_sensors)
    X_flat = np.nan_to_num(X_flat, nan=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat).reshape(n_batches, seq_len, n_sensors)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Step 3: Train/val split
    n_val = max(1, int(n_batches * val_split))
    indices = np.random.RandomState(42).permutation(n_batches)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_train = X_tensor[train_idx]
    X_val = X_tensor[val_idx]

    train_loader = DataLoader(
        TensorDataset(X_train), batch_size=batch_size, shuffle=True
    )

    # Step 4: Build model
    model = TemporalAutoencoder(
        n_sensors=n_sensors,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        seq_len=seq_len,
        n_layers=n_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Step 5: Training loop with early stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), X_val).item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:>4}/{epochs} | "
                  f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} "
                      f"(best val_loss={best_val_loss:.6f})")
                break

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Step 6: Encode all batches
    model.eval()
    with torch.no_grad():
        latent = model.encode(X_tensor).numpy()

    latent_df = pd.DataFrame(
        latent, columns=[f"latent_{i}" for i in range(latent_dim)]
    )
    latent_df.insert(0, "fpbatch", fpbatch_ids)

    if output_path is not None:
        latent_df.to_csv(output_path, index=False)

    print(f"\nLatent vectors: {latent_df.shape[0]} batches × {latent_dim} dims")
    print(f"Final val_loss: {best_val_loss:.6f}")

    return latent_df, model, scaler

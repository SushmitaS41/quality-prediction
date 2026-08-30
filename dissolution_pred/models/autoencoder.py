"""Temporal autoencoder for building per-batch latent vectors from sensor data.

Conv1D architecture — parameter-efficient, works well with small sample sizes
(~230 batches). Resamples each batch to a fixed length, then compresses via
stacked convolutions → global average pool → latent vector.
"""

import os
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 0. Config loader
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "model_config.yaml")


def load_config(path: str | None = None) -> dict:
    """Load autoencoder config from YAML."""
    cfg_path = path or _CONFIG_PATH
    with open(cfg_path) as f:
        return yaml.safe_load(f)["autoencoder"]


# ---------------------------------------------------------------------------
# 1. Resample each batch to a fixed-length time series
# ---------------------------------------------------------------------------

def resample_batches(
    batch_sensor_df: pd.DataFrame,
    seq_len: int = 128,
    meta_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Resample every batch to a fixed number of timesteps via linear interpolation."""
    if meta_cols is None:
        meta_cols = ["fpbatch", "timestamp"]

    sensor_cols = [c for c in batch_sensor_df.columns if c not in meta_cols]
    batches = []
    fpbatch_ids = []

    for fpbatch, grp in batch_sensor_df.groupby("fpbatch", sort=False):
        grp = grp.sort_values("timestamp")
        vals = grp[sensor_cols].values.astype(np.float64)
        T = len(vals)

        if T == 0:
            continue

        if T == 1:
            resampled = np.tile(vals, (seq_len, 1))
        else:
            orig_idx = np.linspace(0, 1, T)
            new_idx = np.linspace(0, 1, seq_len)
            resampled = np.column_stack([
                np.interp(new_idx, orig_idx, vals[:, j])
                for j in range(vals.shape[1])
            ])

        batches.append(resampled)
        fpbatch_ids.append(fpbatch)

    X = np.stack(batches, axis=0)
    return X, fpbatch_ids, sensor_cols


# ---------------------------------------------------------------------------
# 2. Conv1D Autoencoder  (few parameters → works with 230 batches)
# ---------------------------------------------------------------------------

class Conv1DEncoder(nn.Module):
    """Stacked Conv1D → global average pool → latent vector."""

    def __init__(self, n_sensors: int, latent_dim: int,
                 channels: list[int] | None = None, dropout: float = 0.0):
        super().__init__()
        if channels is None:
            channels = [16, 32]

        layers = []
        in_ch = n_sensors
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_sensors) → transpose to (batch, n_sensors, seq_len)
        h = self.conv(x.transpose(1, 2))     # (batch, channels[-1], reduced_len)
        h = h.mean(dim=2)                     # global average pool → (batch, channels[-1])
        return self.fc(h)                     # (batch, latent_dim)


class Conv1DDecoder(nn.Module):
    """Latent vector → upsample via ConvTranspose1d → reconstruct."""

    def __init__(self, n_sensors: int, latent_dim: int, seq_len: int,
                 channels: list[int] | None = None, dropout: float = 0.0):
        super().__init__()
        if channels is None:
            channels = [16, 32]

        self.seq_len = seq_len
        rev_channels = list(reversed(channels))

        # Project latent → initial feature map
        self.init_len = seq_len // (2 ** len(channels))
        if self.init_len < 1:
            self.init_len = 1
        self.fc = nn.Linear(latent_dim, rev_channels[0] * self.init_len)
        self.init_channels = rev_channels[0]

        layers = []
        in_ch = rev_channels[0]
        for out_ch in rev_channels[1:]:
            layers += [
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        # Final layer → back to n_sensors
        layers += [
            nn.ConvTranspose1d(in_ch, n_sensors, kernel_size=4, stride=2, padding=1),
        ]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)                                            # (batch, C*init_len)
        h = h.view(-1, self.init_channels, self.init_len)         # (batch, C, init_len)
        h = self.deconv(h)                                        # (batch, n_sensors, ~seq_len)
        # Interpolate to exact seq_len and transpose back
        h = nn.functional.interpolate(h, size=self.seq_len, mode="linear", align_corners=False)
        return h.transpose(1, 2)                                  # (batch, seq_len, n_sensors)


class Conv1DAutoencoder(nn.Module):
    def __init__(self, n_sensors: int, latent_dim: int, seq_len: int,
                 channels: list[int] | None = None, dropout: float = 0.0):
        super().__init__()
        self.encoder = Conv1DEncoder(n_sensors, latent_dim, channels, dropout)
        self.decoder = Conv1DDecoder(n_sensors, latent_dim, seq_len, channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

def train_autoencoder(
    batch_sensor_df: pd.DataFrame,
    config: dict | None = None,
    config_path: str | None = None,
) -> tuple[pd.DataFrame, Conv1DAutoencoder, StandardScaler]:
    """Train a Conv1D autoencoder and return per-batch latent vectors.

    All hyperparameters are read from config.yaml (or passed as a dict).
    """
    cfg = config or load_config(config_path)

    latent_dim = cfg["latent_dim"]
    channels = cfg.get("channels", [32, 64, 128])
    seq_len = cfg["seq_len"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    weight_decay = cfg.get("weight_decay", 0.0)
    val_split = cfg["val_split"]
    patience = cfg["patience"]
    sched_factor = cfg.get("scheduler_factor", 0.5)
    sched_patience = cfg.get("scheduler_patience", 10)
    seed = cfg.get("seed", 42)
    output_path = cfg.get("output_path")

    # Step 0: Impute NaN within each batch before resampling
    meta_cols = ["fpbatch", "timestamp"]
    sensor_cols = [c for c in batch_sensor_df.columns if c not in meta_cols]
    df = batch_sensor_df.copy()
    # Per-batch linear interpolation (fills interior gaps)
    df[sensor_cols] = df.groupby("fpbatch")[sensor_cols].transform(
        lambda x: x.interpolate(method="linear", limit_area="inside")
    )
    # Remaining edge NaNs → per-batch median, then global median
    df[sensor_cols] = df.groupby("fpbatch")[sensor_cols].transform(
        lambda x: x.fillna(x.median())
    )
    global_medians = df[sensor_cols].median()
    df[sensor_cols] = df[sensor_cols].fillna(global_medians)
    n_remaining = df[sensor_cols].isna().sum().sum()
    print(f"Pre-AE imputation: {batch_sensor_df[sensor_cols].isna().sum().sum()} NaN → {n_remaining} NaN")

    # Step 1: Resample
    X_raw, fpbatch_ids, sensor_cols = resample_batches(df, seq_len=seq_len)
    n_batches, _, n_sensors = X_raw.shape
    print(f"Resampled: {n_batches} batches × {seq_len} timesteps × {n_sensors} sensors")

    # Step 2: Scale per sensor channel
    X_flat = X_raw.reshape(-1, n_sensors)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat).reshape(n_batches, seq_len, n_sensors)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Step 3: Train/val split
    n_val = max(1, int(n_batches * val_split))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_batches)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_tr = X_tensor[train_idx]
    X_vl = X_tensor[val_idx]

    # Fix all random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(TensorDataset(X_tr), batch_size=batch_size,
                              shuffle=True, generator=g)

    # Step 4: Build model
    dropout = cfg.get("dropout", 0.0)
    model = Conv1DAutoencoder(
        n_sensors=n_sensors,
        latent_dim=latent_dim,
        seq_len=seq_len,
        channels=channels,
        dropout=dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Conv1D autoencoder | {n_params:,} parameters | latent_dim={latent_dim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sched_factor, patience=sched_patience,
    )
    criterion = nn.MSELoss()

    # Step 5: Training loop
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

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_vl), X_vl).item()

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:>4}/{epochs} | "
                  f"train={train_loss:.6f} | val={val_loss:.6f} | lr={current_lr:.2e}")

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

    if best_state is not None:
        model.load_state_dict(best_state)

    # Step 6: Encode all batches
    model.eval()
    with torch.no_grad():
        latent = model.encode(X_tensor).numpy()

    latent_df = pd.DataFrame(latent, columns=[f"latent_{i}" for i in range(latent_dim)])
    latent_df.insert(0, "fpbatch", fpbatch_ids)

    # Report
    variance_explained = max(0.0, 1.0 - best_val_loss)
    print(f"\nLatent vectors: {latent_df.shape[0]} batches × {latent_dim} dims")
    print(f"Final val_loss (MSE on standardized): {best_val_loss:.6f}")
    print(f"Approx variance explained: {variance_explained:.1%}")

    # Step 7: Save all artifacts
    save_dir = cfg.get("save_dir", "trained_autoencoder")
    save_artifacts(model, scaler, cfg, latent_df, n_sensors, save_dir)

    return latent_df, model, scaler


def save_artifacts(
    model: Conv1DAutoencoder,
    scaler: StandardScaler,
    config: dict,
    latent_df: pd.DataFrame,
    n_sensors: int,
    save_dir: str = "trained_autoencoder",
) -> None:
    """Save all trained autoencoder artifacts to a folder."""
    os.makedirs(save_dir, exist_ok=True)

    # Model weights
    torch.save(model.state_dict(), os.path.join(save_dir, "autoencoder_weights.pt"))

    # Scaler
    joblib.dump(scaler, os.path.join(save_dir, "sensor_scaler.pkl"))

    # Config used for training (for architecture reconstruction)
    config_to_save = dict(config)
    config_to_save["n_sensors"] = n_sensors
    with open(os.path.join(save_dir, "training_config.yaml"), "w") as f:
        yaml.dump(config_to_save, f, default_flow_style=False)

    # Latent vectors
    latent_df.to_csv(os.path.join(save_dir, "latent_vectors.csv"), index=False)

    print(f"\nArtifacts saved to {save_dir}/")
    for fname in sorted(os.listdir(save_dir)):
        fpath = os.path.join(save_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {fname} ({size_kb:.1f} KB)")


def load_artifacts(
    save_dir: str = "trained_autoencoder",
) -> tuple[Conv1DAutoencoder, StandardScaler, dict, pd.DataFrame]:
    """Load a previously saved autoencoder and its artifacts."""
    # Config
    with open(os.path.join(save_dir, "training_config.yaml")) as f:
        config = yaml.safe_load(f)

    # Rebuild model architecture
    model = Conv1DAutoencoder(
        n_sensors=config["n_sensors"],
        latent_dim=config["latent_dim"],
        seq_len=config["seq_len"],
        channels=config.get("channels", [32, 64, 128]),
    )
    model.load_state_dict(torch.load(os.path.join(save_dir, "autoencoder_weights.pt"), weights_only=True))
    model.eval()

    # Scaler
    scaler = joblib.load(os.path.join(save_dir, "sensor_scaler.pkl"))

    # Latent vectors (preserve fpbatch as string)
    latent_df = pd.read_csv(os.path.join(save_dir, "latent_vectors.csv"),
                            dtype={"fpbatch": str})

    print(f"Loaded autoencoder from {save_dir}/")
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Latent vectors: {latent_df.shape}")

    return model, scaler, config, latent_df

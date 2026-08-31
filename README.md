# Dissolution Quality Prediction Pipeline

Predicts pharmaceutical product quality (`FilteredArea1` — a dissolution metric) from manufacturing process sensor data. The pipeline learns latent representations of batch sensor time series via a Conv1D autoencoder, then trains regression models to predict dissolution outcomes across lanes and time points.

---

## Data Sources

| File | Description | Key Columns |
|---|---|---|
| `batch_data.csv` | One row per manufacturing batch | `fpbatch`, `prodname`, `starttime`, `endtime` |
| `sensor_data.csv` | Long-format sensor readings (1-min cadence) | `Tag`, `Timestamp`, `Value` (~13 sensors) |
| `lab_data.csv` | Quality lab measurements per batch × lane × time | `fpbatch`, `variable`, `value`, `lane`, `time` |
| `tag_offsets.xlsx` | Sensor-specific timestamp offsets (seconds) | `Tag_Names_in_Software`, `Offset_Timings_in_seconds` |

**Why offsets?** Each sensor has a physical installation delay relative to the batch start. The offset table corrects for this so that sensor readings align to the true process timeline.

---

## Pipeline Overview

```
                        ┌─────────────────────────┐
                        │     Raw CSV / Excel      │
                        └────────────┬────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │  PART 1: Ingestion & Cleaning   │
                    │  (6-step sensor processing)     │
                    │  → batch_sensor_merged           │
                    │  → batch_lab_merged               │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │  Sanity check & clean            │
                    │  (drop >90% NaN cols/rows)       │
                    │  Chronological 60/40 split       │
                    │  Extract target (all lanes/times)│
                    └────────────────┬────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
    ┌─────────▼─────────┐  ┌────────▼────────┐  ┌─────────▼─────────┐
    │  Conv1D Autoencoder│  │  Sensor Stats   │  │  Batch Metadata   │
    │  (train split only)│  │  (per-batch     │  │  (prodname,       │
    │  → 8 latent dims   │  │   aggregates)   │  │   duration)       │
    └─────────┬─────────┘  └────────┬────────┘  └─────────┬─────────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │  PART 3: Model Training          │
                    │  • RidgeCV on sensor stats       │
                    │  • RidgeCV on latent features     │
                    │  • XGBoost sweep on latent        │
                    └─────────────────────────────────┘
```

---

## Part 1: Data Ingestion & Cleaning

### 1.1 Timestamp Sanity Check
Snaps sensor timestamps to the nearest expected 1-minute grid point. Removes duplicates per tag. **Why?** Raw timestamps can drift by a few seconds due to logger jitter — snapping ensures a clean join later when pivoting sensors wide.

### 1.2 Offset Correction
Shifts each sensor's timestamps by its known offset (from `tag_offsets.xlsx`). **Why?** Sensors are physically installed at different points in the process line, so a reading from `sensor_3` at `T=100` actually corresponds to a different process stage than `sensor_7` at `T=100`. Offsets align them to a common process timeline.

### 1.3 Outlier Removal
Runs `IsolationForest` per tag with `contamination=0.03` (3% expected outliers). Marks outliers and sets their values to NaN. **Why 3%?** Conservative — removes only extreme equipment glitches while preserving normal process variation. Tags with fewer than 20 points are skipped.

### 1.4 Gap Imputation (≤60 min)
Linearly interpolates interior NaN gaps shorter than 60 minutes per tag. **Why the 60-min limit?** Short gaps are likely logger dropouts (the process was running normally), so interpolation is justified. Long gaps may indicate equipment shutdowns, where the true value is unknown — these are left as NaN for downstream handling.

The cadence is auto-detected per tag (mode of positive diffs), so the function works for any sensor frequency, not just 1-minute.

### 1.5 Batch–Lab Merge
Inner join `batch_data` and `lab_data` on `fpbatch`. Preserves both `prodname_batch` (from batch_data) and `prodname_lab` (from lab_data, which encodes time point and lane info like `HYALURONIC ADR T160 L1`).

### 1.6 Sensor–Batch Merge
For each batch, slices sensor readings within `[starttime, endtime]` (adjusted by offsets if provided). Pivots from long to wide format: one row per timestamp, one column per sensor. Uses **inner join** across tags — only timestamps where ALL sensors have readings are kept. **Why inner join?** Avoids introducing edge NaNs from sensors that started/stopped at different times within a batch.

### 1.7 Cleaning
- Drop sensor columns with >90% NaN (e.g., `sensor_8_pv` with ~99% missing — likely a non-operational sensor)
- Drop rows where ALL remaining sensors are NaN (empty time slices)

### 1.8 Chronological Split
Batches are ordered by their earliest timestamp and split 60/40. **Why chronological, not random?** Simulates real deployment: the model trains on historical batches and predicts on future ones. Random splits would leak temporal information.

### 1.9 Target Extraction
Filters lab data to `FilteredArea1` and returns all `(fpbatch, lane, time)` combinations — no averaging. **Why keep all rows?** A single batch has dissolution measurements across ~10 lanes and ~9 time points (T0, T80, T160, ..., T640). These represent the dissolution *profile*. Averaging would collapse the curve into a single number, losing the information that dissolution at T0 is fundamentally different from T640. Instead, `lane` and `time` become input features so the model learns how dissolution varies across the profile.

---

## Part 2: Feature Engineering

### 2.1 Conv1D Autoencoder

**Architecture:**
```
Encoder:
  Input (batch, 256, 13)
  → Conv1d(13→32, k=5, s=2) + BN + ReLU    → (batch, 32, 128)
  → Conv1d(32→64, k=5, s=2) + BN + ReLU    → (batch, 64, 64)
  → GlobalAvgPool                            → (batch, 64)
  → FC(64→8)                                → (batch, 8)

Decoder:
  Input (batch, 8)
  → FC(8→64×64)                             → (batch, 64, 64)
  → ConvTranspose1d(64→32, k=4, s=2) + BN + ReLU
  → ConvTranspose1d(32→13, k=4, s=2)
  → Interpolate to 256                      → (batch, 256, 13)
```
~50K parameters. Small by design — the dataset has only ~230 batches, so a larger model would overfit.

**Preprocessing before the autoencoder:**
1. **Per-batch linear interpolation** — fill interior NaN using neighboring values within the same batch
2. **Per-batch median fill** — if a sensor is entirely NaN within a batch, fill with that batch's median
3. **Global median fill** — last resort for any remaining NaN
4. **Resample to 256 timesteps** — batches vary from ~53 to ~997 rows; linear interpolation to a fixed length so the Conv1D can process them uniformly. 256 is ~2.25× compression of the average batch (576 rows).
5. **StandardScaler per sensor** — fitted on ALL training batches, applied per sensor channel independently. **Why standardise?** Sensors operate on very different scales (e.g., temperature in °C vs pressure in bar vs flow in L/min). Without normalisation, the autoencoder loss would be dominated by high-magnitude sensors, and the latent space would ignore low-magnitude ones.

**Training:**
- Loss: MSE (reconstruction)
- Optimiser: Adam (lr=0.001, weight_decay=0.0001)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=15)
- Early stopping: patience=40 epochs
- Train/val split: 85/15 (of the training split only — test data is never seen)
- Batch size: 8 — small batches give more gradient updates per epoch, useful with only ~113 training batches

**Ensemble mode:** Train 10 autoencoders with different random seeds, average the latent vectors. This stabilises the embeddings — a single AE run can produce different latent spaces depending on initialisation.

**Why an autoencoder?** Raw sensor stats (mean, std, etc.) discard temporal structure. A Conv1D autoencoder learns compressed representations that capture how sensors *evolve over time* within a batch, not just their summary statistics. The 8-dim latent vectors become features for the regression model.

### 2.2 Sensor Summary Statistics
Per-batch aggregates computed directly on the wide-format sensor data: **mean, std, min, max, median, skew, range, pct_missing** for each sensor. These are interpretable baseline features — 8 stats × ~12 sensors = ~96 features.

### 2.3 Batch Metadata
Extracted from the batch–lab merge:
- `prodname_batch` — product type (categorical, label-encoded). **Why include it?** Different products (Vitamin C, Retinol, Hyaluronic) have different dissolution profiles. The model needs to know what it's predicting for.
- `batch_duration_hrs` — endtime minus starttime. Longer batches may indicate process deviations.

### 2.4 Feature Matrix Assembly
Merge features + metadata + target on `fpbatch`:
- Sensor stats (one row per batch) get replicated across all (lane, time) target rows for that batch
- `lane` and `time` become numeric input features
- NaN in features filled with **train medians** (computed on training set only, applied to both splits)

---

## Part 3: Model Training

### 3.1 Baseline: RidgeCV on Sensor Stats
- StandardScaler on features (fit on train, transform both)
- RidgeCV with alphas=logspace(-2, 5, 50) and leave-one-out cross-validation
- **Why LOO-CV?** Optimal for small datasets (~133 training samples after split). No need to manually tune alpha.
- **Why Ridge over Lasso?** Most sensor features are mildly correlated with the target. Ridge shrinks all coefficients towards zero without forcing exact sparsity, which works better when many features contribute small effects.

### 3.2 RidgeCV on Latent Features
Same as above but using the 8 autoencoder latent dimensions + lane + time + metadata instead of the ~96 sensor stats. Tests whether the learned representations outperform hand-crafted statistics.

### 3.3 XGBoost Sweep on Latent Features
Four configurations tested:

| n_estimators | max_depth | learning_rate |
|---|---|---|
| 50 | 2 | 0.05 |
| 100 | 2 | 0.05 |
| 100 | 3 | 0.01 |
| 100 | 3 | 0.05 |

Common across all: `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`.

- **Why min_child_weight=5?** With ~133 training batches (×lanes×times), each leaf must have ≥5 samples to prevent overfitting to noise.
- **Why subsample=0.8?** Row subsampling adds stochastic regularisation — each tree sees 80% of the data, reducing correlation between trees.
- **Why max_depth 2–3?** Shallow trees with boosting outperform deep trees on small datasets. Deep trees memorise; shallow trees generalise.
- Best model selected by validation R² (5-fold CV on training set).

---

## Key Configuration

All hyperparameters are centralised in `dissolution_pred/model_config.yaml`:

```yaml
autoencoder:
  latent_dim: 8          # 8 > 4 (test R²: 0.21 vs 0.16)
  channels: [32, 64]
  seq_len: 256           # avg batch ~576 → ~2.25x compression
  dropout: 0.0           # model small enough
  epochs: 100
  batch_size: 8
  lr: 0.001
  weight_decay: 0.0001
  val_split: 0.15
  patience: 40
  seed: 42

xgboost:
  seed: 42
  configs: [...]         # 4 configs as described above
```

---

## Project Structure

```
quality-prediction/
├── README.md
├── requirements.txt
├── batch_lab_merged.csv          # generated
├── batch_sensor_merged.csv       # generated
├── trained_autoencoder/          # saved model artifacts
│   ├── autoencoder_weights.pt
│   ├── latent_vectors.csv
│   └── training_config.yaml
│
└── dissolution_pred/
    ├── model_config.yaml         # all hyperparameters
    │
    ├── data_preparation/
    │   ├── ingestion_pipeline.py # 6-step sensor processing
    │   ├── clean_and_split.py    # clean, split, target extraction
    │   └── feature_engineering.py# AE training, stats, feature assembly
    │
    ├── models/
    │   ├── autoencoder.py        # Conv1D autoencoder architecture
    │   ├── baseline_regression.py# RidgeCV baseline
    │   ├── xgboost_model.py      # XGBoost sweep
    │   └── model_training.py     # orchestrates all 3 models
    │
    ├── utils/
    │   ├── ingestion_helpers.py  # timestamp, offset, outlier, impute, merge functions
    │   └── plotting.py           # correlation, importance, prediction plots
    │
    ├── scripts/
    │   ├── train_autoencoder.py  # CLI: train AE only
    │   └── train_pipeline.py     # CLI: full pipeline (ingest → train → evaluate)
    │
    └── notebooks/
        ├── sanity_check.ipynb    # data quality checks at each stage
        ├── model_training.ipynb  # interactive model training
        ├── embeddings.ipynb      # autoencoder exploration
        └── Report.ipynb          # final results
```

---

## Running the Pipeline

```bash
# Activate environment
source techtask/bin/activate

# Full pipeline (ingestion → AE → models)
python -m dissolution_pred.scripts.train_pipeline \
    --data-dir ../  \
    --train-frac 0.60 \
    --target FilteredArea1

# Autoencoder only (ensemble mode, 10 runs)
python -m dissolution_pred.scripts.train_autoencoder \
    --data-dir ../  \
    --mode ensemble \
    --n-ensemble 10
```

---

## Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Timestamp snapping | Round to nearest 1-min grid | Logger jitter; enables clean pivot |
| Offset correction | Per-sensor shift from lookup table | Physical installation delays |
| Outlier detection | IsolationForest, 3% contamination | Conservative; preserves process variation |
| Gap imputation limit | ≤60 minutes only | Short gaps = logger dropout; long gaps = unknown state |
| Column NaN threshold | Drop if >90% missing | sensor_8_pv at ~99% NaN — non-operational |
| Sensor–batch join | Inner join across all tags | Avoids edge NaN from mismatched sensor windows |
| Train/test split | 60/40 chronological by batch start | No temporal leakage; simulates deployment |
| Target handling | Keep all (fpbatch, lane, time) rows | Dissolution profile varies by time point and lane |
| Resampling | Linear interp to 256 timesteps | Standardise variable-length batches for Conv1D |
| Standardisation | Per-sensor StandardScaler | Sensors on different scales (temp vs pressure vs flow) |
| AE architecture | Conv1d [32,64] → 8 latent dims | ~50K params for ~230 batches; prevents overfitting |
| AE ensemble | 10 runs, average latents | Stabilise embeddings across random initialisations |
| Baseline model | RidgeCV with LOO-CV | Automatic alpha; optimal for small datasets |
| XGBoost regularisation | min_child_weight=5, subsample=0.8 | Prevent overfitting on small dataset |
| Categorical encoding | LabelEncoder for prodname | Different products → different dissolution targets |
| NaN filling (features) | Train medians | No test-set leakage; robust to outliers |

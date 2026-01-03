"""
train_model_nasa_optimal.py
===========================

This module implements a highly optimised training pipeline for the
classification of exoplanet transit signals.  It blends ideas from
NASA’s ExoMiner/ExoMiner++ and Astronet pipelines with best practices
from the machine‑learning literature to maximise recall, precision and
F1 on heavily imbalanced datasets.  Compared to the baseline and
"advanced" scripts, this version introduces several key
enhancements:

* **Morphological feature extraction** – In addition to the seven
  astrophysical scalar inputs (orbital period, transit duration,
  transit depth, planet radius, stellar radius, effective temperature
  and surface gravity), a set of ten morphology descriptors are
  computed directly from each global light curve.  These features
  capture transit depth, width, slopes, area, symmetry, signal‑to‑
  noise ratio and the maximum cross‑correlation with triangular
  templates of three different widths.  Incorporating hand‑crafted
  features alongside learned embeddings helps the model discriminate
  subtle transit signatures.

* **Deeper CNN with Squeeze‑and‑Excitation (SE) gating and
  Transformer encoders** – The global branch employs residual
  convolutional blocks with SE modules to recalibrate channel
  responses.  Transformer encoders model long‑range periodic
  dependencies.  The local branch uses a shallow CNN with SE gating.
  This architecture is inspired by ExoMiner++, which shows that SE
  gating improves the network’s ability to focus on informative
  channels.

* **Robust data handling** – A built‑in validation routine scans
  `.npz` files for corrupted archives, missing keys and shape
  mismatches.  The data generator skips unreadable files at runtime
  instead of crashing.  Oversampling is applied only to genuine
  positives, and synthetic samples are tracked via a boolean mask.

* **Enhanced synthetic transits** – Synthetic light curves are
  trapezoidal with configurable ingress/egress fractions and
  astrophysical scalar draws.  A user‑specified fraction of positive
  examples can be supplemented with synthetic samples, following
  evidence that a 70–75 % synthetic mix improves F1.

* **Hybrid ensembling with calibrated stacking** – Learned embeddings
  from the neural network are fed to an XGBoost classifier.  Out‑of‑
  fold predictions from both models are combined via averaging,
  weighted averaging or logistic/ridge stacking.  Thresholds are
  selected via precision–recall analysis to maximise F1.

This pipeline is a research prototype approximating NASA standards for
transit classification.  Achieving state‑of‑the‑art recall and
precision (e.g. ExoMiner’s 93.6 % recall at 99 % precision)
requires access to the full Kepler/TESS training sets and more
physical realism in synthetic signals.  Nonetheless, the design here
lays a strong foundation for further optimisation.

Usage example:

```bash
python scripts/train_model_nasa_optimal.py \
    --hybrid \
    --ensemble_method stacking \
    --synthetic_ratio 0.7 \
    --augment --augment_positive_only \
    --oversample \
    --loss focal \
    --num_conv_blocks 5 \
    --num_transformer_blocks 2 \
    --conv_filters 64 \
    --tune_xgboost
```
"""

import os
import glob
import argparse
from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Try importing XGBoost.  If unavailable, the hybrid pipeline will be disabled.
try:
    import xgboost as xgb  # type: ignore
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# Configuration Defaults
# -----------------------------
DATA_DIR = "notebooks/results_koi"
IMG_SHAPE_GLOBAL = (2001, 1)
IMG_SHAPE_LOCAL = (201, 1)
# The seven astrophysical features: period, duration, depth, planet radius,
# stellar radius, effective temperature, surface gravity.
ASTRO_SCALAR_DIM = 7

# -----------------------------
# Loss Functions
# -----------------------------
class FocalLoss(tf.keras.losses.Loss):
    """Binary focal loss that down‑weights well‑classified examples.

    Parameters
    ----------
    gamma : float
        Focusing parameter; higher values focus more on hard examples.
    alpha : float
        Weighting factor for the positive class; useful for imbalanced data.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        y_true = tf.reshape(y_true, [-1, 1])
        y_pred = tf.reshape(y_pred, [-1, 1])
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow((1.0 - p_t), self.gamma)
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)


# -----------------------------
# Morphological Feature Extraction
# -----------------------------
def compute_morphological_features(global_view: np.ndarray) -> np.ndarray:
    """Compute a set of morphology descriptors from a normalised global light curve.

    The features include:
    * maximum transit depth (baseline minus minimum value)
    * transit width at half depth
    * ingress and egress slopes (depth divided by rise/fall lengths)
    * area under the dip (sum of dip signal)
    * symmetry (difference between areas of the two halves)
    * signal‑to‑noise ratio (depth divided by baseline noise standard deviation)
    * maximum cross‑correlation with triangular templates of widths 5, 10 and 20
    The baseline is estimated from the median of the first and last 10 % of
    samples.  Dips are considered as positive deviations (baseline − flux).

    Parameters
    ----------
    global_view : np.ndarray, shape (N, 1)
        Normalised global view of the light curve.

    Returns
    -------
    features : np.ndarray, shape (10,)
        Morphological feature vector.
    """
    g = global_view.reshape(-1)
    # Estimate baseline from the edges of the light curve
    edge_len = max(20, len(g) // 10)
    baseline = np.median(np.concatenate([g[:edge_len], g[-edge_len:]]))
    dip_signal = baseline - g  # positive for dips
    # Estimate noise from edges
    noise_baseline = np.concatenate([dip_signal[:edge_len], dip_signal[-edge_len:]])
    noise_std = np.std(noise_baseline) + 1e-7
    # Maximum dip depth
    max_dip = float(np.max(dip_signal))
    # Transit width at half depth
    threshold = max_dip * 0.5
    indices = np.where(dip_signal >= threshold)[0]
    if indices.size > 0:
        width = float(indices[-1] - indices[0] + 1)
        center = (indices[0] + indices[-1]) // 2
        ingress_len = max(center - indices[0], 1)
        egress_len = max(indices[-1] - center, 1)
        slope_ingress = max_dip / ingress_len
        slope_egress = max_dip / egress_len
        area = float(np.sum(dip_signal[indices]))
        half = len(indices) // 2
        area_left = float(np.sum(dip_signal[indices[:half]]))
        area_right = float(np.sum(dip_signal[indices[half:]]))
        symmetry = area_left - area_right
    else:
        width = 0.0
        slope_ingress = 0.0
        slope_egress = 0.0
        area = 0.0
        symmetry = 0.0
    snr = max_dip / noise_std
    # Cross‑correlation with triangular templates
    corr_feats: List[float] = []
    for w in [5, 10, 20]:
        # Build triangular template of length w
        if w > 1:
            t = np.array([1 - abs(i - (w - 1) / 2) / ((w - 1) / 2) for i in range(w)])
        else:
            t = np.array([1.0])
        # Compute cross‑correlation (valid region)
        conv = np.correlate(dip_signal, t, mode='valid')
        corr_feats.append(float(np.max(conv)))
    return np.array([max_dip, width, slope_ingress, slope_egress, area, symmetry, snr] + corr_feats, dtype=np.float32)


# Determine morphological feature dimension using a dummy light curve
_dummy = np.zeros(IMG_SHAPE_GLOBAL)
MORPH_FEATURE_DIM = len(compute_morphological_features(_dummy))

# Total scalar feature dimension: astrophysical + morphology
SCALAR_DIM = ASTRO_SCALAR_DIM + MORPH_FEATURE_DIM


# -----------------------------
# Synthetic Data Generation
# -----------------------------
def generate_synthetic_transits(num_samples: int,
                                length_global: int = IMG_SHAPE_GLOBAL[0],
                                length_local: int = IMG_SHAPE_LOCAL[0],
                                noise_level: float = 0.001,
                                depth_range: Tuple[float, float] = (0.001, 0.02),
                                duration_range: Tuple[int, int] = (10, 100),
                                seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic light curves with trapezoidal transit shapes.

    Each synthetic light curve consists of a baseline with Gaussian noise and
    a transit with configurable ingress/egress durations.  The transit is
    modelled as a trapezoid: a linear ingress, a flat bottom and a linear
    egress.  The scalar features are drawn from astrophysically plausible
    ranges.  This approximates the injection of realistic transits suggested
    in exoplanet studies.

    Parameters
    ----------
    num_samples : int
        Number of synthetic samples to generate.
    length_global : int
        Length of the global light curve (default 2001).
    length_local : int
        Length of the local light curve (default 201).
    noise_level : float
        Standard deviation of the Gaussian noise added to the baseline.
    depth_range : tuple
        Min and max depth (fractional drop) of the transit.
    duration_range : tuple
        Min and max duration (number of samples) of the transit event.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    global_arr : np.ndarray
        Full synthetic light curves of shape (num_samples, length_global, 1).
    local_arr : np.ndarray
        Zoomed‑in views around the transit of shape (num_samples, length_local, 1).
    scalars : np.ndarray
        Astrophysical scalar features (no morphology) of shape (num_samples, 7).
    """
    rng = np.random.default_rng(seed)
    global_arr = np.zeros((num_samples, length_global, 1), dtype=np.float32)
    local_arr = np.zeros((num_samples, length_local, 1), dtype=np.float32)
    scalars = np.zeros((num_samples, ASTRO_SCALAR_DIM), dtype=np.float32)
    for i in range(num_samples):
        baseline = rng.normal(0.0, noise_level, size=length_global)
        lightcurve = np.ones(length_global) + baseline
        depth = rng.uniform(*depth_range)
        duration = rng.integers(*duration_range)
        ingress = max(1, int(duration * 0.15))
        egress = max(1, int(duration * 0.15))
        flat = duration - ingress - egress
        center = rng.integers(duration, length_global - duration)
        start = center - duration // 2
        idx = start
        # Ingress
        for j in range(ingress):
            if 0 <= idx < length_global:
                frac = (j + 1) / ingress
                lightcurve[idx] -= depth * frac
            idx += 1
        # Flat bottom
        for j in range(max(flat, 0)):
            if 0 <= idx < length_global:
                lightcurve[idx] -= depth
            idx += 1
        # Egress
        for j in range(egress):
            if 0 <= idx < length_global:
                frac = 1 - (j + 1) / egress
                lightcurve[idx] -= depth * frac
            idx += 1
        global_arr[i, :, 0] = lightcurve.astype(np.float32)
        l_start = max(0, center - length_local // 2)
        l_end = l_start + length_local
        if l_end > length_global:
            l_end = length_global
            l_start = l_end - length_local
        local_arr[i, :, 0] = lightcurve[l_start:l_end].astype(np.float32)
        # Scalars: period, duration, depth, planet radius, star radius, Teff, log g
        period = rng.uniform(0.5, 365.0)
        planet_radius = rng.uniform(0.5, 20.0)
        star_radius = rng.uniform(0.5, 2.0)
        teff = rng.uniform(3000, 7000)
        logg = rng.uniform(4.0, 5.0)
        scalars[i] = np.array([
            period,
            duration * 0.02,
            depth,
            planet_radius,
            star_radius,
            teff,
            logg
        ], dtype=np.float32)
    return global_arr, local_arr, scalars


# -----------------------------
# Data Generator
# -----------------------------
class ExoplanetDataGenerator(Sequence):
    """Keras sequence to load and batch light curves with optional augmentation.

    This generator performs on‑the‑fly loading of `.npz` files, optional data
    augmentation (flip, roll, jitter) and oversampling of the positive class.  If
    synthetic samples are provided they are concatenated to the real data.  A
    boolean mask tracks synthetic entries so that oversampling is applied only
    to genuine positives.  Morphological features are computed for every
    example and appended to the astrophysical scalars.
    """

    def __init__(self,
                 file_paths: np.ndarray,
                 labels: np.ndarray,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 augment: bool = False,
                 augment_positive_only: bool = False,
                 oversample: bool = False,
                 synthetic_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                 synthetic_ratio: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.augment_positive_only = augment_positive_only

        # Start with real data
        self.file_paths = file_paths.copy()
        self.labels = labels.copy()
        # Track synthetic entries
        self.is_synthetic = np.zeros(len(self.file_paths), dtype=bool)
        self.synthetic_count = 0
        self.synthetic_index_offset = len(file_paths)

        # Add synthetic data
        if synthetic_data is not None and synthetic_ratio > 0:
            syn_global, syn_local, syn_scalar = synthetic_data
            syn_labels = np.ones(len(syn_global), dtype=np.int32)
            self.synthetic_count = len(syn_global)
            self.synthetic_global = syn_global
            self.synthetic_local = syn_local
            self.synthetic_scalar = syn_scalar
            self.file_paths = np.concatenate([self.file_paths, np.array([None] * self.synthetic_count)])
            self.labels = np.concatenate([self.labels, syn_labels])
            self.is_synthetic = np.concatenate([
                self.is_synthetic,
                np.ones(self.synthetic_count, dtype=bool)
            ])
        else:
            self.synthetic_global = None
            self.synthetic_local = None
            self.synthetic_scalar = None

        # Oversample genuine positives to balance classes
        if oversample:
            real_pos_indices = np.where((self.labels == 1) & (~self.is_synthetic))[0]
            neg_indices = np.where(self.labels == 0)[0]
            total_pos = np.sum(self.labels == 1)
            if total_pos < len(neg_indices) and len(real_pos_indices) > 0:
                extra_needed = len(neg_indices) - total_pos
                extra = np.random.choice(real_pos_indices, size=extra_needed, replace=True)
                self.file_paths = np.concatenate([self.file_paths, self.file_paths[extra]])
                self.labels = np.concatenate([self.labels, self.labels[extra]])
                self.is_synthetic = np.concatenate([
                    self.is_synthetic,
                    np.zeros(len(extra), dtype=bool)
                ])

        # Recompute synthetic offset: first synthetic index
        if self.synthetic_count > 0:
            synthetic_mask_indices = np.where(self.is_synthetic)[0]
            if synthetic_mask_indices.size > 0:
                self.synthetic_index_offset = synthetic_mask_indices[0]

        # Prepare indices
        self.indices = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        # Generate batch indices
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_global, X_local, X_scalar, y = [], [], [], []
        for idx in batch_indices:
            label = self.labels[idx]
            try:
                if self.synthetic_global is not None and self.is_synthetic[idx]:
                    # Synthetic sample: compute morphological features from synthetic global
                    synthetic_idx = (idx - self.synthetic_index_offset) % self.synthetic_count
                    g_view = self.synthetic_global[synthetic_idx]
                    l_view = self.synthetic_local[synthetic_idx]
                    scalars = self.synthetic_scalar[synthetic_idx]
                else:
                    path = self.file_paths[idx]
                    with np.load(path) as data:
                        g_view = data['global_view']
                        l_view = data['local_view']
                        scalars = data['scalars']
                # Check shapes
                if g_view.shape != IMG_SHAPE_GLOBAL or l_view.shape != IMG_SHAPE_LOCAL:
                    continue
                # Augmentation
                if self.augment:
                    should_aug = True
                    if self.augment_positive_only and label != 1:
                        should_aug = False
                    if should_aug:
                        if np.random.rand() < 0.5:
                            g_view = np.flip(g_view, axis=0)
                            l_view = np.flip(l_view, axis=0)
                        shift = np.random.randint(-10, 11)
                        g_view = np.roll(g_view, shift, axis=0)
                        l_view = np.roll(l_view, shift, axis=0)
                        jitter = np.random.uniform(0.98, 1.02)
                        g_view = g_view * jitter
                        l_view = l_view * jitter
                # Compute morphology features and extend scalars
                morph = compute_morphological_features(g_view)
                scalars_ext = np.concatenate([scalars, morph]).astype(np.float32)
                X_global.append(g_view)
                X_local.append(l_view)
                X_scalar.append(scalars_ext)
                y.append(label)
            except Exception:
                # Skip unreadable sample
                continue
        return (np.array(X_global), np.array(X_local), np.array(X_scalar)), np.array(y)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)


# -----------------------------
# Model Architecture
# -----------------------------
def squeeze_excite_block(x: tf.Tensor, ratio: int = 8) -> tf.Tensor:
    """Squeeze‑and‑Excitation block for 1D inputs.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    ratio : int
        Reduction ratio for the SE block.

    Returns
    -------
    tf.Tensor
        Output tensor after SE recalibration.
    """
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(max(filters // ratio, 1), activation='relu')(se)
    se = layers.Dense(int(filters), activation='sigmoid')(se)
    se = layers.Reshape((1, int(filters)))(se)
    return layers.Multiply()([x, se])


def build_cnn_se_transformer(num_conv_blocks: int = 5,
                             num_transformer_blocks: int = 2,
                             conv_filters: int = 64,
                             se_ratio: int = 8,
                             transformer_head_size: int = 64,
                             num_heads: int = 4,
                             ff_dim: int = 128,
                             dropout_rate: float = 0.2,
                             l2_reg: float = 1e-6,
                             loss_type: str = 'bce') -> models.Model:
    """Build a hybrid CNN + SE + Transformer model.

    This architecture uses residual convolutional blocks with SE gating on the
    global view, a shallow CNN with SE gating on the local view, and a dense
    network on the extended scalar inputs.  Transformer encoders capture
    long‑range dependencies in the global branch.  The three branches are
    concatenated and followed by fully connected layers.  The loss can be
    binary cross‑entropy or focal loss.
    """
    # Global branch
    input_global = layers.Input(shape=IMG_SHAPE_GLOBAL, name='global_input')
    x1 = input_global
    x1 = layers.Conv1D(conv_filters, 7, padding='same', activation='relu',
                       kernel_regularizer=regularizers.l2(l2_reg))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling1D(2)(x1)
    for _ in range(num_conv_blocks):
        # Residual block
        shortcut = x1
        x1 = layers.Conv1D(conv_filters, 5, padding='same', activation='relu',
                           kernel_regularizer=regularizers.l2(l2_reg))(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Conv1D(conv_filters, 5, padding='same', activation=None,
                           kernel_regularizer=regularizers.l2(l2_reg))(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Add()([shortcut, x1])
        x1 = layers.Activation('relu')(x1)
        # SE block
        x1 = squeeze_excite_block(x1, ratio=se_ratio)
        x1 = layers.MaxPooling1D(2)(x1)
        x1 = layers.Dropout(dropout_rate)(x1)
    for _ in range(num_transformer_blocks):
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=transformer_head_size,
                                               dropout=dropout_rate)(x1, x1)
        x1 = layers.Add()([x1, attn_output])
        x1 = layers.LayerNormalization(epsilon=1e-6)(x1)
        ff = layers.Conv1D(ff_dim, 1, activation='relu')(x1)
        ff = layers.Conv1D(x1.shape[-1], 1)(ff)
        x1 = layers.Add()([x1, ff])
        x1 = layers.LayerNormalization(epsilon=1e-6)(x1)
    x1 = layers.GlobalAveragePooling1D()(x1)
    
    # Local branch
    input_local = layers.Input(shape=IMG_SHAPE_LOCAL, name='local_input')
    x2 = input_local
    for i in range(3):
        filters = 32 if i == 0 else 64
        x2 = layers.Conv1D(filters, 3, padding='same', activation='relu',
                           kernel_regularizer=regularizers.l2(l2_reg))(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.MaxPooling1D(2)(x2)
        # SE gating on local branch
        x2 = squeeze_excite_block(x2, ratio=se_ratio)
        x2 = layers.Dropout(dropout_rate)(x2)
    x2 = layers.GlobalMaxPooling1D()(x2)
    
    # Scalar branch
    input_scalar = layers.Input(shape=(SCALAR_DIM,), name='scalar_input')
    x3 = layers.BatchNormalization()(input_scalar)
    x3 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x3)
    x3 = layers.Dropout(dropout_rate)(x3)
    x3 = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x3)
    x3 = layers.Dropout(dropout_rate)(x3)
    
    # Fusion
    concatenated = layers.Concatenate()([x1, x2, x3])
    fusion = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), name='fusion_layer')(concatenated)
    fusion = layers.Dropout(dropout_rate)(fusion)
    output = layers.Dense(1, activation='sigmoid', name='output')(fusion)
    
    model = models.Model(inputs=[input_global, input_local, input_scalar], outputs=output)
    # Loss selection
    if loss_type == 'focal':
        loss = FocalLoss(gamma=2.0, alpha=0.25)
    else:
        loss = 'binary_crossentropy'
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss=loss,
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')])
    return model


# -----------------------------
# Helper Functions
# -----------------------------
def extract_features(model: models.Model, generator: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embedding features from the fusion layer for all samples in a generator.

    The generator must be configured with `shuffle=False` and should skip
    corrupt or unreadable samples.  Synthetic data are ignored for feature
    extraction since embeddings are derived from the CNN only on real data.
    """
    extractor = models.Model(inputs=model.input, outputs=model.get_layer('fusion_layer').output)
    embeddings = []
    scalars = []
    labels = []
    for (X_g, X_l, X_s), y_batch in generator:
        emb = extractor.predict_on_batch((X_g, X_l, X_s))
        embeddings.append(emb)
        scalars.append(X_s)
        labels.append(y_batch)
    embeddings_arr = np.vstack(embeddings)
    scalars_arr = np.vstack(scalars)
    labels_arr = np.concatenate(labels)
    features = np.hstack([scalars_arr, embeddings_arr])
    return features, labels_arr


def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray, n_iter: int = 10) -> dict:
    """Run a randomised hyperparameter search over XGBoost parameters.

    Returns the best parameters found.  If xgboost is not available,
    returns an empty dictionary.  Search ranges include depth, min child
    weight, subsample, colsample and learning rate.
    """
    if not XGB_AVAILABLE:
        return {}
    param_dist = {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
    }
    clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', use_label_encoder=False)
    search = RandomizedSearchCV(clf, param_dist, n_iter=n_iter, scoring='roc_auc', cv=3, verbose=0, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_params_


def evaluate_thresholds(y_true: np.ndarray, y_pred_prob: np.ndarray, model_name: str, fold: str = '') -> Tuple[float, float]:
    """Compute precision–recall curve and return the threshold maximising F1.

    Plots the PR curve and saves it to disk.  Returns the best threshold and
    its corresponding F1 score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-7)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, label=f'{model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision–Recall Curve {model_name} {fold}')
    plt.grid(True)
    plt.legend()
    out_path = os.path.join(DATA_DIR, f'pr_curve_{model_name}{("_" + fold) if fold else ""}.png')
    plt.savefig(out_path)
    plt.close()
    return float(best_thresh), float(best_f1)


def validate_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Scan data directory and return arrays of valid files and labels.

    Files with missing keys, corrupt archives or shape mismatches are
    skipped.  A report of invalid files is saved to disk.
    """
    all_files = glob.glob(os.path.join(data_dir, "*.npz"))
    valid_files: List[str] = []
    valid_labels: List[int] = []
    corrupt_files: List[str] = []
    missing_label: List[str] = []
    shape_mismatch: List[str] = []
    for f in all_files:
        try:
            with np.load(f) as data:
                if 'label' not in data or 'global_view' not in data or 'local_view' not in data or 'scalars' not in data:
                    missing_label.append(f)
                    continue
                g_view = data['global_view']
                l_view = data['local_view']
                if g_view.shape != IMG_SHAPE_GLOBAL or l_view.shape != IMG_SHAPE_LOCAL:
                    shape_mismatch.append(f)
                    continue
                valid_files.append(f)
                valid_labels.append(int(data['label']))
        except Exception:
            corrupt_files.append(f)
    # Save report
    if len(corrupt_files) > 0:
        with open(os.path.join(data_dir, 'corrupt_files.txt'), 'w') as cf:
            for cfname in corrupt_files:
                cf.write(f'{cfname}\n')
    with open(os.path.join(data_dir, 'valid_file_index.txt'), 'w') as vf:
        for fname in valid_files:
            vf.write(f'{fname}\n')
    print(f"[INFO] Dataset validation complete:\n"
          f"  Valid samples: {len(valid_files)}\n"
          f"  Corrupt/unreadable: {len(corrupt_files)}\n"
          f"  Missing label or keys: {len(missing_label)}\n"
          f"  Shape mismatch: {len(shape_mismatch)}")
    return np.array(valid_files), np.array(valid_labels)


# -----------------------------
# Main training function
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Optimised exoplanet detection pipeline (NASA style)")
    parser.add_argument('--validate_data_only', action='store_true', help='Only validate data and exit')
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'weighted_bce', 'focal'], help='Loss function type for CNN')
    parser.add_argument('--oversample', action='store_true', help='Enable oversampling of minority class')
    parser.add_argument('--augment_positive_only', action='store_true', help='Apply augmentations only to positive samples')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--num_conv_blocks', type=int, default=5, help='Number of residual blocks in the global branch')
    parser.add_argument('--num_transformer_blocks', type=int, default=2, help='Number of transformer encoder blocks')
    parser.add_argument('--conv_filters', type=int, default=64, help='Number of filters in the global branch convolutions')
    parser.add_argument('--folds', type=int, default=5, help='Number of stratified K‑folds')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--synthetic_ratio', type=float, default=0.0, help='Ratio of synthetic positive samples relative to real positive count')
    parser.add_argument('--hybrid', action='store_true', help='Train hybrid CNN + XGBoost model')
    parser.add_argument('--ensemble_method', type=str, default='average', choices=['average', 'weighted_average', 'stacking', 'cnn_only', 'xgboost_only'], help='Method to combine CNN and XGBoost')
    parser.add_argument('--meta_learner', type=str, default='logistic', choices=['logistic', 'ridge'], help='Type of meta learner for stacking')
    parser.add_argument('--tune_xgboost', action='store_true', help='Tune XGBoost hyperparameters')
    parser.add_argument('--xgb_rounds', type=int, default=100, help='Number of boosting rounds for XGBoost')
    parser.add_argument('--xgb_early_stopping', type=int, default=10, help='Early stopping rounds for XGBoost')
    parser.add_argument('--xgb_max_depth', type=int, default=6, help='Max depth for XGBoost trees')
    parser.add_argument('--xgb_eta', type=float, default=0.1, help='Learning rate for XGBoost')
    parser.add_argument('--cnn_weight', type=float, default=0.5, help='Weight for CNN predictions in weighted average')
    parser.add_argument('--xgb_weight', type=float, default=0.5, help='Weight for XGBoost predictions in weighted average')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Abort if hybrid requested but XGBoost not available
    if args.hybrid and not XGB_AVAILABLE:
        print("[ERROR] XGBoost is not installed.  Install it or disable --hybrid.")
        return

    # Set seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print(f"[INFO] Configuration: {args}")

    # Validate data
    print("[INFO] Indexing and validating files...")
    valid_files, valid_labels = validate_dataset(DATA_DIR)
    if args.validate_data_only:
        return
    print(f"[INFO] Proceeding with {len(valid_files)} valid samples.")

    # Generate synthetic data if requested
    synthetic_data = None
    if args.synthetic_ratio > 0:
        pos_count = np.sum(valid_labels == 1)
        syn_count = int(pos_count * args.synthetic_ratio)
        if syn_count > 0:
            print(f"[INFO] Generating {syn_count} synthetic transit samples...")
            synthetic_data = generate_synthetic_transits(num_samples=syn_count)

    # Cross‑validation
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_cnn = np.zeros(len(valid_files))
    oof_xgb = np.zeros(len(valid_files))
    oof_y = np.zeros(len(valid_files))

    fold_num = 1
    for train_idx, val_idx in skf.split(valid_files, valid_labels):
        print(f"\n[INFO] Starting Fold {fold_num}/{args.folds}")
        X_train_paths, y_train = valid_files[train_idx], valid_labels[train_idx]
        X_val_paths, y_val = valid_files[val_idx], valid_labels[val_idx]
        # Class weights
        cw = None
        if args.loss == 'weighted_bce' and not args.oversample:
            weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            cw = dict(enumerate(weights))
        # Generators
        train_gen = ExoplanetDataGenerator(
            X_train_paths, y_train,
            batch_size=32,
            shuffle=True,
            augment=args.augment,
            augment_positive_only=args.augment_positive_only,
            oversample=args.oversample,
            synthetic_data=synthetic_data,
            synthetic_ratio=args.synthetic_ratio
        )
        val_gen = ExoplanetDataGenerator(
            X_val_paths, y_val,
            batch_size=32,
            shuffle=False,
            augment=False
        )
        # Build model
        cnn_model = build_cnn_se_transformer(
            num_conv_blocks=args.num_conv_blocks,
            num_transformer_blocks=args.num_transformer_blocks,
            conv_filters=args.conv_filters,
            se_ratio=8,
            transformer_head_size=64,
            num_heads=4,
            ff_dim=128,
            dropout_rate=0.2,
            l2_reg=1e-6,
            loss_type=args.loss
        )
        cb_list = [
            callbacks.EarlyStopping(monitor='val_recall', mode='max', patience=10, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ]
        cnn_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.epochs,
            callbacks=cb_list,
            class_weight=cw,
            verbose=1
        )
        # Predict on validation set
        def predict_on_files(paths: np.ndarray) -> np.ndarray:
            preds: List[float] = []
            chunk = 64
            for i in range(0, len(paths), chunk):
                batch_paths = paths[i:i + chunk]
                X_g, X_l, X_s = [], [], []
                for p in batch_paths:
                    try:
                        with np.load(p) as d:
                            g_view = d['global_view']
                            l_view = d['local_view']
                            scalars = d['scalars']
                            if g_view.shape != IMG_SHAPE_GLOBAL or l_view.shape != IMG_SHAPE_LOCAL:
                                continue
                            morph = compute_morphological_features(g_view)
                            scalars_ext = np.concatenate([scalars, morph]).astype(np.float32)
                            X_g.append(g_view)
                            X_l.append(l_view)
                            X_s.append(scalars_ext)
                    except Exception:
                        continue
                if len(X_g) == 0:
                    continue
                X_g_arr = np.array(X_g)
                X_l_arr = np.array(X_l)
                X_s_arr = np.array(X_s)
                preds.extend(cnn_model.predict_on_batch((X_g_arr, X_l_arr, X_s_arr)).flatten())
            return np.array(preds)
        cnn_preds_fold = predict_on_files(X_val_paths)
        oof_cnn[val_idx] = cnn_preds_fold
        oof_y[val_idx] = y_val
        # Hybrid: train XGBoost on embeddings
        if args.hybrid:
            X_train_feats, y_train_feats = extract_features(cnn_model, train_gen)
            # Extract features on full validation set
            def extract_features_full(paths: np.ndarray) -> np.ndarray:
                feats: List[np.ndarray] = []
                chunk = 64
                extractor = models.Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('fusion_layer').output)
                for i in range(0, len(paths), chunk):
                    batch_paths = paths[i:i + chunk]
                    X_g, X_l, X_s = [], [], []
                    for p in batch_paths:
                        try:
                            with np.load(p) as d:
                                g_view = d['global_view']
                                l_view = d['local_view']
                                scalars = d['scalars']
                                if g_view.shape != IMG_SHAPE_GLOBAL or l_view.shape != IMG_SHAPE_LOCAL:
                                    continue
                                morph = compute_morphological_features(g_view)
                                scalars_ext = np.concatenate([scalars, morph]).astype(np.float32)
                                X_g.append(g_view)
                                X_l.append(l_view)
                                X_s.append(scalars_ext)
                        except Exception:
                            continue
                    if len(X_g) == 0:
                        continue
                    X_g_arr = np.array(X_g)
                    X_l_arr = np.array(X_l)
                    X_s_arr = np.array(X_s)
                    emb = extractor.predict_on_batch((X_g_arr, X_l_arr, X_s_arr))
                    feats.append(np.hstack([X_s_arr, emb]))
                return np.vstack(feats)
            X_val_feats = extract_features_full(X_val_paths)
            # Tune XGBoost if requested
            xgb_params = {
                'max_depth': args.xgb_max_depth,
                'eta': args.xgb_eta,
                'objective': 'binary:logistic',
                'eval_metric': 'auc'
            }
            if args.tune_xgboost:
                best_params = tune_xgboost(X_train_feats, y_train_feats)
                xgb_params.update(best_params)
            dtrain = xgb.DMatrix(X_train_feats, label=y_train_feats)  # type: ignore
            dval = xgb.DMatrix(X_val_feats, label=y_val)  # type: ignore
            booster = xgb.train(xgb_params, dtrain, num_boost_round=args.xgb_rounds,
                                evals=[(dval, 'eval')], early_stopping_rounds=args.xgb_early_stopping,
                                verbose_eval=False)  # type: ignore
            xgb_preds_fold = booster.predict(dval)  # type: ignore
            oof_xgb[val_idx] = xgb_preds_fold
        fold_num += 1

    # Ensemble and evaluation
    print("\n[INFO] Training meta‑learner and evaluating ensemble...")
    if not args.hybrid or args.ensemble_method == 'cnn_only':
        final_probs = oof_cnn
    elif args.ensemble_method == 'xgboost_only':
        final_probs = oof_xgb
    elif args.ensemble_method == 'average':
        final_probs = (oof_cnn + oof_xgb) / 2.0
    elif args.ensemble_method == 'weighted_average':
        final_probs = (args.cnn_weight * oof_cnn + args.xgb_weight * oof_xgb) / (args.cnn_weight + args.xgb_weight)
    elif args.ensemble_method == 'stacking':
        if args.meta_learner == 'logistic':
            meta = LogisticRegression(max_iter=1000)
            final_probs = cross_val_predict(meta, np.column_stack((oof_cnn, oof_xgb)), oof_y, cv=args.folds, method='predict_proba')[:, 1]
            meta.fit(np.column_stack((oof_cnn, oof_xgb)), oof_y)
            print(f"[INFO] Meta‑learner coefficients: {meta.coef_}")
        else:
            meta = RidgeClassifier()
            scores = cross_val_predict(meta, np.column_stack((oof_cnn, oof_xgb)), oof_y, cv=args.folds, method='decision_function')
            final_probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-7)
            meta.fit(np.column_stack((oof_cnn, oof_xgb)), oof_y)
            print(f"[INFO] Meta‑learner coefficients: {meta.coef_}")
    else:
        final_probs = oof_cnn
    # Final threshold selection and reporting
    best_thresh, best_f1 = evaluate_thresholds(oof_y, final_probs, model_name='Ensemble')
    print(f"[INFO] Best threshold: {best_thresh:.2f}, F1: {best_f1:.4f}")
    y_pred_final = (final_probs >= best_thresh).astype(int)
    print(classification_report(oof_y, y_pred_final, target_names=['False Positive', 'Planet']))
    cm = confusion_matrix(oof_y, y_pred_final)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Aggregate Confusion Matrix (OOF)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(DATA_DIR, 'confusion_matrix_optimal_final.png'))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_exominer_balanced.py
==========================

BALANCED Exoplanet Detection Model
----------------------------------

This model finds the sweet spot between:
- TOO SIMPLE (10K params, underfitting, PR-AUC ~0.13)
- TOO COMPLEX (180K params, overfitting, Train 0.99 vs Val 0.52)

Target: ~40K parameters, proper regularization, realistic PR-AUC ~0.40-0.60

Key Design Principles (from NASA ExoMiner paper):
1. Multi-scale convolutions to capture different transit durations
2. Proper class weighting (NOT balanced batching)
3. Moderate dropout (0.4, not 0.6)
4. L2 regularization but not extreme
5. Batch normalization for stable training

Realistic Expectations:
- With 975 positives, we cannot match NASA's 0.90 PR-AUC
- Target: PR-AUC 0.40-0.60 (realistic for this dataset size)
- This would still be a GOOD result for an IEEE paper

Author: Exoplanet AI Research Team
Date: January 2026
"""

import os
import sys
import glob
import json
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, regularizers, initializers
from tensorflow.keras.utils import Sequence

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix, f1_score, roc_curve
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Set Seeds
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# =============================================================================
# CONFIGURATION - BALANCED (Not too simple, not too complex)
# =============================================================================

@dataclass
class Config:
    """Balanced configuration for realistic performance without overfitting."""
    
    # Input shapes (matching your .npz files)
    global_shape: Tuple[int, int] = (2001, 1)
    local_shape: Tuple[int, int] = (201, 1)
    scalar_shape: Tuple[int,] = (7,)
    
    # Architecture - MODERATE SIZE (~40K params)
    cnn_filters: int = 32           # Moderate (not 16, not 64)
    dense_units: int = 64           # Moderate
    
    # Regularization - MODERATE
    dropout_rate: float = 0.4       # Moderate (not 0.6, not 0.3)
    l2_rate: float = 1e-3           # Moderate regularization
    
    # Training
    batch_size: int = 32
    epochs: int = 80
    learning_rate: float = 3e-4     # Moderate LR
    warmup_epochs: int = 5
    
    # Early stopping
    patience: int = 12
    min_delta: float = 0.001
    
    # Class imbalance - use LOSS WEIGHTING, not balanced batching
    pos_weight: float = 10.0        # Penalize missing positives 10x more
    
    # Augmentation
    augment_prob: float = 0.5


# =============================================================================
# SQUEEZE-AND-EXCITATION BLOCK (From ExoMiner)
# =============================================================================

class SqueezeExcitation(layers.Layer):
    """Channel attention mechanism - key component of ExoMiner."""
    
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        filters = input_shape[-1]
        self.squeeze = layers.GlobalAveragePooling1D()
        self.excite = keras.Sequential([
            layers.Dense(max(1, filters // self.ratio), activation='relu'),
            layers.Dense(filters, activation='sigmoid')
        ])
        super().build(input_shape)

    def call(self, x):
        se = self.squeeze(x)
        se = self.excite(se)
        se = tf.expand_dims(se, axis=1)
        return x * se
    
    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


# =============================================================================
# DATA GENERATOR - MAINTAINS ORIGINAL DISTRIBUTION
# =============================================================================

class ExoplanetDataGenerator(Sequence):
    """
    Data generator that maintains ORIGINAL class distribution.
    Uses sample weights in loss function for class imbalance.
    """
    
    def __init__(
        self,
        file_paths: np.ndarray,
        labels: np.ndarray,
        config: Config,
        augment: bool = False,
        shuffle: bool = True
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.config = config
        self.augment = augment
        self.shuffle = shuffle
        
        self.n_samples = len(file_paths)
        self.indices = np.arange(self.n_samples)
        
        self.on_epoch_end()
        
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        print(f"  Samples: {n_pos} pos ({100*n_pos/len(labels):.1f}%), {n_neg} neg")
    
    def __len__(self):
        return int(np.ceil(self.n_samples / self.config.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.config.batch_size : (index + 1) * self.config.batch_size
        ]
        
        X_global, X_local, X_scalar, Y = [], [], [], []
        
        for i in batch_indices:
            data = self._load_file(self.file_paths[i])
            if data is not None:
                # Apply augmentation with probability
                if self.augment and np.random.rand() < self.config.augment_prob:
                    data = self._augment(data, self.labels[i])
                
                X_global.append(data['global'])
                X_local.append(data['local'])
                X_scalar.append(data['scalar'])
                Y.append(self.labels[i])
        
        return (
            {
                "global_input": np.array(X_global),
                "local_input": np.array(X_local),
                "scalar_input": np.array(X_scalar)
            },
            np.array(Y)
        )
    
    def _load_file(self, path):
        try:
            with np.load(path) as data:
                g = data['global_view'].astype(np.float32)
                l = data['local_view'].astype(np.float32)
                s = data['scalars'].astype(np.float32)
            
            if g.ndim == 1:
                g = g.reshape(-1, 1)
            if l.ndim == 1:
                l = l.reshape(-1, 1)
            
            if g.shape[0] != self.config.global_shape[0]:
                g = self._resize(g, self.config.global_shape[0])
            if l.shape[0] != self.config.local_shape[0]:
                l = self._resize(l, self.config.local_shape[0])
            
            s_padded = np.zeros(self.config.scalar_shape[0], dtype=np.float32)
            s_len = min(len(s), self.config.scalar_shape[0])
            s_padded[:s_len] = s[:s_len]
            s_padded = np.nan_to_num(s_padded, nan=0.0)
            
            return {'global': g, 'local': l, 'scalar': s_padded}
            
        except Exception:
            return None
    
    def _resize(self, arr, target_len):
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, arr.flatten()).reshape(-1, 1)
    
    def _augment(self, data, label):
        """Physics-aware augmentation - MORE for positives."""
        g = data['global'].copy()
        l = data['local'].copy()
        s = data['scalar'].copy()
        
        # 1. Time reversal (50% chance)
        if np.random.rand() > 0.5:
            g = np.flip(g, axis=0).copy()
            l = np.flip(l, axis=0).copy()
        
        # 2. Phase shift
        shift = np.random.randint(-30, 31)
        g = np.roll(g, shift, axis=0)
        
        # 3. Flux scaling
        scale = np.random.uniform(0.95, 1.05)
        g = g * scale
        l = l * scale
        
        # 4. Gaussian noise (more for positives)
        noise_level = np.random.uniform(0.005, 0.02) if label == 1 else np.random.uniform(0.002, 0.01)
        g = g + np.random.normal(0, noise_level, g.shape)
        l = l + np.random.normal(0, noise_level, l.shape)
        
        # 5. Transit depth variation (only for positives)
        if label == 1:
            depth_scale = np.random.uniform(0.85, 1.15)
            g = 1 + (g - 1) * depth_scale
            l = 1 + (l - 1) * depth_scale
        
        return {
            'global': g.astype(np.float32),
            'local': l.astype(np.float32),
            'scalar': s
        }
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# =============================================================================
# MODEL ARCHITECTURE - BALANCED (~40K PARAMS)
# =============================================================================

def build_balanced_model(config: Config, pos_ratio: float = 0.08):
    """
    Balanced ExoMiner-inspired model.
    
    Architecture:
    - Multi-scale CNN for global view (captures different transit durations)
    - Simple CNN for local view (transit shape)
    - Dense network for scalars
    - Feature fusion with moderate regularization
    
    ~40K parameters - enough capacity to learn, not enough to memorize
    """
    l2 = regularizers.l2(config.l2_rate)
    drop = config.dropout_rate
    
    # Output bias for imbalanced data
    output_bias = math.log(pos_ratio / (1 - pos_ratio))
    
    # =========================================================================
    # BRANCH 1: GLOBAL VIEW (Multi-scale CNN)
    # =========================================================================
    input_global = layers.Input(shape=config.global_shape, name="global_input")
    
    # Multi-scale convolutions (like ExoMiner) - different kernel sizes capture
    # different transit durations
    conv_3 = layers.Conv1D(config.cnn_filters, 3, padding='same', activation='relu',
                           kernel_regularizer=l2)(input_global)
    conv_7 = layers.Conv1D(config.cnn_filters, 7, padding='same', activation='relu',
                           kernel_regularizer=l2)(input_global)
    conv_15 = layers.Conv1D(config.cnn_filters, 15, padding='same', activation='relu',
                            kernel_regularizer=l2)(input_global)
    
    # Concatenate multi-scale features
    x1 = layers.Concatenate()([conv_3, conv_7, conv_15])
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling1D(4)(x1)
    x1 = layers.Dropout(drop)(x1)
    
    # Second conv layer with SE attention
    x1 = layers.Conv1D(config.cnn_filters * 2, 5, padding='same', activation='relu',
                       kernel_regularizer=l2)(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = SqueezeExcitation(ratio=4)(x1)
    x1 = layers.MaxPooling1D(4)(x1)
    x1 = layers.Dropout(drop)(x1)
    
    # Third conv layer
    x1 = layers.Conv1D(config.cnn_filters * 2, 3, padding='same', activation='relu',
                       kernel_regularizer=l2)(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.GlobalAveragePooling1D()(x1)
    
    # =========================================================================
    # BRANCH 2: LOCAL VIEW (Transit shape CNN)
    # =========================================================================
    input_local = layers.Input(shape=config.local_shape, name="local_input")
    
    x2 = layers.Conv1D(config.cnn_filters, 5, padding='same', activation='relu',
                       kernel_regularizer=l2)(input_local)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling1D(2)(x2)
    x2 = layers.Dropout(drop)(x2)
    
    x2 = layers.Conv1D(config.cnn_filters * 2, 3, padding='same', activation='relu',
                       kernel_regularizer=l2)(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = SqueezeExcitation(ratio=4)(x2)
    x2 = layers.MaxPooling1D(2)(x2)
    x2 = layers.Dropout(drop)(x2)
    
    x2 = layers.Conv1D(config.cnn_filters * 2, 3, padding='same', activation='relu',
                       kernel_regularizer=l2)(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.GlobalAveragePooling1D()(x2)
    
    # =========================================================================
    # BRANCH 3: SCALAR FEATURES
    # =========================================================================
    input_scalar = layers.Input(shape=config.scalar_shape, name="scalar_input")
    
    x3 = layers.BatchNormalization()(input_scalar)
    x3 = layers.Dense(32, activation='relu', kernel_regularizer=l2)(x3)
    x3 = layers.Dropout(drop)(x3)
    x3 = layers.Dense(16, activation='relu', kernel_regularizer=l2)(x3)
    
    # =========================================================================
    # FUSION
    # =========================================================================
    fusion = layers.Concatenate()([x1, x2, x3])
    
    fusion = layers.Dense(config.dense_units, activation='relu', kernel_regularizer=l2)(fusion)
    fusion = layers.BatchNormalization()(fusion)
    fusion = layers.Dropout(drop)(fusion)
    
    fusion = layers.Dense(32, activation='relu', kernel_regularizer=l2)(fusion)
    fusion = layers.Dropout(drop * 0.5)(fusion)  # Lighter dropout before output
    
    # OUTPUT
    output = layers.Dense(
        1,
        activation='sigmoid',
        bias_initializer=initializers.Constant(output_bias),
        kernel_regularizer=l2,
        name="output"
    )(fusion)
    
    model = models.Model(
        inputs=[input_global, input_local, input_scalar],
        outputs=output,
        name="ExoMiner_Balanced"
    )
    
    return model


# =============================================================================
# FOCAL LOSS (Better for imbalanced data than BCE)
# =============================================================================

def focal_loss(gamma=2.0, alpha=0.75):
    """
    Focal Loss - focuses on hard examples, ignores easy negatives.
    
    alpha: weight for positive class (>0.5 means more weight on positives)
    gamma: focusing parameter (higher = more focus on hard examples)
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Cross entropy
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Focal weight
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, gamma)
        
        # Alpha weight
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        return tf.reduce_mean(alpha_weight * focal_weight * ce)
    
    return loss_fn


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Warmup followed by cosine decay."""
    
    def __init__(self, peak_lr, warmup_steps, total_steps):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Linear warmup
        warmup_lr = self.peak_lr * (step / self.warmup_steps)
        
        # Cosine decay
        decay_steps = self.total_steps - self.warmup_steps
        decay_step = step - self.warmup_steps
        cosine_decay = 0.5 * (1 + tf.cos(3.14159 * decay_step / decay_steps))
        decay_lr = self.peak_lr * cosine_decay
        
        return tf.where(step < self.warmup_steps, warmup_lr, decay_lr)
    
    def get_config(self):
        return {
            "peak_lr": float(self.peak_lr),
            "warmup_steps": int(self.warmup_steps),
            "total_steps": int(self.total_steps)
        }


# =============================================================================
# METRICS CALLBACK
# =============================================================================

class MetricsCallback(keras.callbacks.Callback):
    """Print clean metrics summary at end of each epoch."""
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        
        train_auc = logs.get('auc', 0)
        val_auc = logs.get('val_auc', 0)
        train_pr = logs.get('pr_auc', 0)
        val_pr = logs.get('val_pr_auc', 0)
        
        gap_auc = train_auc - val_auc
        gap_pr = train_pr - val_pr
        
        # Color code the gap
        gap_status = "OK" if abs(gap_auc) < 0.15 else "WARNING" if abs(gap_auc) < 0.25 else "OVERFITTING"
        
        print(f"\n  [Epoch {epoch+1}] AUC: {train_auc:.3f}/{val_auc:.3f} | PR-AUC: {train_pr:.3f}/{val_pr:.3f} | Gap: {gap_auc:.3f} ({gap_status})")


# =============================================================================
# EVALUATION
# =============================================================================

def comprehensive_evaluation(y_true, y_pred, output_dir, fold=None):
    """Generate comprehensive metrics and plots."""
    
    # Basic metrics
    auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    
    # Recall at high precision
    recall_at_p90 = recall[precision >= 0.90].max() if (precision >= 0.90).any() else 0
    recall_at_p95 = recall[precision >= 0.95].max() if (precision >= 0.95).any() else 0
    recall_at_p99 = recall[precision >= 0.99].max() if (precision >= 0.99).any() else 0
    
    results = {
        'auc': float(auc),
        'pr_auc': float(pr_auc),
        'best_f1': float(best_f1),
        'best_threshold': float(best_threshold),
        'best_precision': float(best_precision),
        'best_recall': float(best_recall),
        'recall_at_p90': float(recall_at_p90),
        'recall_at_p95': float(recall_at_p95),
        'recall_at_p99': float(recall_at_p99)
    }
    
    return results


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_model(data_dir: str, output_dir: str):
    """Main training function."""
    
    config = Config()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"balanced_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("BALANCED EXOMINER MODEL")
    print("="*70)
    print(f"Architecture: Multi-scale CNN + SE-Attention (~40K params)")
    print(f"Regularization: Dropout={config.dropout_rate}, L2={config.l2_rate}")
    print(f"Loss: Focal Loss (alpha=0.75, gamma=2.0)")
    print(f"Output: {output_path}")
    print("="*70)
    
    # Load and index data
    print("\nLoading dataset...")
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    
    X_files, Y = [], []
    for f in tqdm(npz_files, desc="Indexing"):
        try:
            with np.load(f) as data:
                label = int(data['label'])
            X_files.append(f)
            Y.append(label)
        except:
            pass
    
    X_files = np.array(X_files)
    Y = np.array(Y)
    
    n_pos = np.sum(Y == 1)
    n_neg = np.sum(Y == 0)
    pos_ratio = n_pos / len(Y)
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(X_files)}")
    print(f"  Positives: {n_pos} ({pos_ratio*100:.1f}%)")
    print(f"  Negatives: {n_neg} ({(1-pos_ratio)*100:.1f}%)")
    print(f"  Imbalance ratio: 1:{n_neg/n_pos:.1f}")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    all_val_preds = []
    all_val_labels = []
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_files, Y)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1}/5")
        print(f"{'='*70}")
        
        X_train, X_val = X_files[train_idx], X_files[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]
        
        # Create generators
        print("\nCreating data generators...")
        train_gen = ExoplanetDataGenerator(
            X_train, y_train, config,
            augment=True, shuffle=True
        )
        
        val_gen = ExoplanetDataGenerator(
            X_val, y_val, config,
            augment=False, shuffle=False
        )
        
        # Build model
        print("\nBuilding model...")
        model = build_balanced_model(config, pos_ratio)
        
        if fold == 0:
            model.summary()
            print(f"\nTotal parameters: {model.count_params():,}")
        
        # Learning rate schedule
        steps_per_epoch = len(train_gen)
        total_steps = steps_per_epoch * config.epochs
        warmup_steps = steps_per_epoch * config.warmup_epochs
        
        lr_schedule = WarmupCosineDecay(
            config.learning_rate,
            warmup_steps,
            total_steps
        )
        
        # Compile
        model.compile(
            optimizer=optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
            loss=focal_loss(gamma=2.0, alpha=0.75),
            metrics=[
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(curve='PR', name='pr_auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        # Callbacks
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_pr_auc',
                patience=config.patience,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                output_path / f"model_fold_{fold+1}.keras",
                monitor='val_pr_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            MetricsCallback()
        ]
        
        # Train
        print("\nTraining...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.epochs,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Evaluate
        print("\nEvaluating...")
        val_preds = []
        val_labels = []
        
        for i in range(len(val_gen)):
            X_batch, y_batch = val_gen[i]
            preds = model.predict(X_batch, verbose=0)
            val_preds.extend(preds.flatten())
            val_labels.extend(y_batch)
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        # Get metrics
        fold_metrics = comprehensive_evaluation(val_labels, val_preds, output_path, fold)
        
        print(f"\n--- Fold {fold+1} Results ---")
        print(f"  AUC: {fold_metrics['auc']:.4f}")
        print(f"  PR-AUC: {fold_metrics['pr_auc']:.4f}")
        print(f"  Best F1: {fold_metrics['best_f1']:.4f} @ threshold={fold_metrics['best_threshold']:.3f}")
        print(f"  Recall@P90: {fold_metrics['recall_at_p90']:.4f}")
        print(f"  Recall@P95: {fold_metrics['recall_at_p95']:.4f}")
        
        fold_results.append(fold_metrics)
        all_val_preds.extend(val_preds)
        all_val_labels.extend(val_labels)
        
        # Clear memory
        keras.backend.clear_session()
    
    # Overall results
    all_val_preds = np.array(all_val_preds)
    all_val_labels = np.array(all_val_labels)
    
    overall_metrics = comprehensive_evaluation(all_val_labels, all_val_preds, output_path)
    
    print("\n" + "="*70)
    print("FINAL RESULTS (5-Fold Cross-Validation)")
    print("="*70)
    print(f"Overall AUC: {overall_metrics['auc']:.4f}")
    print(f"Overall PR-AUC: {overall_metrics['pr_auc']:.4f}")
    print(f"Best F1: {overall_metrics['best_f1']:.4f}")
    print(f"Recall@Precision=90%: {overall_metrics['recall_at_p90']:.4f}")
    print(f"Recall@Precision=95%: {overall_metrics['recall_at_p95']:.4f}")
    print(f"Recall@Precision=99%: {overall_metrics['recall_at_p99']:.4f}")
    
    # Comparison with benchmarks
    print("\n" + "-"*70)
    print("BENCHMARK COMPARISON")
    print("-"*70)
    print(f"Random baseline (positive ratio): PR-AUC = {pos_ratio:.3f}")
    print(f"Your model:                       PR-AUC = {overall_metrics['pr_auc']:.3f}")
    print(f"NASA ExoMiner (36K samples):      PR-AUC ~ 0.90")
    print(f"")
    print(f"Improvement over baseline: {overall_metrics['pr_auc']/pos_ratio:.1f}x")
    print("-"*70)
    
    # Save results
    results = {
        'overall': overall_metrics,
        'fold_results': fold_results,
        'config': {
            'cnn_filters': config.cnn_filters,
            'dense_units': config.dense_units,
            'dropout_rate': config.dropout_rate,
            'l2_rate': config.l2_rate,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate
        },
        'dataset': {
            'total_samples': len(X_files),
            'positives': int(n_pos),
            'negatives': int(n_neg),
            'positive_ratio': float(pos_ratio)
        }
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot PR curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(all_val_labels, all_val_preds)
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'Model (PR-AUC = {overall_metrics["pr_auc"]:.3f})')
    plt.axhline(y=pos_ratio, color='r', linestyle='--', label=f'Random (PR-AUC = {pos_ratio:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve (5-Fold CV)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(output_path / 'pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(all_val_labels, all_val_preds)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Model (AUC = {overall_metrics["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random (AUC = 0.5)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve (5-Fold CV)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    train_model(args.data_dir, args.output_dir)

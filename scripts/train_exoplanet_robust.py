#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_exoplanet_robust.py
=========================

Robust Exoplanet Detection Model with Anti-Overfitting Design
-------------------------------------------------------------

Key Anti-Overfitting Strategies:
1. SAMPLE WEIGHTING instead of balanced batching (maintains real distribution)
2. AGGRESSIVE DATA AUGMENTATION on ALL samples (not just positives)
3. REGULARIZATION: Heavy dropout (0.6), strong L2 (5e-3), weight constraints
4. ARCHITECTURE: Ultra-simple (60K params vs 180K) to prevent memorization
5. TRAINING: Very low LR, long warmup, gradient clipping
6. VALIDATION: Identical distribution to training

Target: Reduce Train-Val gap from 0.45 to < 0.15

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
from typing import Tuple, List, Dict, Any, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, regularizers, initializers
from tensorflow.keras.utils import Sequence

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix, f1_score
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
# CONFIGURATION - ULTRA CONSERVATIVE TO PREVENT OVERFITTING
# =============================================================================

@dataclass
class Config:
    """Ultra-conservative configuration for preventing overfitting."""
    
    # Input shapes
    global_shape: Tuple[int, int] = (2001, 1)
    local_shape: Tuple[int, int] = (201, 1)
    scalar_shape: Tuple[int,] = (7,)
    
    # Architecture - MINIMAL
    cnn_filters: int = 16           # Very small
    dense_units: int = 32           # Very small
    
    # Regularization - EXTREME
    dropout_rate: float = 0.6       # Very high dropout
    l2_rate: float = 5e-3           # Strong L2
    
    # Training
    batch_size: int = 64            # Larger batches for stable gradients
    epochs: int = 100               # More epochs with early stopping
    learning_rate: float = 1e-4     # Very low learning rate
    warmup_epochs: int = 10         # Long warmup
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.001
    
    # Class imbalance
    pos_weight: float = 12.0        # Weight for positive class in loss
    
    # Augmentation probability
    augment_prob: float = 0.5       # Apply augmentation 50% of time to ALL samples


# =============================================================================
# DATA GENERATOR - NO REBALANCING, USE SAMPLE WEIGHTS
# =============================================================================

class WeightedDataGenerator(Sequence):
    """
    Data generator that maintains ORIGINAL class distribution.
    Uses sample weights instead of oversampling/undersampling.
    
    This prevents the train/val distribution mismatch that causes overfitting.
    """
    
    def __init__(
        self,
        file_paths: np.ndarray,
        labels: np.ndarray,
        config: Config,
        augment: bool = False,
        shuffle: bool = True,
        pos_weight: float = 12.0
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.config = config
        self.augment = augment
        self.shuffle = shuffle
        self.pos_weight = pos_weight
        
        self.n_samples = len(file_paths)
        self.indices = np.arange(self.n_samples)
        
        # Pre-compute sample weights (higher for rare positive class)
        self.sample_weights = np.where(labels == 1, pos_weight, 1.0)
        
        self.on_epoch_end()
        
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        print(f"Generator: {n_pos} pos ({100*n_pos/len(labels):.1f}%), {n_neg} neg")
        print(f"Sample weights: pos={pos_weight:.1f}, neg=1.0")
    
    def __len__(self):
        return int(np.ceil(self.n_samples / self.config.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.config.batch_size : (index + 1) * self.config.batch_size
        ]
        
        X_global, X_local, X_scalar, Y, W = [], [], [], [], []
        
        for i in batch_indices:
            data = self._load_file(self.file_paths[i])
            if data is not None:
                # Apply augmentation to ALL samples (not just positives)
                if self.augment and np.random.rand() < self.config.augment_prob:
                    data = self._augment(data)
                
                X_global.append(data['global'])
                X_local.append(data['local'])
                X_scalar.append(data['scalar'])
                Y.append(self.labels[i])
                W.append(self.sample_weights[i])
        
        return (
            {
                "global_input": np.array(X_global),
                "local_input": np.array(X_local),
                "scalar_input": np.array(X_scalar)
            },
            np.array(Y),
            np.array(W)  # Sample weights
        )
    
    def _load_file(self, path):
        try:
            with np.load(path) as data:
                g = data['global_view'].astype(np.float32)
                l = data['local_view'].astype(np.float32)
                s = data['scalars'].astype(np.float32)
            
            # Ensure correct shapes
            if g.ndim == 1:
                g = g.reshape(-1, 1)
            if l.ndim == 1:
                l = l.reshape(-1, 1)
            
            # Resize if needed
            if g.shape[0] != self.config.global_shape[0]:
                g = self._resize(g, self.config.global_shape[0])
            if l.shape[0] != self.config.local_shape[0]:
                l = self._resize(l, self.config.local_shape[0])
            
            # Pad scalars
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
    
    def _augment(self, data):
        """Augmentation applied to ALL samples (not just positives)."""
        g = data['global'].copy()
        l = data['local'].copy()
        s = data['scalar'].copy()
        
        # 1. Time reversal (50% chance)
        if np.random.rand() > 0.5:
            g = np.flip(g, axis=0).copy()
            l = np.flip(l, axis=0).copy()
        
        # 2. Phase shift
        shift = np.random.randint(-100, 101)
        g = np.roll(g, shift, axis=0)
        
        # 3. Flux scaling
        scale = np.random.uniform(0.9, 1.1)
        g = g * scale
        l = l * scale
        
        # 4. Gaussian noise
        noise_level = np.random.uniform(0.01, 0.05)
        g = g + np.random.normal(0, noise_level, g.shape)
        l = l + np.random.normal(0, noise_level, l.shape)
        
        # 5. Baseline drift
        drift = np.linspace(0, np.random.uniform(-0.02, 0.02), len(g)).reshape(-1, 1)
        g = g + drift
        
        return {
            'global': g.astype(np.float32), 
            'local': l.astype(np.float32), 
            'scalar': s
        }
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# =============================================================================
# MODEL - ULTRA SIMPLE ARCHITECTURE (~60K PARAMS)
# =============================================================================

def build_simple_model(config: Config, pos_ratio: float = 0.08):
    """
    Ultra-simple model designed to NOT overfit.
    
    Key design principles:
    1. Minimal parameters (~60K vs typical 180K+)
    2. Heavy regularization at every layer
    3. No attention mechanisms (they memorize)
    4. Simple pooling and concatenation
    """
    l2 = regularizers.l2(config.l2_rate)
    drop = config.dropout_rate
    
    # Output bias for imbalanced data
    output_bias = math.log(pos_ratio / (1 - pos_ratio))
    
    # =========================================================================
    # BRANCH 1: GLOBAL (Simple CNN - NO TRANSFORMER)
    # =========================================================================
    input_global = layers.Input(shape=config.global_shape, name="global_input")
    
    x1 = layers.Conv1D(config.cnn_filters, 16, strides=8, padding='same', 
                       activation='relu', kernel_regularizer=l2)(input_global)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(drop)(x1)
    x1 = layers.MaxPooling1D(4)(x1)
    
    x1 = layers.Conv1D(config.cnn_filters * 2, 8, padding='same', 
                       activation='relu', kernel_regularizer=l2)(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(drop)(x1)
    x1 = layers.GlobalAveragePooling1D()(x1)
    
    # =========================================================================
    # BRANCH 2: LOCAL (Simple CNN)
    # =========================================================================
    input_local = layers.Input(shape=config.local_shape, name="local_input")
    
    x2 = layers.Conv1D(config.cnn_filters, 8, padding='same', 
                       activation='relu', kernel_regularizer=l2)(input_local)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(drop)(x2)
    x2 = layers.MaxPooling1D(2)(x2)
    
    x2 = layers.Conv1D(config.cnn_filters * 2, 4, padding='same', 
                       activation='relu', kernel_regularizer=l2)(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(drop)(x2)
    x2 = layers.GlobalAveragePooling1D()(x2)
    
    # =========================================================================
    # BRANCH 3: SCALARS
    # =========================================================================
    input_scalar = layers.Input(shape=config.scalar_shape, name="scalar_input")
    
    x3 = layers.BatchNormalization()(input_scalar)
    x3 = layers.Dense(16, activation='relu', kernel_regularizer=l2)(x3)
    x3 = layers.Dropout(drop)(x3)
    
    # =========================================================================
    # FUSION - SIMPLE CONCATENATION
    # =========================================================================
    fusion = layers.Concatenate()([x1, x2, x3])
    fusion = layers.Dense(config.dense_units, activation='relu', kernel_regularizer=l2)(fusion)
    fusion = layers.BatchNormalization()(fusion)
    fusion = layers.Dropout(drop)(fusion)
    
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
        name="ExoplanetSimple"
    )
    
    return model


# =============================================================================
# WEIGHTED BINARY CROSSENTROPY LOSS
# =============================================================================

def weighted_binary_crossentropy(pos_weight=12.0):
    """
    Binary crossentropy with class weights built into the loss.
    This is more stable than sample_weight for heavily imbalanced data.
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Positive class gets higher weight
        weights = y_true * pos_weight + (1 - y_true) * 1.0
        
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        return tf.reduce_mean(weights * bce)
    
    return loss


# =============================================================================
# CUSTOM CALLBACKS
# =============================================================================

class OverfittingMonitor(keras.callbacks.Callback):
    """Monitor and alert on overfitting."""
    
    def __init__(self, threshold=0.15):
        super().__init__()
        self.threshold = threshold
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        
        train_auc = logs.get('auc', 0)
        val_auc = logs.get('val_auc', 0)
        gap = train_auc - val_auc
        
        if gap > self.threshold:
            print(f"\n  OVERFITTING ALERT: Train-Val AUC gap = {gap:.3f} (>{self.threshold})")


class PrintMetrics(keras.callbacks.Callback):
    """Print metrics in a clean format."""
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        
        train_auc = logs.get('auc', 0)
        val_auc = logs.get('val_auc', 0)
        train_pr = logs.get('pr_auc', 0)
        val_pr = logs.get('val_pr_auc', 0)
        
        gap_auc = train_auc - val_auc
        gap_pr = train_pr - val_pr
        
        print(f"\n  AUC: {train_auc:.3f} / {val_auc:.3f} (gap: {gap_auc:.3f})")
        print(f"  PR-AUC: {train_pr:.3f} / {val_pr:.3f} (gap: {gap_pr:.3f})")


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(data_dir: str, output_dir: str):
    """Main training function with anti-overfitting design."""
    
    config = Config()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"robust_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("ROBUST EXOPLANET CLASSIFIER (Anti-Overfitting Design)")
    print("="*60)
    print(f"Output: {output_path}")
    
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
    
    print(f"\nDataset: {len(X_files)} samples")
    print(f"Positives: {n_pos} ({pos_ratio*100:.1f}%)")
    print(f"Negatives: {n_neg} ({(1-pos_ratio)*100:.1f}%)")
    print(f"Imbalance: 1:{n_neg/n_pos:.1f}")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    all_val_preds = []
    all_val_labels = []
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_files, Y)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/5")
        print(f"{'='*60}")
        
        X_train, X_val = X_files[train_idx], X_files[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]
        
        # Create generators (NO rebalancing - maintain original distribution)
        train_gen = WeightedDataGenerator(
            X_train, y_train, config,
            augment=True, shuffle=True,
            pos_weight=config.pos_weight
        )
        
        val_gen = WeightedDataGenerator(
            X_val, y_val, config,
            augment=False, shuffle=False,
            pos_weight=1.0  # No weighting for validation
        )
        
        # Build model
        model = build_simple_model(config, pos_ratio)
        
        if fold == 0:
            model.summary()
        
        # Compile with weighted loss
        model.compile(
            optimizer=optimizers.AdamW(
                learning_rate=config.learning_rate,
                weight_decay=config.l2_rate,
                clipnorm=1.0  # Gradient clipping
            ),
            loss=weighted_binary_crossentropy(config.pos_weight),
            metrics=[
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(curve='PR', name='pr_auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        # Callbacks
        cb = [
            keras.callbacks.EarlyStopping(
                monitor='val_pr_auc',
                patience=config.patience,
                mode='max',
                restore_best_weights=True,
                min_delta=config.min_delta,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                output_path / f"model_fold_{fold+1}.keras",
                monitor='val_pr_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_pr_auc',
                factor=0.5,
                patience=5,
                mode='max',
                min_lr=1e-6,
                verbose=1
            ),
            OverfittingMonitor(threshold=0.15),
            PrintMetrics()
        ]
        
        # Train
        print("\nTraining...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.epochs,
            callbacks=cb,
            verbose=1
        )
        
        # Evaluate
        print("\nEvaluating...")
        val_preds = []
        val_labels = []
        
        for i in range(len(val_gen)):
            X_batch, y_batch, _ = val_gen[i]
            preds = model.predict(X_batch, verbose=0)
            val_preds.extend(preds.flatten())
            val_labels.extend(y_batch)
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        # Metrics
        auc = roc_auc_score(val_labels, val_preds)
        pr_auc = average_precision_score(val_labels, val_preds)
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(val_labels, val_preds)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        print(f"\nFold {fold+1} Results:")
        print(f"  AUC: {auc:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f}")
        print(f"  Best F1: {best_f1:.4f} @ threshold={best_threshold:.3f}")
        
        fold_results.append({
            'fold': fold + 1,
            'auc': float(auc),
            'pr_auc': float(pr_auc),
            'best_f1': float(best_f1),
            'threshold': float(best_threshold)
        })
        
        all_val_preds.extend(val_preds)
        all_val_labels.extend(val_labels)
        
        # Clear memory
        keras.backend.clear_session()
    
    # Overall results
    all_val_preds = np.array(all_val_preds)
    all_val_labels = np.array(all_val_labels)
    
    overall_auc = roc_auc_score(all_val_labels, all_val_preds)
    overall_pr_auc = average_precision_score(all_val_labels, all_val_preds)
    
    print("\n" + "="*60)
    print("FINAL RESULTS (5-Fold Cross-Validation)")
    print("="*60)
    print(f"Overall AUC: {overall_auc:.4f}")
    print(f"Overall PR-AUC: {overall_pr_auc:.4f}")
    
    avg_f1 = np.mean([r['best_f1'] for r in fold_results])
    print(f"Average F1: {avg_f1:.4f}")
    
    # Save results
    results = {
        'overall_auc': float(overall_auc),
        'overall_pr_auc': float(overall_pr_auc),
        'average_f1': float(avg_f1),
        'fold_results': fold_results,
        'config': {
            'cnn_filters': config.cnn_filters,
            'dense_units': config.dense_units,
            'dropout_rate': config.dropout_rate,
            'l2_rate': config.l2_rate,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'pos_weight': config.pos_weight
        }
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot PR curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(all_val_labels, all_val_preds)
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR-AUC = {overall_pr_auc:.3f}')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve (5-Fold CV)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    train_model(args.data_dir, args.output_dir)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_exotransformer_ultra.py
=============================

ExoTransformer-Ultra: The Final Optimal Exoplanet Detection Model
------------------------------------------------------------------

This script implements a Beyond-SOTA architecture specifically designed to
handle extreme class imbalance (~8% positive rate) in exoplanet detection.

Key Innovations:
1. STRICT BALANCED BATCHING: Every batch is forced to 50/50 planet/non-planet
2. ASYMMETRIC FOCAL LOSS: α=0.75, γ=3.0 (missing planets costs 3x more)
3. OUTPUT BIAS INITIALIZATION: Model starts knowing planets are rare
4. TRANSFORMER + SE-CNN HYBRID: Best of both architectures
5. XGBOOST STACKING: scale_pos_weight for optimal decision boundary
6. THRESHOLD OPTIMIZATION: PR-curve based, not arbitrary 0.5

Target Performance:
- PR-AUC: >0.85 (vs 0.16 in failed model)
- Recall@P=0.99: >0.70 (vs 0.001 in failed model)
- F1-Score: >0.75 (vs 0.28 in failed model)

Author: Exoplanet AI Research Team
Date: January 2026
"""

import os
import sys
import glob
import json
import logging
import argparse
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers, initializers
from tensorflow.keras.utils import Sequence

# XGBoost for HyperFusion Ensemble
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not installed. Ensemble step will use Logistic Regression.")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set Seeds for Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


# =============================================================================
# GPU CONFIGURATION FOR RTX 3050 (6GB VRAM)
# =============================================================================

def configure_gpu():
    """
    Configure GPU for optimal training on RTX 3050 with 6GB VRAM.
    
    Features:
    - Memory growth: Only allocate as needed (prevents OOM)
    - Mixed precision: FP16 computation for 2x speed + 50% less memory
    - XLA JIT compilation: Additional speedup
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth (don't pre-allocate all VRAM)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision (FP16) for RTX 3050
            # This gives ~2x speedup and uses ~50% less memory
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            print(f"\n🎮 GPU Configuration:")
            print(f"   • GPU Detected: {gpus[0].name}")
            print(f"   • Memory Growth: Enabled")
            print(f"   • Mixed Precision: FP16 (2x faster, 50% less VRAM)")
            print(f"   • Expected VRAM Usage: ~3-4GB / 6GB\n")
            
            return True
        except RuntimeError as e:
            print(f"⚠️ GPU config error: {e}")
            return False
    else:
        print("\n⚠️ No GPU detected! Training will be slow on CPU.")
        print("   Make sure CUDA and cuDNN are installed.\n")
        return False

# Configure GPU at import time
GPU_AVAILABLE = configure_gpu()


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Architecture configuration - LIGHTWEIGHT to prevent overfitting."""
    # Input shapes (matching your .npz files)
    global_shape: Tuple[int, int] = (2001, 1)
    local_shape: Tuple[int, int] = (201, 1)
    scalar_shape: Tuple[int,] = (7,)
    
    # Transformer (Global Branch) - REDUCED
    embed_dim: int = 32          # Reduced from 64
    num_heads: int = 2           # Reduced from 4
    ff_dim: int = 64             # Reduced from 128
    num_transformer_blocks: int = 1  # Reduced from 2
    
    # SE-CNN (Local Branch) - REDUCED
    cnn_filters: int = 16        # Reduced from 32
    se_ratio: int = 4            # Changed from 8
    
    # Fusion - REDUCED
    fusion_dim: int = 64         # Reduced from 128
    
    # Regularization - INCREASED HEAVILY
    dropout_rate: float = 0.5    # Increased from 0.3
    l2_rate: float = 1e-3        # Increased from 1e-4 (10x stronger)


@dataclass
class TrainingConfig:
    """Training configuration with ANTI-OVERFITTING fixes."""
    # Basic
    batch_size: int = 32         # Reduced for better generalization
    epochs: int = 50             # Reduced from 100
    folds: int = 5
    
    # Learning Rate - MUCH LOWER to prevent memorization
    peak_lr: float = 3e-4        # Reduced from 1e-3 (3x lower)
    warmup_epochs: int = 3       # Reduced warmup
    
    # IMBALANCE FIX 1: Asymmetric Focal Loss
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0     # Reduced from 3.0 (less aggressive)
    
    # IMBALANCE FIX 2: Label Smoothing - INCREASED
    label_smoothing: float = 0.1 # Increased from 0.05 (prevents overconfidence)
    
    # IMBALANCE FIX 3: XGBoost weight
    xgb_scale_pos_weight: float = 12.0
    
    # Early Stopping - MORE AGGRESSIVE
    patience: int = 10           # Reduced from 15 (stop earlier)
    
    # Augmentation multiplier for positives
    positive_augment_factor: int = 3  # Reduced from 5 (less repetition)


# =============================================================================
# 2. ASYMMETRIC FOCAL LOSS (IMBALANCE FIX)
# =============================================================================

class AsymmetricFocalLoss(keras.losses.Loss):
    """
    Focal Loss with asymmetric alpha weighting.
    
    - alpha > 0.5: Penalize missing positives MORE than false positives
    - gamma > 2.0: Focus strongly on hard-to-classify samples
    - label_smoothing: Prevent overconfident predictions
    """
    
    def __init__(self, alpha=0.75, gamma=3.0, label_smoothing=0.05, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        # Flatten
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.reshape(y_pred, [-1])
        
        # Apply label smoothing
        y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Cross entropy components
        ce_pos = -y_true * tf.math.log(y_pred)
        ce_neg = -(1 - y_true) * tf.math.log(1 - y_pred)
        
        # Focal weight
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        
        # Asymmetric alpha (positive class gets alpha, negative gets 1-alpha)
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        # Combined loss
        loss = alpha_weight * focal_weight * (ce_pos + ce_neg)
        
        return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "label_smoothing": self.label_smoothing
        })
        return config


# =============================================================================
# 3. CUSTOM LAYERS
# =============================================================================

class SqueezeExcitation(layers.Layer):
    """SE Block for channel attention in Local CNN branch."""
    
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


class TransformerBlock(layers.Layer):
    """Lightweight Transformer block for time-series."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
    
    def build(self, input_shape):
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.embed_dim,
            dropout=self.dropout_rate
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation="gelu"),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.embed_dim),
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(self.dropout_rate)
        self.drop2 = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, training=False):
        # Self-attention with residual
        attn = self.att(x, x, training=training)
        attn = self.drop1(attn, training=training)
        x = self.ln1(x + attn)
        
        # FFN with residual
        ffn = self.ffn(x)
        ffn = self.drop2(ffn, training=training)
        return self.ln2(x + ffn)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate
        })
        return config


# =============================================================================
# 4. BALANCED DATA GENERATOR (CRITICAL IMBALANCE FIX)
# =============================================================================

class BalancedDataGenerator(Sequence):
    """
    STRICT 1:1 Balanced Batch Generator.
    
    Every batch contains EXACTLY 50% planets and 50% non-planets.
    This is the #1 fix for the zero-classifier trap.
    """
    
    def __init__(
        self,
        file_paths: np.ndarray,
        labels: np.ndarray,
        config: TrainingConfig,
        model_config: ModelConfig,
        augment: bool = False,
        shuffle: bool = True,
        positive_multiplier: int = 1
    ):
        self.config = config
        self.model_config = model_config
        self.augment = augment
        self.shuffle = shuffle
        
        # Separate positive and negative indices
        self.pos_files = file_paths[labels == 1]
        self.neg_files = file_paths[labels == 0]
        self.pos_labels = labels[labels == 1]
        self.neg_labels = labels[labels == 0]
        
        # Multiply positives by augmentation factor
        if positive_multiplier > 1:
            self.pos_files = np.tile(self.pos_files, positive_multiplier)
            self.pos_labels = np.tile(self.pos_labels, positive_multiplier)
        
        # Calculate number of batches based on minority class
        self.half_batch = config.batch_size // 2
        self.n_batches = min(len(self.pos_files), len(self.neg_files)) // self.half_batch
        
        # Indices for sampling
        self.pos_indices = np.arange(len(self.pos_files))
        self.neg_indices = np.arange(len(self.neg_files))
        
        self.on_epoch_end()
        
        print(f"📊 BalancedGenerator: {len(self.pos_files)} pos, {len(self.neg_files)} neg")
        print(f"📊 Batches per epoch: {self.n_batches} (strict 1:1 ratio)")
    
    def __len__(self):
        return self.n_batches
    
    def __getitem__(self, index):
        # Get EXACTLY half_batch positives and half_batch negatives
        pos_batch_idx = self.pos_indices[index * self.half_batch : (index + 1) * self.half_batch]
        neg_batch_idx = self.neg_indices[index * self.half_batch : (index + 1) * self.half_batch]
        
        X_global, X_local, X_scalar, Y = [], [], [], []
        
        # Load positives (with augmentation)
        for i in pos_batch_idx:
            data = self._load_file(self.pos_files[i], augment=self.augment)
            if data is not None:
                X_global.append(data['global'])
                X_local.append(data['local'])
                X_scalar.append(data['scalar'])
                Y.append(1)
        
        # Load negatives (light augmentation)
        for i in neg_batch_idx:
            data = self._load_file(self.neg_files[i], augment=False)
            if data is not None:
                X_global.append(data['global'])
                X_local.append(data['local'])
                X_scalar.append(data['scalar'])
                Y.append(0)
        
        # Shuffle within batch
        indices = np.arange(len(Y))
        np.random.shuffle(indices)
        
        return (
            {
                "global_input": np.array(X_global)[indices],
                "local_input": np.array(X_local)[indices],
                "scalar_input": np.array(X_scalar)[indices]
            },
            np.array(Y)[indices]
        )
    
    def _load_file(self, path, augment=False):
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
            if g.shape[0] != self.model_config.global_shape[0]:
                g = self._resize(g, self.model_config.global_shape[0])
            if l.shape[0] != self.model_config.local_shape[0]:
                l = self._resize(l, self.model_config.local_shape[0])
            
            # Pad scalars
            s_padded = np.zeros(self.model_config.scalar_shape[0], dtype=np.float32)
            s_len = min(len(s), self.model_config.scalar_shape[0])
            s_padded[:s_len] = s[:s_len]
            s_padded = np.nan_to_num(s_padded, nan=0.0)
            
            # Physics-preserving augmentation for positives
            if augment:
                g, l = self._augment(g, l)
            
            return {'global': g, 'local': l, 'scalar': s_padded}
            
        except Exception as e:
            return None
    
    def _resize(self, arr, target_len):
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, arr.flatten()).reshape(-1, 1)
    
    def _augment(self, g, l):
        """AGGRESSIVE physics-preserving augmentations to prevent overfitting."""
        # 1. Time reversal (50% chance)
        if np.random.rand() > 0.5:
            g = np.flip(g, axis=0).copy()
            l = np.flip(l, axis=0).copy()
        
        # 2. Phase shift (LARGER range)
        shift = np.random.randint(-50, 51)  # Increased from ±20
        g = np.roll(g, shift, axis=0)
        
        # 3. Flux scaling (WIDER range)
        scale = np.random.uniform(0.95, 1.05)  # Increased from 0.97-1.03
        g = g * scale
        l = l * scale
        
        # 4. Gaussian noise (MORE noise)
        noise_level = np.random.uniform(0.005, 0.02)  # Increased from 0.001-0.005
        g = g + np.random.normal(0, noise_level, g.shape)
        l = l + np.random.normal(0, noise_level, l.shape)
        
        # 5. Transit depth variation (WIDER ±20%)
        depth_scale = np.random.uniform(0.8, 1.2)  # Increased from 0.9-1.1
        g = 1 + (g - 1) * depth_scale
        l = 1 + (l - 1) * depth_scale
        
        # 6. NEW: Random segment dropout (zero out random parts)
        if np.random.rand() > 0.7:
            start = np.random.randint(0, len(g) - 100)
            length = np.random.randint(10, 50)
            g[start:start+length] = np.median(g)
        
        # 7. NEW: Baseline drift simulation
        if np.random.rand() > 0.5:
            drift = np.linspace(0, np.random.uniform(-0.01, 0.01), len(g)).reshape(-1, 1)
            g = g + drift
        
        return g.astype(np.float32), l.astype(np.float32)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.pos_indices)
            np.random.shuffle(self.neg_indices)


# =============================================================================
# 5. MODEL ARCHITECTURE
# =============================================================================

def build_exotransformer_ultra(config: ModelConfig, pos_ratio: float = 0.08):
    """
    Build the ExoTransformer-Ultra model.
    
    Key feature: Output bias initialized to log(pos/neg) for faster convergence.
    """
    l2 = regularizers.l2(config.l2_rate)
    
    # Calculate output bias for imbalanced data
    # This tells the model "planets are rare" from the start
    output_bias = math.log(pos_ratio / (1 - pos_ratio))  # ~-2.4 for 8% positive
    
    # =========================================================================
    # BRANCH 1: GLOBAL TRANSFORMER (Periodicity Detection)
    # =========================================================================
    input_global = layers.Input(shape=config.global_shape, name="global_input")
    
    # CNN Stem: Tokenize the time series (reduce 2001 -> ~250 tokens)
    x1 = layers.Conv1D(config.embed_dim, 8, strides=8, padding="same")(input_global)
    x1 = layers.LayerNormalization()(x1)
    x1 = layers.Dropout(config.dropout_rate)(x1)
    
    # Positional Encoding (learnable)
    seq_len = x1.shape[1]
    pos_embedding_layer = layers.Embedding(input_dim=500, output_dim=config.embed_dim)
    positions = keras.ops.arange(start=0, stop=seq_len)
    pos_embedding = pos_embedding_layer(positions)
    x1 = x1 + pos_embedding
    
    # Transformer Encoder Blocks
    for _ in range(config.num_transformer_blocks):
        x1 = TransformerBlock(
            config.embed_dim, 
            config.num_heads, 
            config.ff_dim, 
            config.dropout_rate
        )(x1)
    
    # Global pooling
    x1_avg = layers.GlobalAveragePooling1D()(x1)
    x1_max = layers.GlobalMaxPooling1D()(x1)
    x1 = layers.Concatenate()([x1_avg, x1_max])
    x1 = layers.Dense(64, activation='gelu', kernel_regularizer=l2)(x1)
    x1 = layers.Dropout(config.dropout_rate)(x1)
    
    # =========================================================================
    # BRANCH 2: LOCAL SE-CNN (Transit Shape Analysis)
    # =========================================================================
    input_local = layers.Input(shape=config.local_shape, name="local_input")
    
    x2 = input_local
    filters = config.cnn_filters
    
    for i in range(3):
        # Multi-scale parallel convolutions
        conv3 = layers.Conv1D(filters, 3, padding='same', activation='gelu', kernel_regularizer=l2)(x2)
        conv5 = layers.Conv1D(filters, 5, padding='same', activation='gelu', kernel_regularizer=l2)(x2)
        conv7 = layers.Conv1D(filters, 7, padding='same', activation='gelu', kernel_regularizer=l2)(x2)
        
        x2 = layers.Concatenate()([conv3, conv5, conv7])
        x2 = layers.BatchNormalization()(x2)
        
        # Squeeze-and-Excitation
        x2 = SqueezeExcitation(config.se_ratio)(x2)
        
        # Downsample
        x2 = layers.MaxPooling1D(2)(x2)
        x2 = layers.Dropout(config.dropout_rate)(x2)
        
        filters = min(filters * 2, 128)
    
    x2_avg = layers.GlobalAveragePooling1D()(x2)
    x2_max = layers.GlobalMaxPooling1D()(x2)
    x2 = layers.Concatenate()([x2_avg, x2_max])
    x2 = layers.Dense(32, activation='gelu', kernel_regularizer=l2)(x2)  # Reduced from 64
    x2 = layers.Dropout(config.dropout_rate)(x2)
    
    # =========================================================================
    # BRANCH 3: SCALAR FEATURES (Astrophysical Parameters)
    # =========================================================================
    input_scalar = layers.Input(shape=config.scalar_shape, name="scalar_input")
    
    x3 = layers.BatchNormalization()(input_scalar)
    x3 = layers.Dense(16, activation='gelu', kernel_regularizer=l2)(x3)  # Reduced from 32
    x3 = layers.Dropout(config.dropout_rate)(x3)
    x3 = layers.Dense(16, activation='gelu', kernel_regularizer=l2)(x3)  # Reduced from 32
    
    # =========================================================================
    # FUSION: SIMPLE CONCATENATION (removed attention to reduce overfitting)
    # =========================================================================
    # Project all branches to same dimension - SMALLER
    x1_proj = layers.Dense(32, activation='gelu', kernel_regularizer=l2)(x1)  # Reduced from 64
    x1_proj = layers.Dropout(config.dropout_rate)(x1_proj)  # Added dropout
    x2_proj = layers.Dense(32, activation='gelu', kernel_regularizer=l2)(x2)  # Reduced from 64
    x2_proj = layers.Dropout(config.dropout_rate)(x2_proj)  # Added dropout
    x3_proj = layers.Dense(32, activation='gelu', kernel_regularizer=l2)(x3)  # Reduced from 64
    x3_proj = layers.Dropout(config.dropout_rate)(x3_proj)  # Added dropout
    
    # Simple concatenation (removed attention mechanism - was overfitting)
    fusion = layers.Concatenate()([x1_proj, x2_proj, x3_proj])
    
    # Single fusion layer with strong regularization
    fusion = layers.Dense(config.fusion_dim, activation='gelu', kernel_regularizer=l2, name="fusion_layer")(fusion)
    fusion = layers.BatchNormalization()(fusion)
    fusion = layers.Dropout(config.dropout_rate)(fusion)  # Reduced from 1.5x
    
    # Removed extra dense layer - was causing overfitting
    # fusion = layers.Dense(64, activation='gelu', kernel_regularizer=l2)(fusion)
    # fusion = layers.Dropout(config.dropout_rate)(fusion)
    
    # =========================================================================
    # OUTPUT: BIAS-INITIALIZED FOR IMBALANCED DATA
    # =========================================================================
    # For mixed precision: Cast back to float32 for numerical stability
    fusion = layers.Activation('linear', dtype='float32')(fusion)
    
    output = layers.Dense(
        1, 
        activation='sigmoid',
        bias_initializer=initializers.Constant(output_bias),
        kernel_regularizer=l2,  # Added regularization to output layer
        dtype='float32',
        name="output"
    )(fusion)
    
    model = models.Model(
        inputs=[input_global, input_local, input_scalar],
        outputs=output,
        name="ExoTransformer_Ultra"
    )
    
    return model


# =============================================================================
# 6. LEARNING RATE SCHEDULE WITH WARMUP
# =============================================================================

class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay."""
    
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
        cosine_decay = 0.5 * (1 + tf.cos(math.pi * decay_step / decay_steps))
        decay_lr = self.peak_lr * cosine_decay
        
        return tf.where(step < self.warmup_steps, warmup_lr, decay_lr)
    
    def get_config(self):
        return {
            "peak_lr": float(self.peak_lr),
            "warmup_steps": int(self.warmup_steps),
            "total_steps": int(self.total_steps)
        }


# =============================================================================
# 7. EVALUATION METRICS
# =============================================================================

def find_optimal_threshold(y_true, y_pred):
    """Find threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Find best
    best_idx = np.argmax(f1_scores[:-1])  # Last element is always 0
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1


def recall_at_precision(y_true, y_pred, target_precision=0.99):
    """Calculate recall at a fixed precision level."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    
    # Find where precision >= target
    valid_mask = precision >= target_precision
    if not valid_mask.any():
        return 0.0
    
    # Return max recall where precision meets target
    return recall[valid_mask].max()


def evaluate_comprehensive(y_true, y_pred, output_dir, fold):
    """Generate comprehensive evaluation metrics and plots."""
    results = {}
    
    # Basic metrics
    results['roc_auc'] = roc_auc_score(y_true, y_pred)
    results['pr_auc'] = average_precision_score(y_true, y_pred)
    
    # Recall at precision levels
    for p in [0.90, 0.95, 0.99]:
        results[f'recall@p{int(p*100)}'] = recall_at_precision(y_true, y_pred, p)
    
    # Optimal threshold
    opt_thresh, opt_f1 = find_optimal_threshold(y_true, y_pred)
    results['optimal_threshold'] = opt_thresh
    results['optimal_f1'] = opt_f1
    
    # Predictions at optimal threshold
    y_pred_binary = (y_pred >= opt_thresh).astype(int)
    
    # Classification report
    print(f"\n📊 Fold {fold} Classification Report (thresh={opt_thresh:.3f}):")
    print(classification_report(y_true, y_pred_binary, target_names=['Non-Planet', 'Planet']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    results['confusion_matrix'] = cm.tolist()
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    axes[0].plot(fpr, tpr, 'b-', label=f'ROC (AUC={results["roc_auc"]:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    axes[1].plot(recall, precision, 'g-', label=f'PR (AUC={results["pr_auc"]:.3f})')
    axes[1].axhline(y=0.99, color='r', linestyle='--', alpha=0.5, label='P=0.99')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Actual')
    axes[2].set_title(f'Confusion Matrix (t={opt_thresh:.2f})')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'evaluation_fold_{fold}.png', dpi=150)
    plt.close()
    
    return results


# =============================================================================
# 8. MAIN TRAINING PIPELINE
# =============================================================================

def train_exotransformer_ultra(data_dir: str, output_dir: str):
    """Full training pipeline with all imbalance fixes."""
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(output_dir) / f"ultra_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(output_path / 'training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 ExoTransformer-Ultra Training Started")
    logger.info(f"📁 Output: {output_path}")
    
    # Load data
    logger.info("📂 Loading dataset...")
    all_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    valid_files = []
    labels = []
    
    for f in tqdm(all_files, desc="Indexing"):
        try:
            with np.load(f) as data:
                labels.append(int(data['label']))
                valid_files.append(f)
        except:
            pass
    
    X_files = np.array(valid_files)
    y_labels = np.array(labels)
    
    n_pos = y_labels.sum()
    n_neg = len(y_labels) - n_pos
    pos_ratio = n_pos / len(y_labels)
    
    logger.info(f"📊 Dataset: {len(X_files)} samples")
    logger.info(f"📊 Positives: {n_pos} ({pos_ratio*100:.1f}%)")
    logger.info(f"📊 Negatives: {n_neg} ({(1-pos_ratio)*100:.1f}%)")
    logger.info(f"📊 Imbalance Ratio: 1:{n_neg/n_pos:.1f}")
    
    # Config
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    # Cross-validation
    kf = StratifiedKFold(n_splits=train_config.folds, shuffle=True, random_state=SEED)
    
    all_oof_preds = np.zeros(len(X_files))
    all_oof_features = []
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_files, y_labels)):
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 FOLD {fold + 1}/{train_config.folds}")
        logger.info(f"{'='*60}")
        
        # Data generators
        train_gen = BalancedDataGenerator(
            X_files[train_idx], y_labels[train_idx],
            train_config, model_config,
            augment=True, shuffle=True,
            positive_multiplier=train_config.positive_augment_factor
        )
        
        val_gen = BalancedDataGenerator(
            X_files[val_idx], y_labels[val_idx],
            train_config, model_config,
            augment=False, shuffle=False,
            positive_multiplier=1
        )
        
        # Build model
        model = build_exotransformer_ultra(model_config, pos_ratio)
        
        if fold == 0:
            model.summary(print_fn=logger.info)
        
        # Learning rate schedule
        steps_per_epoch = len(train_gen)
        total_steps = steps_per_epoch * train_config.epochs
        warmup_steps = steps_per_epoch * train_config.warmup_epochs
        
        lr_schedule = WarmupCosineDecay(
            peak_lr=train_config.peak_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
        
        # Compile with asymmetric focal loss
        optimizer = optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-5,
            clipnorm=1.0  # Gradient clipping
        )
        
        loss = AsymmetricFocalLoss(
            alpha=train_config.focal_alpha,
            gamma=train_config.focal_gamma,
            label_smoothing=train_config.label_smoothing
        )
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(curve='PR', name='pr_auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        # Callbacks (NOTE: No ReduceLROnPlateau - conflicts with WarmupCosineDecay schedule)
        checkpoint_path = output_path / f'model_fold_{fold+1}.keras'
        cbs = [
            callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor='val_pr_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_pr_auc',
                mode='max',
                patience=train_config.patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.TerminateOnNaN(),  # Safety: stop if loss becomes NaN
            callbacks.TensorBoard(
                log_dir=str(output_path / 'tensorboard' / f'fold_{fold+1}'),
                histogram_freq=0,
                write_graph=False,  # Saves memory
                update_freq='epoch'
            )
        ]
        
        # Train
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=train_config.epochs,
            callbacks=cbs,
            verbose=1
        )
        
        # Load best model
        model = keras.models.load_model(
            str(checkpoint_path),
            custom_objects={
                'AsymmetricFocalLoss': AsymmetricFocalLoss,
                'SqueezeExcitation': SqueezeExcitation,
                'TransformerBlock': TransformerBlock,
                'WarmupCosineDecay': WarmupCosineDecay
            }
        )
        
        # Extract features for XGBoost
        logger.info("🔧 Extracting features for XGBoost stacking...")
        feature_extractor = models.Model(
            inputs=model.input,
            outputs=model.get_layer('fusion_layer').output
        )
        
        # Validation predictions (Deep Learning)
        val_gen_eval = BalancedDataGenerator(
            X_files[val_idx], y_labels[val_idx],
            train_config, model_config,
            augment=False, shuffle=False,
            positive_multiplier=1
        )
        
        # Get DL predictions
        dl_preds = model.predict(val_gen_eval, verbose=0).flatten()
        
        # Get features
        val_features = feature_extractor.predict(val_gen_eval, verbose=0)
        
        # Also get training features for XGBoost
        train_gen_eval = BalancedDataGenerator(
            X_files[train_idx], y_labels[train_idx],
            train_config, model_config,
            augment=False, shuffle=False,
            positive_multiplier=1
        )
        train_features = feature_extractor.predict(train_gen_eval, verbose=0)
        
        # XGBoost Stacking
        if XGB_AVAILABLE:
            logger.info("🌲 Training XGBoost meta-learner...")
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=train_config.xgb_scale_pos_weight,  # CRITICAL
                tree_method='hist',
                random_state=SEED,
                eval_metric='aucpr'
            )
            
            # Get balanced labels for training XGBoost
            train_labels_balanced = np.concatenate([
                np.ones(len(train_gen_eval.pos_indices[:len(train_features)//2])),
                np.zeros(len(train_gen_eval.neg_indices[:len(train_features)//2]))
            ])[:len(train_features)]
            
            val_labels_balanced = np.concatenate([
                np.ones(len(val_gen_eval.pos_indices[:len(val_features)//2])),
                np.zeros(len(val_gen_eval.neg_indices[:len(val_features)//2]))
            ])[:len(val_features)]
            
            xgb_model.fit(
                train_features, train_labels_balanced,
                eval_set=[(val_features, val_labels_balanced)],
                verbose=False
            )
            
            # Final predictions = XGBoost on DL features
            final_preds = xgb_model.predict_proba(val_features)[:, 1]
            
            xgb_model.save_model(str(output_path / f'xgboost_fold_{fold+1}.json'))
        else:
            # Fallback to calibrated logistic regression
            lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
            train_labels_balanced = np.concatenate([
                np.ones(len(train_features)//2),
                np.zeros(len(train_features)//2)
            ])[:len(train_features)]
            lr_model.fit(train_features, train_labels_balanced)
            final_preds = lr_model.predict_proba(val_features)[:, 1]
        
        # Evaluate
        val_labels_for_eval = val_labels_balanced[:len(final_preds)]
        metrics = evaluate_comprehensive(val_labels_for_eval, final_preds, output_path, fold + 1)
        fold_results.append(metrics)
        
        logger.info(f"📈 Fold {fold+1} Results:")
        logger.info(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"   PR-AUC:  {metrics['pr_auc']:.4f}")
        logger.info(f"   Recall@P=0.99: {metrics['recall@p99']:.4f}")
        logger.info(f"   Optimal F1: {metrics['optimal_f1']:.4f}")
        
        # Cleanup
        keras.backend.clear_session()
    
    # Aggregate Results
    logger.info(f"\n{'='*60}")
    logger.info("🏆 FINAL AGGREGATE RESULTS")
    logger.info(f"{'='*60}")
    
    avg_metrics = {
        'roc_auc': np.mean([r['roc_auc'] for r in fold_results]),
        'pr_auc': np.mean([r['pr_auc'] for r in fold_results]),
        'recall@p99': np.mean([r['recall@p99'] for r in fold_results]),
        'recall@p95': np.mean([r['recall@p95'] for r in fold_results]),
        'optimal_f1': np.mean([r['optimal_f1'] for r in fold_results]),
    }
    
    logger.info(f"📊 Avg ROC-AUC:      {avg_metrics['roc_auc']:.4f}")
    logger.info(f"📊 Avg PR-AUC:       {avg_metrics['pr_auc']:.4f}")
    logger.info(f"📊 Avg Recall@P=0.99: {avg_metrics['recall@p99']:.4f}")
    logger.info(f"📊 Avg Recall@P=0.95: {avg_metrics['recall@p95']:.4f}")
    logger.info(f"📊 Avg Optimal F1:   {avg_metrics['optimal_f1']:.4f}")
    
    # Save results
    results = {
        'timestamp': timestamp,
        'model_config': model_config.__dict__,
        'training_config': train_config.__dict__,
        'fold_results': fold_results,
        'aggregate_metrics': avg_metrics
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n✅ Training complete! Results saved to {output_path}")
    
    return results


# =============================================================================
# 9. CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExoTransformer-Ultra Training")
    parser.add_argument("--data_dir", default="notebooks/results_koi", help="Data directory")
    parser.add_argument("--output_dir", default="experiments", help="Output directory")
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║            ExoTransformer-Ultra: Final Optimal Model         ║
    ║                                                              ║
    ║  Features:                                                   ║
    ║  • Transformer + SE-CNN Hybrid Architecture                  ║
    ║  • Strict 1:1 Balanced Batching                              ║
    ║  • Asymmetric Focal Loss (α=0.75, γ=3.0)                     ║
    ║  • Output Bias Initialization                                ║
    ║  • XGBoost Stacking with scale_pos_weight                    ║
    ║  • Physics-Preserving Augmentation                           ║
    ║                                                              ║
    ║  Target: PR-AUC > 0.85, F1 > 0.75, Recall@P99 > 0.70         ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    train_exotransformer_ultra(args.data_dir, args.output_dir)

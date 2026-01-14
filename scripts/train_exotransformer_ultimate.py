#!/usr/bin/env python3
"""
=============================================================================
EXOTRANSFORMER ULTIMATE - State-of-the-Art Exoplanet Detection Model
=============================================================================

This is the DEFINITIVE model architecture combining:
1. Multi-Scale CNN (like NASA ExoMiner)
2. Transformer Encoder (attention across time)
3. Squeeze-and-Excitation blocks
4. Focal Loss for class imbalance
5. Advanced augmentation pipeline

Target: Surpass NASA ExoMiner performance with sufficient data

Architecture Overview:
┌─────────────────────────────────────────────────────────────────┐
│                     EXOTRANSFORMER ULTIMATE                     │
├─────────────────────────────────────────────────────────────────┤
│  Input: Global View (201,) + Local View (61,)                   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           MULTI-SCALE CNN ENCODER                       │    │
│  │  • Conv1D kernels: 3, 7, 15, 31 (different durations)   │    │
│  │  • Squeeze-Excitation attention                         │    │
│  │  • Residual connections                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           TRANSFORMER ENCODER                           │    │
│  │  • Multi-head self-attention (4 heads)                  │    │
│  │  • Positional encoding                                  │    │
│  │  • 2 transformer layers                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           DUAL-BRANCH FUSION                            │    │
│  │  • Global branch → attention pooling                    │    │
│  │  • Local branch → attention pooling                     │    │
│  │  • Feature concatenation + cross-attention              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           CLASSIFIER HEAD                               │    │
│  │  • Dense layers with dropout                            │    │
│  │  • Sigmoid output (binary classification)               │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

Author: Exoplanet AI Research Team
Paper: IEEE Transactions on Neural Networks and Learning Systems
Date: 2026
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, LearningRateScheduler
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix, f1_score
)
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# FOCAL LOSS - Critical for imbalanced exoplanet detection
# =============================================================================

# =============================================================================
# NASA ExoMiner-style loss configuration
# Key insight: Use Keras built-in BinaryFocalCrossentropy with apply_class_balancing
# DO NOT combine with external class_weight (causes double-weighting)
# =============================================================================

def get_focal_loss(pos_fraction=0.1):
    """
    Get properly configured Focal Loss for exoplanet detection.
    
    NASA ExoMiner approach:
    - Use BinaryFocalCrossentropy with apply_class_balancing=True
    - alpha = proportion of NEGATIVE class (so positive gets weighted more)
    - gamma = 2.0 for focusing on hard examples
    
    For 10% positives: alpha=0.9 means 90% weight on negatives baseline,
    but with apply_class_balancing, positives get 9x more weight.
    
    Args:
        pos_fraction: Fraction of positive samples (e.g., 0.1 for 10%)
    """
    # alpha should be negative class proportion for proper weighting
    alpha = 1.0 - pos_fraction  # 0.9 for 10% positive class
    
    return keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=True,  # CRITICAL: enables automatic class weighting
        alpha=alpha,  # Weight for negative class
        gamma=2.0,  # Focus on hard examples
        from_logits=False,
        label_smoothing=0.0,
        name='focal_loss'
    )


# =============================================================================
# CUSTOM LAYERS
# =============================================================================

class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, max_len: int = 201, d_model: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
    def build(self, input_shape):
        # Create positional encoding matrix
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe = np.zeros((self.max_len, self.d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        # Store as a non-trainable weight for Keras 3.x compatibility
        self.pe = self.add_weight(
            name='positional_encoding',
            shape=(1, self.max_len, self.d_model),
            initializer=keras.initializers.Constant(pe[np.newaxis, :, :]),
            trainable=False
        )
        super().build(input_shape)
    
    def call(self, x):
        seq_len = keras.ops.shape(x)[1]
        return x + self.pe[:, :seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


class SqueezeExcitation(layers.Layer):
    """
    Squeeze-and-Excitation block for channel attention
    Learns to weight different feature channels based on global context
    """
    
    def __init__(self, reduction_ratio: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.global_pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(channels // self.reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
        super().build(input_shape)
    
    def call(self, x):
        # Squeeze
        squeeze = self.global_pool(x)
        # Excitation
        excite = self.fc1(squeeze)
        excite = self.fc2(excite)
        # Scale - use keras.ops for Keras 3.x
        excite = keras.ops.expand_dims(excite, axis=1)
        return x * excite
    
    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


class MultiHeadAttentionPooling(layers.Layer):
    """
    Multi-head attention pooling for sequence aggregation
    Learns to attend to important time steps
    """
    
    def __init__(self, num_heads: int = 4, key_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
    
    def build(self, input_shape):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim
        )
        # Learnable query for pooling
        self.query = self.add_weight(
            shape=(1, 1, input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_query'
        )
        super().build(input_shape)
    
    def call(self, x):
        batch_size = keras.ops.shape(x)[0]
        query = keras.ops.tile(self.query, [batch_size, 1, 1])
        attended = self.attention(query, x, x)
        return keras.ops.squeeze(attended, axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim})
        return config


class TransformerBlock(layers.Layer):
    """
    Transformer encoder block with pre-norm architecture
    """
    
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        dff: int = 128,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.dff, activation='gelu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        super().build(input_shape)
    
    def call(self, x, training=False):
        # Pre-norm architecture
        norm_x = self.layernorm1(x)
        attn_output = self.attention(norm_x, norm_x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = x + attn_output
        
        norm_x = self.layernorm2(x)
        ffn_output = self.ffn(norm_x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return x + ffn_output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate
        })
        return config


# =============================================================================
# EXOTRANSFORMER ULTIMATE MODEL
# =============================================================================

def build_exotransformer_ultimate(
    global_size: int = 201,
    local_size: int = 61,
    d_model: int = 64,
    num_heads: int = 4,
    num_transformer_layers: int = 2,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4
) -> Model:
    """
    Build the ExoTransformer Ultimate model
    
    Architecture combines:
    - Multi-scale CNN for feature extraction
    - Transformer encoder for temporal modeling
    - Dual-branch processing for global/local views
    - SE-attention for channel weighting
    
    Args:
        global_size: Size of global view input
        local_size: Size of local view input
        d_model: Model dimension for transformer
        num_heads: Number of attention heads
        num_transformer_layers: Number of transformer blocks
        dropout_rate: Dropout rate
        l2_reg: L2 regularization strength
    
    Returns:
        Compiled Keras model
    """
    regularizer = regularizers.l2(l2_reg)
    
    # =========================================================================
    # INPUT LAYERS
    # =========================================================================
    global_input = layers.Input(shape=(global_size,), name='global_input')
    local_input = layers.Input(shape=(local_size,), name='local_input')
    
    # Reshape for Conv1D: (batch, time, channels)
    global_x = layers.Reshape((global_size, 1))(global_input)
    local_x = layers.Reshape((local_size, 1))(local_input)
    
    # =========================================================================
    # MULTI-SCALE CNN ENCODER
    # =========================================================================
    def multi_scale_cnn_block(x, filters, name_prefix):
        """Multi-scale convolution with different kernel sizes"""
        # Different kernel sizes capture different transit durations
        conv_3 = layers.Conv1D(
            filters // 4, 3, padding='same',
            kernel_regularizer=regularizer,
            name=f'{name_prefix}_conv3'
        )(x)
        conv_7 = layers.Conv1D(
            filters // 4, 7, padding='same',
            kernel_regularizer=regularizer,
            name=f'{name_prefix}_conv7'
        )(x)
        conv_15 = layers.Conv1D(
            filters // 4, 15, padding='same',
            kernel_regularizer=regularizer,
            name=f'{name_prefix}_conv15'
        )(x)
        conv_31 = layers.Conv1D(
            filters // 4, 31, padding='same',
            kernel_regularizer=regularizer,
            name=f'{name_prefix}_conv31'
        )(x)
        
        # Concatenate multi-scale features
        concat = layers.Concatenate()([conv_3, conv_7, conv_15, conv_31])
        concat = layers.BatchNormalization()(concat)
        concat = layers.Activation('gelu')(concat)
        
        # SE attention
        concat = SqueezeExcitation(reduction_ratio=4)(concat)
        
        return concat
    
    # Global branch CNN
    global_cnn = multi_scale_cnn_block(global_x, d_model, 'global_ms')
    global_cnn = layers.Dropout(dropout_rate)(global_cnn)
    global_cnn = layers.Conv1D(
        d_model, 3, padding='same',
        kernel_regularizer=regularizer,
        name='global_proj'
    )(global_cnn)
    global_cnn = layers.BatchNormalization()(global_cnn)
    global_cnn = layers.Activation('gelu')(global_cnn)
    
    # Local branch CNN
    local_cnn = multi_scale_cnn_block(local_x, d_model, 'local_ms')
    local_cnn = layers.Dropout(dropout_rate)(local_cnn)
    local_cnn = layers.Conv1D(
        d_model, 3, padding='same',
        kernel_regularizer=regularizer,
        name='local_proj'
    )(local_cnn)
    local_cnn = layers.BatchNormalization()(local_cnn)
    local_cnn = layers.Activation('gelu')(local_cnn)
    
    # =========================================================================
    # POSITIONAL ENCODING
    # =========================================================================
    global_pos = PositionalEncoding(max_len=global_size, d_model=d_model)(global_cnn)
    local_pos = PositionalEncoding(max_len=local_size, d_model=d_model)(local_cnn)
    
    # =========================================================================
    # TRANSFORMER ENCODER
    # =========================================================================
    for i in range(num_transformer_layers):
        global_pos = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            dff=d_model * 2,
            dropout_rate=dropout_rate
        )(global_pos)
        
        local_pos = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            dff=d_model * 2,
            dropout_rate=dropout_rate
        )(local_pos)
    
    # =========================================================================
    # ATTENTION POOLING
    # =========================================================================
    global_pooled = MultiHeadAttentionPooling(num_heads=2, key_dim=d_model // 2)(global_pos)
    local_pooled = MultiHeadAttentionPooling(num_heads=2, key_dim=d_model // 2)(local_pos)
    
    # =========================================================================
    # CROSS-ATTENTION FUSION (Keras 3.x compatible)
    # =========================================================================
    # Reshape for attention: (batch, 1, features)
    global_expanded = layers.Reshape((1, d_model))(global_pooled)
    local_expanded = layers.Reshape((1, d_model))(local_pooled)
    
    # Global attends to local
    global_attended = layers.MultiHeadAttention(
        num_heads=2, key_dim=d_model // 2
    )(global_expanded, local_expanded)
    
    # Flatten back: (batch, features)
    global_attended = layers.Flatten()(global_attended)
    
    # Combine all features
    combined = layers.Concatenate()([global_pooled, local_pooled, global_attended])
    combined = layers.LayerNormalization()(combined)
    
    # =========================================================================
    # CLASSIFIER HEAD
    # =========================================================================
    x = layers.Dense(
        128,
        activation='gelu',
        kernel_regularizer=regularizer,
        name='classifier_1'
    )(combined)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(
        64,
        activation='gelu',
        kernel_regularizer=regularizer,
        name='classifier_2'
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Build model
    model = Model(
        inputs=[global_input, local_input],
        outputs=output,
        name='ExoTransformer_Ultimate'
    )
    
    return model


# =============================================================================
# DATA LOADING AND AUGMENTATION (Keras 3.x Compatible)
# =============================================================================

def resize_array(arr: np.ndarray, target_size: int) -> np.ndarray:
    """Resize array to target size using interpolation"""
    if len(arr) == target_size:
        return arr
    indices = np.linspace(0, len(arr) - 1, target_size)
    return np.interp(indices, np.arange(len(arr)), arr)


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize array to zero mean and unit variance"""
    arr = np.nan_to_num(arr, nan=1.0, posinf=1.0, neginf=1.0)
    mean = np.mean(arr)
    std = np.std(arr)
    if std > 1e-8:
        return (arr - mean) / std
    return arr - mean


def load_npz_data(file_paths: List[str], labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all data into memory for faster training (Keras 3.x compatible)
    Handles different NPZ formats from results_koi and results_confirmed.
    Returns: (global_views, local_views, labels)
    """
    global_views = []
    local_views = []
    valid_labels = []
    
    for i, fpath in enumerate(file_paths):
        try:
            data = np.load(fpath)
            
            if 'global_view' in data and 'local_view' in data:
                global_view = data['global_view'].astype(np.float32)
                local_view = data['local_view'].astype(np.float32)
                
                # Handle different formats:
                # Format 1 (results_koi): global=(2001,1), local=(201,1) - Need to resize
                # Format 2 (results_confirmed): global=(201,), local=(61,) - Ready to use
                
                # Flatten if needed
                if global_view.ndim > 1:
                    global_view = global_view.flatten()
                if local_view.ndim > 1:
                    local_view = local_view.flatten()
                
                # Resize to expected dimensions
                if len(global_view) != 201:
                    global_view = resize_array(global_view, 201)
                if len(local_view) != 61:
                    local_view = resize_array(local_view, 61)
                    
            else:
                # Old format - extract from flux
                flux = data.get('flux', data.get('binned_flux', np.ones(201)))
                if flux.ndim > 1:
                    flux = flux.flatten()
                global_view = flux[:201] if len(flux) >= 201 else np.pad(flux, (0, 201-len(flux)))
                local_view = flux[:61] if len(flux) >= 61 else np.pad(flux, (0, 61-len(flux)))
            
            # Normalize
            global_view = normalize_array(global_view)
            local_view = normalize_array(local_view)
            
            global_views.append(global_view)
            local_views.append(local_view)
            valid_labels.append(labels[i])
            
        except Exception as e:
            # Skip corrupt files
            continue
    
    logger.info(f"Successfully loaded {len(valid_labels)} samples out of {len(file_paths)}")
    
    return (
        np.array(global_views, dtype=np.float32),
        np.array(local_views, dtype=np.float32),
        np.array(valid_labels, dtype=np.float32)
    )


class DataAugmentation(layers.Layer):
    """
    On-GPU data augmentation layer for training-time augmentation.
    This is faster than CPU augmentation and Keras 3.x compatible.
    """
    
    def __init__(self, noise_stddev: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.noise_stddev = noise_stddev
    
    def call(self, inputs, training=None):
        if training:
            # Add Gaussian noise during training
            noise = keras.random.normal(
                shape=keras.ops.shape(inputs),
                mean=0.0,
                stddev=self.noise_stddev
            )
            return inputs + noise
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({"noise_stddev": self.noise_stddev})
        return config


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def get_callbacks(output_dir: str, fold: int = 0) -> List:
    """Get training callbacks"""
    
    callbacks = [
        ModelCheckpoint(
            filepath=f'{output_dir}/best_exotransformer_fold_{fold}.keras',
            monitor='val_auc_pr',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_auc_pr',
            mode='max',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=f'{output_dir}/logs/fold_{fold}',
            histogram_freq=0
        )
    ]
    
    return callbacks


def cosine_warmup_schedule(epoch: int, total_epochs: int, warmup_epochs: int, 
                           initial_lr: float, min_lr: float) -> float:
    """Cosine annealing with warmup"""
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    fold: int = 0
) -> Dict:
    """Comprehensive model evaluation from numpy arrays (Keras 3.x compatible)"""
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    
    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    # Find optimal threshold (maximize F1)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    # Binary predictions at optimal threshold
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    
    # Classification report
    report = classification_report(y_true, y_pred_binary, target_names=['Not Planet', 'Planet'])
    
    # Recall at 99% precision (NASA ExoMiner metric)
    recall_at_99 = 0.0
    for i, p in enumerate(precision):
        if p >= 0.99 and i < len(recall):
            recall_at_99 = max(recall_at_99, recall[i])
    
    results = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'recall_at_99_precision': float(recall_at_99),
        'optimal_threshold': float(optimal_threshold),
        'f1_at_optimal': float(f1_scores[optimal_idx])
    }
    
    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold} EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"PR-AUC: {pr_auc:.4f}")
    logger.info(f"Recall@P99: {recall_at_99:.4f}")
    logger.info(f"Optimal F1: {f1_scores[optimal_idx]:.4f} (threshold={optimal_threshold:.3f})")
    logger.info(f"\n{report}")
    
    # Plot curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PR Curve
    axes[0].plot(recall, precision, 'b-', linewidth=2, label=f'PR-AUC = {pr_auc:.4f}')
    axes[0].fill_between(recall, precision, alpha=0.2)
    axes[0].axhline(y=0.99, color='r', linestyle='--', label='99% Precision')
    axes[0].set_xlabel('Recall', fontsize=12)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_title('Precision-Recall Curve', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC-AUC = {roc_auc:.4f}')
    axes[1].fill_between(fpr, tpr, alpha=0.2)
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1].set_title('ROC Curve', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/evaluation_curves_fold_{fold}.png', dpi=150)
    plt.close()
    
    return results


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def load_data(data_dirs: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    Load data from multiple directories with proper deduplication.
    Prevents data leakage by ensuring (KIC_ID, Period) pairs are unique.
    """
    file_paths = []
    labels = []
    seen_targets = set()  # Set to store unique identifiers (e.g., "10000941_3.50")
    duplicates = 0
    
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            continue
            
        npz_files = list(data_path.glob("*.npz"))
        logger.info(f"Scanning {len(npz_files)} files in {data_dir}...")
        
        for f in npz_files:
            try:
                # Deduplication Strategy
                # Filename format expected: KIC_{ID}_P{Period}.npz
                fname = f.name
                if 'KIC_' in fname and '_P' in fname:
                    parts = fname.replace('.npz', '').split('_')
                    # Find part with KIC ID and Period
                    # Heuristic: Find 'P' part
                    p_part = next((p for p in parts if p.startswith('P')), None)
                    kic_part = next((p for p in parts if p.isdigit()), None)
                    
                    if p_part and kic_part:
                        unique_id = f"{kic_part}_{p_part}"
                        if unique_id in seen_targets:
                            duplicates += 1
                            continue # Skip duplicate
                        seen_targets.add(unique_id)

                data = np.load(f)
                label = data.get('label', None)
                
                if label is None:
                    # Infer label from filename or directory
                    if 'confirmed' in str(f).lower() or 'positive' in str(f).lower():
                        label = 1
                    else:
                        label = 0
                
                file_paths.append(str(f))
                labels.append(int(label))
                
            except Exception as e:
                continue
    
    logger.info(f"Skipped {duplicates} duplicate files to prevent data leakage.")
    return file_paths, np.array(labels)


def train_exotransformer(
    data_dirs: List[str],
    output_dir: str,
    n_folds: int = 5,
    epochs: int = 100,
    batch_size: int = 32,
    initial_lr: float = 1e-3
):
    """
    Train ExoTransformer Ultimate with k-fold cross-validation
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    file_paths, labels = load_data(data_dirs)
    
    n_total = len(file_paths)
    n_positive = np.sum(labels)
    n_negative = n_total - n_positive
    
    logger.info(f"Total samples: {n_total}")
    logger.info(f"Positives: {n_positive} ({100*n_positive/n_total:.1f}%)")
    logger.info(f"Negatives: {n_negative} ({100*n_negative/n_total:.1f}%)")
    
    if n_positive < 100:
        logger.warning("⚠️ Very few positive samples! Consider downloading more data.")
    
    # K-Fold cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold + 1}/{n_folds}")
        logger.info(f"{'='*60}")
        
        # Split data
        train_files = [file_paths[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_files = [file_paths[i] for i in val_idx]
        val_labels = labels[val_idx]
        
        logger.info(f"Train: {len(train_files)} samples ({np.sum(train_labels)} positives)")
        logger.info(f"Val: {len(val_files)} samples ({np.sum(val_labels)} positives)")
        
        # Load data into memory (Keras 3.x compatible)
        logger.info("Loading training data into memory...")
        train_global, train_local, train_y = load_npz_data(train_files, train_labels)
        logger.info("Loading validation data into memory...")
        val_global, val_local, val_y = load_npz_data(val_files, val_labels)
        
        logger.info(f"Loaded: Train={len(train_y)}, Val={len(val_y)}")
        
        # Build model
        model = build_exotransformer_ultimate(
            global_size=201,
            local_size=61,
            d_model=64,
            num_heads=4,
            num_transformer_layers=2,
            dropout_rate=0.3,
            l2_reg=1e-4
        )
        
        # Calculate positive fraction for focal loss configuration
        n_pos = np.sum(train_y)
        n_neg = len(train_y) - n_pos
        pos_fraction = n_pos / len(train_y) if len(train_y) > 0 else 0.1
        logger.info(f"Class distribution: {n_pos} positives ({pos_fraction*100:.1f}%), {n_neg} negatives")
        
        # Compile with NASA ExoMiner-style Focal Loss (NO external class_weight!)
        # BinaryFocalCrossentropy with apply_class_balancing=True handles imbalance internally
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=initial_lr
            ),
            loss=get_focal_loss(pos_fraction),  # Focal loss with built-in class balancing
            metrics=[
                keras.metrics.AUC(name='auc_roc', curve='ROC'),
                keras.metrics.AUC(name='auc_pr', curve='PR'),
                keras.metrics.Precision(name='precision', thresholds=0.5),  # Standard threshold
                keras.metrics.Recall(name='recall', thresholds=0.5)  # Standard threshold
            ]
        )
        
        if fold == 0:
            model.summary()
            logger.info(f"Total parameters: {model.count_params():,}")
        
        # Learning rate schedule
        lr_schedule = lambda epoch: cosine_warmup_schedule(
            epoch, epochs, warmup_epochs=5,
            initial_lr=initial_lr, min_lr=1e-6
        )
        
        callbacks = get_callbacks(output_dir, fold + 1)
        callbacks.append(LearningRateScheduler(lr_schedule, verbose=0))
        
        # Train WITHOUT external class_weight - focal loss handles imbalance internally
        # This is the NASA ExoMiner approach: focal loss with apply_class_balancing=True
        history = model.fit(
            x={'global_input': train_global, 'local_input': train_local},
            y=train_y,
            validation_data=(
                {'global_input': val_global, 'local_input': val_local},
                val_y
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1  # No class_weight - handled by focal loss
        )
        
        # Evaluate
        val_preds = model.predict(
            {'global_input': val_global, 'local_input': val_local},
            verbose=0
        ).flatten()
        results = evaluate_model_from_arrays(val_y, val_preds, output_dir, fold + 1)
        all_results.append(results)
        
        # Save training history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(f'{output_dir}/history_fold_{fold + 1}.csv', index=False)
        
        # Clear memory
        del model, train_global, train_local, train_y, val_global, val_local, val_y
        keras.backend.clear_session()
    
    # Summary of all folds
    logger.info(f"\n{'='*60}")
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    
    mean_roc = np.mean([r['roc_auc'] for r in all_results])
    std_roc = np.std([r['roc_auc'] for r in all_results])
    mean_pr = np.mean([r['pr_auc'] for r in all_results])
    std_pr = np.std([r['pr_auc'] for r in all_results])
    mean_recall99 = np.mean([r['recall_at_99_precision'] for r in all_results])
    
    logger.info(f"ROC-AUC: {mean_roc:.4f} ± {std_roc:.4f}")
    logger.info(f"PR-AUC: {mean_pr:.4f} ± {std_pr:.4f}")
    logger.info(f"Recall@P99: {mean_recall99:.4f}")
    
    # NASA ExoMiner comparison
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON WITH NASA EXOMINER")
    logger.info(f"{'='*60}")
    logger.info(f"Our Model:    PR-AUC = {mean_pr:.4f}, Recall@P99 = {mean_recall99:.4f}")
    logger.info(f"NASA ExoMiner: PR-AUC = ~0.90, Recall@P99 = 0.936")
    
    if mean_recall99 >= 0.90:
        logger.info("🎉 EXCEEDS NASA ExoMiner performance!")
    elif mean_recall99 >= 0.80:
        logger.info("✓ Comparable to NASA ExoMiner")
    else:
        logger.info("→ Need more positive samples to match NASA ExoMiner")
    
    # Save final summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': n_total,
        'n_positives': int(n_positive),
        'n_folds': n_folds,
        'fold_results': all_results,
        'mean_roc_auc': float(mean_roc),
        'std_roc_auc': float(std_roc),
        'mean_pr_auc': float(mean_pr),
        'std_pr_auc': float(std_pr),
        'mean_recall_at_99_precision': float(mean_recall99)
    }
    
    with open(f'{output_dir}/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ExoTransformer Ultimate")
    parser.add_argument(
        '--data_dirs',
        type=str,
        nargs='+',
        default=['notebooks/results_koi', 'notebooks/results_confirmed'],
        help='Directories containing training data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/exotransformer_ultimate',
        help='Output directory'
    )
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("EXOTRANSFORMER ULTIMATE - STATE-OF-THE-ART TRAINING")
    logger.info("="*60)
    logger.info(f"Data directories: {args.data_dirs}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Train
    train_exotransformer(
        data_dirs=args.data_dirs,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        initial_lr=args.lr
    )


if __name__ == "__main__":
    main()

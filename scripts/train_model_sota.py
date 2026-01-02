import os
import glob
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# Configuration Defaults
# ==========================================
DATA_DIR = "notebooks/results_koi"
IMG_SHAPE_GLOBAL = (2001, 1)
IMG_SHAPE_LOCAL = (201, 1)
SCALAR_SHAPE = (7,)

# ==========================================
# Custom Focal Loss
# ==========================================
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        y_true = tf.reshape(y_true, [-1, 1])
        y_pred = tf.reshape(y_pred, [-1, 1])
        
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce = tf.reshape(bce, [-1, 1])
        
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow((1.0 - p_t), self.gamma)
        
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)

# ==========================================
# Enhanced Data Generator
# ==========================================
class ExoplanetDataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size=32, shuffle=True, 
                 augment=False, augment_positive_only=False, oversample=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.augment_positive_only = augment_positive_only
        
        # Balancing Logic
        if oversample:
            pos_indices = np.where(labels == 1)[0]
            neg_indices = np.where(labels == 0)[0]
            
            # Oversample minority (Positive) to match majority (Negative)
            n_samples = len(neg_indices)
            if len(pos_indices) < n_samples:
                pos_indices = np.random.choice(pos_indices, n_samples, replace=True)
            
            balanced_indices = np.concatenate([pos_indices, neg_indices])
            np.random.shuffle(balanced_indices)
            
            self.file_paths = file_paths[balanced_indices]
            self.labels = labels[balanced_indices]
        else:
            self.file_paths = file_paths
            self.labels = labels
            
        self.indices = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_paths = [self.file_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]

        X_global = []
        X_local = []
        X_scalar = []
        y = []

        for i, path in enumerate(batch_paths):
            try:
                with np.load(path) as data:
                    g_view = data['global_view']
                    l_view = data['local_view']
                    scalars = data['scalars']
                    
                    if g_view.shape != IMG_SHAPE_GLOBAL:
                        continue 
                    
                    label = batch_labels[i]
                    
                    # Augmentation Logic
                    if self.augment:
                        should_augment = True
                        if self.augment_positive_only and label != 1:
                            should_augment = False
                        
                        if should_augment:
                            # 1. Flip (Mirroring/Time Reversal)
                            if np.random.rand() > 0.5:
                                g_view = np.flip(g_view, axis=0)
                                l_view = np.flip(l_view, axis=0)
                            
                            # 2. Roll (Phase Shift)
                            shift = np.random.randint(-10, 11)
                            g_view = np.roll(g_view, shift, axis=0)
                            l_view = np.roll(l_view, shift, axis=0)
                            
                            # 3. Flux Jitter
                            jitter = np.random.uniform(0.98, 1.02)
                            g_view = g_view * jitter
                            l_view = l_view * jitter

                    X_global.append(g_view)
                    X_local.append(l_view)
                    X_scalar.append(scalars)
                    y.append(label)

            except Exception:
                continue

        return (np.array(X_global), np.array(X_local), np.array(X_scalar)), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ==========================================
# Transformer Block Helper
# ==========================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

# ==========================================
# Configurable Model Architecture
# ==========================================
def build_transformer_model(num_conv_blocks=3, num_transformer_blocks=1, 
                            reg_strength=1e-5, dropout_rate=0.1, loss_type='bce'):
    
    l2_reg = regularizers.l2(reg_strength)

    # --- Branch 1: Global Transformer ---
    input_global = layers.Input(shape=IMG_SHAPE_GLOBAL, name='global_input')
    x1 = input_global
    
    # Stackable CNN Blocks
    for _ in range(num_conv_blocks):
        x1 = layers.Conv1D(32, 7, padding='same', activation='relu', kernel_regularizer=l2_reg)(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling1D(4)(x1)
        x1 = layers.Dropout(dropout_rate)(x1)
    
    # Stackable Transformer Blocks
    for _ in range(num_transformer_blocks):
        x1 = transformer_encoder(x1, head_size=32, num_heads=2, ff_dim=32, dropout=dropout_rate)
        
    x1 = layers.GlobalAveragePooling1D()(x1)

    # --- Branch 2: Local CNN ---
    input_local = layers.Input(shape=IMG_SHAPE_LOCAL, name='local_input')
    x2 = input_local
    for _ in range(2): # Fixed small CNN for local view
        x2 = layers.Conv1D(16, 3, padding='same', activation='relu', kernel_regularizer=l2_reg)(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.MaxPooling1D(2)(x2)
        x2 = layers.Dropout(dropout_rate)(x2)
    x2 = layers.GlobalMaxPooling1D()(x2)

    # --- Branch 3: Scalars ---
    input_scalar = layers.Input(shape=SCALAR_SHAPE, name='scalar_input')
    x3 = layers.BatchNormalization()(input_scalar)
    x3 = layers.Dense(16, activation='relu', kernel_regularizer=l2_reg)(x3)
    x3 = layers.Dropout(dropout_rate)(x3)

    # --- Fusion ---
    concatenated = layers.Concatenate()([x1, x2, x3])
    fusion = layers.Dense(64, activation='relu', kernel_regularizer=l2_reg)(concatenated)
    fusion = layers.Dropout(0.3)(fusion)
    
    output = layers.Dense(1, activation='sigmoid', name='output')(fusion)

    model = models.Model(inputs=[input_global, input_local, input_scalar], outputs=output)
    
    # Loss Selection
    if loss_type == 'focal':
        loss = FocalLoss(gamma=2.0, alpha=0.25)
    else:
        loss = 'binary_crossentropy'

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss=loss,
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')])
    return model

# ==========================================
# Evaluation Helper
# ==========================================
def evaluate_thresholds(y_true, y_pred_prob, fold_num):
    thresholds = np.arange(0.1, 0.95, 0.05)
    precisions = []
    recalls = []
    f1_scores = []
    
    best_f1 = 0
    best_thresh = 0.5
    
    for t in thresholds:
        y_pred = (y_pred_prob >= t).astype(int)
        p = np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_pred == 1) + 1e-7)
        r = np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-7)
        f1 = 2 * p * r / (p + r + 1e-7)
        
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    # Plot Recall vs Precision
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (Fold {fold_num})')
    plt.grid(True)
    plt.savefig(os.path.join(DATA_DIR, f'pr_curve_fold_{fold_num}.png'))
    plt.close()
    
    return best_thresh, best_f1

# ==========================================
# Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Exoplanet AI Training Script")
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'weighted_bce', 'focal'], help='Loss function type')
    parser.add_argument('--oversample', action='store_true', help='Enable oversampling of positive class')
    parser.add_argument('--augment_positive_only', action='store_true', help='Augment only positive samples')
    parser.add_argument('--num_conv_blocks', type=int, default=3, help='Number of CNN blocks')
    parser.add_argument('--num_transformer_blocks', type=int, default=1, help='Number of Transformer blocks')
    parser.add_argument('--folds', type=int, default=5, help='Number of K-Folds')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    args = parser.parse_args()

    print(f"[INFO] Configuration: {args}")
    
    # 1. Scan Data
    all_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    valid_files = []
    valid_labels = []
    
    print("[INFO] Indexing files...")
    for f in tqdm(all_files):
        try:
            with np.load(f) as data:
                if 'label' in data:
                    valid_labels.append(data['label'])
                    valid_files.append(f)
        except: continue
            
    valid_files = np.array(valid_files)
    valid_labels = np.array(valid_labels)
    
    print(f"[INFO] Found {len(valid_files)} samples.")
    
    # 2. Stratified K-Fold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    
    aggregate_cm = np.zeros((2, 2))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(valid_files, valid_labels)):
        print(f"\n[INFO] Starting Fold {fold+1}/{args.folds}")
        
        X_train_paths, y_train = valid_files[train_idx], valid_labels[train_idx]
        X_val_paths, y_val = valid_files[val_idx], valid_labels[val_idx]

        # Compute Class Weights (if needed)
        cw = None
        if args.loss == 'weighted_bce':
            if args.oversample:
                print("[WARNING] Oversampling is enabled. Disabling Class Weights to prevent double-correction.")
                cw = None
            else:
                weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                cw = dict(enumerate(weights))
                print(f"[INFO] Class Weights: {cw}")

        # Generators
        # Note: If oversample is True, the generator handles balancing internally.
        # If oversample is False, we might rely on class weights or focal loss.
        train_gen = ExoplanetDataGenerator(
            X_train_paths, y_train, 
            batch_size=32, shuffle=True, augment=True, 
            augment_positive_only=args.augment_positive_only,
            oversample=args.oversample
        )
        val_gen = ExoplanetDataGenerator(
            X_val_paths, y_val, 
            batch_size=32, shuffle=False, augment=False
        )

        # Build Model
        model = build_transformer_model(
            num_conv_blocks=args.num_conv_blocks,
            num_transformer_blocks=args.num_transformer_blocks,
            loss_type=args.loss
        )
        
        # Callbacks - Monitor Recall
        callbacks_list = [
            callbacks.ModelCheckpoint(f"best_model_fold_{fold+1}.keras", save_best_only=True, monitor='val_recall', mode='max'),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            callbacks.EarlyStopping(monitor='val_recall', mode='max', patience=15, restore_best_weights=True, verbose=1)
        ]
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.epochs,
            callbacks=callbacks_list,
            class_weight=cw,
            verbose=1
        )
        
        # Evaluation
        print(f"[INFO] Evaluating Fold {fold+1}...")
        y_true_fold = []
        y_pred_prob_fold = []
        
        for i in range(len(val_gen)):
            X_batch, y_batch = val_gen[i]
            preds = model.predict_on_batch(X_batch)
            y_true_fold.extend(y_batch)
            y_pred_prob_fold.extend(preds.flatten())
            
        y_true_fold = np.array(y_true_fold)
        y_pred_prob_fold = np.array(y_pred_prob_fold)
        
        # Threshold Analysis
        best_thresh, best_f1 = evaluate_thresholds(y_true_fold, y_pred_prob_fold, fold+1)
        print(f"[INFO] Fold {fold+1} Best Threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")
        
        # Confusion Matrix at Best Threshold
        y_pred_class = (y_pred_prob_fold >= best_thresh).astype(int)
        cm = confusion_matrix(y_true_fold, y_pred_class)
        aggregate_cm += cm
        
        # Store Metrics
        report = classification_report(y_true_fold, y_pred_class, output_dict=True, zero_division=0)
        fold_metrics.append(report['1']['recall']) # Track Planet Recall

    # Final Aggregation
    print("\n[INFO] Cross-Validation Complete.")
    print(f"[INFO] Average Planet Recall: {np.mean(fold_metrics):.4f}")
    
    # Save Aggregate Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(aggregate_cm, annot=True, fmt='.0f', cmap='Blues')
    plt.title('Aggregate Confusion Matrix (All Folds)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(DATA_DIR, 'confusion_matrix_aggregate.png'))
    print(f"[INFO] Saved aggregate confusion matrix to {DATA_DIR}")

if __name__ == "__main__":
    main()

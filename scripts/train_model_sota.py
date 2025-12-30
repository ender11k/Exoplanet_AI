import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# Configuration
# ==========================================
DATA_DIR = "notebooks/results_koi"
BATCH_SIZE = 32
EPOCHS = 50
IMG_SHAPE_GLOBAL = (2001, 1)
IMG_SHAPE_LOCAL = (201, 1)
SCALAR_SHAPE = (7,)

# ==========================================
# Robust Data Generator (Balanced)
# ==========================================
class ExoplanetDataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size=32, shuffle=True, augment=False, balance=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        
        # --- CRITICAL: 1:1 Balancing Logic ---
        if balance:
            pos_indices = np.where(labels == 1)[0]
            neg_indices = np.where(labels == 0)[0]
            
            # Undersample majority to match minority
            n_samples = len(pos_indices)
            if len(neg_indices) > n_samples:
                neg_indices = np.random.choice(neg_indices, n_samples, replace=False)
            
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
                        continue # Skip bad shapes
                    
                    # Augmentation
                    if self.augment:
                        if np.random.rand() > 0.5: # Flip
                            g_view = np.flip(g_view, axis=0)
                            l_view = np.flip(l_view, axis=0)
                        
                        shift = np.random.randint(-5, 6) # Roll
                        g_view = np.roll(g_view, shift, axis=0)
                        l_view = np.roll(l_view, shift, axis=0)
                        
                        jitter = np.random.uniform(0.99, 1.01) # Jitter
                        g_view = g_view * jitter
                        l_view = l_view * jitter

                    X_global.append(g_view)
                    X_local.append(l_view)
                    X_scalar.append(scalars)
                    y.append(batch_labels[i])

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
    # Attention and Normalization
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

# ==========================================
# SOTA Model Architecture (CNN + Transformer)
# ==========================================
def build_transformer_model():
    l2_reg = regularizers.l2(1e-5)

    # --- Branch 1: Global Transformer ---
    input_global = layers.Input(shape=IMG_SHAPE_GLOBAL, name='global_input')
    x1 = layers.Conv1D(32, 7, padding='same', activation='relu')(input_global)
    x1 = layers.MaxPooling1D(4)(x1)
    x1 = layers.BatchNormalization()(x1)
    
    # Transformer Block
    x1 = transformer_encoder(x1, head_size=32, num_heads=2, ff_dim=32, dropout=0.1)
    x1 = layers.GlobalAveragePooling1D()(x1)

    # --- Branch 2: Local CNN ---
    input_local = layers.Input(shape=IMG_SHAPE_LOCAL, name='local_input')
    x2 = layers.Conv1D(16, 3, padding='same', activation='relu')(input_local)
    x2 = layers.MaxPooling1D(2)(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Conv1D(32, 3, padding='same', activation='relu')(x2)
    x2 = layers.GlobalMaxPooling1D()(x2)

    # --- Branch 3: Scalars ---
    input_scalar = layers.Input(shape=SCALAR_SHAPE, name='scalar_input')
    x3 = layers.BatchNormalization()(input_scalar)
    x3 = layers.Dense(16, activation='relu', kernel_regularizer=l2_reg)(x3)

    # --- Fusion ---
    concatenated = layers.Concatenate()([x1, x2, x3])
    fusion = layers.Dense(64, activation='relu', kernel_regularizer=l2_reg)(concatenated)
    fusion = layers.Dropout(0.3)(fusion)
    
    output = layers.Dense(1, activation='sigmoid', name='output')(fusion)

    model = models.Model(inputs=[input_global, input_local, input_scalar], outputs=output)
    
    # Standard Binary Crossentropy (Best for balanced data)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(name='auc')])
    return model

# ==========================================
# Main Execution
# ==========================================
def main():
    print("[INFO] Starting Transformer-SOTA Training...")
    
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
    print(f"[INFO] Distribution: {dict(zip(*np.unique(valid_labels, return_counts=True)))}")

    # 2. Train on Fold 1 (Single Fold for SOTA Run)
    # Using 80/20 Split
    split_idx = int(len(valid_files) * 0.8)
    indices = np.arange(len(valid_files))
    np.random.shuffle(indices)
    
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_train_paths, y_train = valid_files[train_idx], valid_labels[train_idx]
    X_val_paths, y_val = valid_files[val_idx], valid_labels[val_idx]

    # Generators (Balanced!)
    train_gen = ExoplanetDataGenerator(X_train_paths, y_train, batch_size=BATCH_SIZE, shuffle=True, augment=True, balance=True)
    val_gen = ExoplanetDataGenerator(X_val_paths, y_val, batch_size=BATCH_SIZE, shuffle=False, augment=False, balance=False)

    print(f"[INFO] Training Steps per Epoch: {len(train_gen)}")

    # 3. Build & Train
    model = build_transformer_model()
    
    callbacks_list = [
        # Save the model with the highest AUC (separation power)
        callbacks.ModelCheckpoint("best_model_transformer.keras", save_best_only=True, monitor='val_auc', mode='max'),
        
        # Reduce LR if AUC stops improving
        callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, verbose=1, mode='max'),
        
        # Stop if AUC doesn't improve for 15 epochs
        callbacks.EarlyStopping(monitor='val_auc', patience=15, restore_best_weights=True, verbose=1, mode='max')
    ]
    
    # NO CLASS WEIGHTS (Balancing is handled by Generator)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # 4. Evaluation
    print("[INFO] Evaluating...")
    y_true = []
    y_pred_prob = []
    
    for i in range(len(val_gen)):
        X_batch, y_batch = val_gen[i]
        preds = model.predict_on_batch(X_batch)
        y_true.extend(y_batch)
        y_pred_prob.extend(preds.flatten())
        
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    
    # Report
    print(classification_report(y_true, y_pred_class, target_names=['False Positive', 'Planet']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Transformer Model)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix_sota.png')
    print("Saved confusion_matrix_sota.png")

if __name__ == "__main__":
    main()

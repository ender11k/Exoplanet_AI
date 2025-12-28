import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set random seeds
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
# Robust Data Generator with Augmentation
# ==========================================
class ExoplanetDataGenerator(Sequence):
    def __init__(self, file_paths, batch_size=32, shuffle=True, augment=False):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_paths = [self.file_paths[k] for k in indexes]

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
                    label = data['label']

                    if g_view.shape != IMG_SHAPE_GLOBAL:
                        raise ValueError(f"Global shape mismatch: {g_view.shape}")
                    
                    # Augmentation
                    if self.augment:
                        # 1. Time Mirroring (50% chance)
                        if np.random.rand() > 0.5:
                            g_view = np.flip(g_view, axis=0)
                            l_view = np.flip(l_view, axis=0)
                        
                        # 2. Time Shifting (Roll)
                        shift = np.random.randint(-5, 6)
                        g_view = np.roll(g_view, shift, axis=0)
                        l_view = np.roll(l_view, shift, axis=0)
                        
                        # 3. Flux Jitter
                        jitter = np.random.uniform(0.99, 1.01)
                        g_view = g_view * jitter
                        l_view = l_view * jitter
                        
                        # 4. Gaussian Noise
                        noise_g = np.random.normal(0, 0.001, g_view.shape)
                        noise_l = np.random.normal(0, 0.001, l_view.shape)
                        g_view = g_view + noise_g
                        l_view = l_view + noise_l

                    X_global.append(g_view)
                    X_local.append(l_view)
                    X_scalar.append(scalars)
                    y.append(label)

            except Exception as e:
                print(f"[Warning] Corrupt file {path}: {e}")
                if len(X_global) > 0:
                    X_global.append(X_global[-1])
                    X_local.append(X_local[-1])
                    X_scalar.append(X_scalar[-1])
                    y.append(y[-1])
                else:
                    X_global.append(np.zeros(IMG_SHAPE_GLOBAL))
                    X_local.append(np.zeros(IMG_SHAPE_LOCAL))
                    X_scalar.append(np.zeros(SCALAR_SHAPE))
                    y.append(0)

        return (np.array(X_global), np.array(X_local), np.array(X_scalar)), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ==========================================
# SOTA Model Architecture
# ==========================================
def build_hybrid_model():
    l2_reg = regularizers.l2(0.00001)

    # Branch 1: Global Morphological Tower (CNN)
    input_global = layers.Input(shape=IMG_SHAPE_GLOBAL, name='global_input')
    x1 = layers.Conv1D(16, 3, padding='same', kernel_regularizer=l2_reg)(input_global)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.MaxPooling1D(2)(x1)
    
    x1 = layers.Conv1D(32, 3, padding='same', kernel_regularizer=l2_reg)(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.MaxPooling1D(2)(x1)
    
    x1 = layers.Conv1D(64, 3, padding='same', kernel_regularizer=l2_reg)(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.MaxPooling1D(2)(x1)
    
    # Dual Pooling
    x1_avg = layers.GlobalAveragePooling1D()(x1)
    x1_max = layers.GlobalMaxPooling1D()(x1)
    x1 = layers.Concatenate()([x1_avg, x1_max])

    # Branch 2: Local Morphological Tower (CNN)
    input_local = layers.Input(shape=IMG_SHAPE_LOCAL, name='local_input')
    x2 = layers.Conv1D(16, 3, padding='same', kernel_regularizer=l2_reg)(input_local)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.MaxPooling1D(2)(x2)
    
    x2 = layers.Conv1D(32, 3, padding='same', kernel_regularizer=l2_reg)(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.MaxPooling1D(2)(x2)
    
    # Dual Pooling
    x2_avg = layers.GlobalAveragePooling1D()(x2)
    x2_max = layers.GlobalMaxPooling1D()(x2)
    x2 = layers.Concatenate()([x2_avg, x2_max])

    # Branch 3: Physical Context Tower (Dense + In-Network Norm)
    input_scalar = layers.Input(shape=SCALAR_SHAPE, name='scalar_input')
    x3 = layers.BatchNormalization()(input_scalar)
    x3 = layers.Dense(16, kernel_regularizer=l2_reg)(x3)
    x3 = layers.ReLU()(x3)
    x3 = layers.Dropout(0.3)(x3)

    # Fusion Block
    concatenated = layers.Concatenate()([x1, x2, x3])
    fusion = layers.Dense(64, kernel_regularizer=l2_reg)(concatenated)
    fusion = layers.ReLU()(fusion)
    fusion = layers.Dropout(0.4)(fusion)
    
    output = layers.Dense(1, activation='sigmoid', name='output', kernel_regularizer=l2_reg)(fusion)

    model = models.Model(inputs=[input_global, input_local, input_scalar], outputs=output)
    
    # Label Smoothing in Loss
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(name='auc')])
    return model

# ==========================================
# Main Pipeline
# ==========================================
def main():
    print("[INFO] Starting SOTA Training Pipeline...")
    
    # 1. Pre-Scan
    print("[INFO] Scanning .npz files to collect labels...")
    all_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    
    if not all_files:
        print(f"[Error] No .npz files found in {DATA_DIR}")
        return

    valid_files = []
    valid_labels = []
    
    for f in tqdm(all_files):
        try:
            with np.load(f) as data:
                if 'label' in data:
                    valid_labels.append(data['label'])
                    valid_files.append(f)
        except Exception:
            continue
            
    valid_files = np.array(valid_files)
    valid_labels = np.array(valid_labels)
    
    print(f"[INFO] Successfully indexed {len(valid_files)} samples.")
    unique, counts = np.unique(valid_labels, return_counts=True)
    print(f"[INFO] Class Distribution: {dict(zip(unique, counts))}")

    # 2. Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(valid_files, valid_labels)):
        print(f"\n[INFO] Training Fold {fold+1}/5")
        
        X_train_paths, y_train = valid_files[train_idx], valid_labels[train_idx]
        X_val_paths, y_val = valid_files[val_idx], valid_labels[val_idx]
        
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(valid_labels), y=y_train)
        class_weights = dict(enumerate(weights))
        
        # Boost Planet Sensitivity (Class 1)
        class_weights[1] = class_weights[1] * 1.5
        
        print(f"[INFO] Class Weights: {class_weights}")

        # Generators - Enable Augmentation for Training
        train_gen = ExoplanetDataGenerator(X_train_paths, batch_size=BATCH_SIZE, shuffle=True, augment=True)
        val_gen = ExoplanetDataGenerator(X_val_paths, batch_size=BATCH_SIZE, shuffle=False, augment=False)

        # Build & Train
        model = build_hybrid_model()
        
        callbacks_list = [
            callbacks.ModelCheckpoint(f"best_model_fold_{fold+1}.h5", save_best_only=True, monitor='val_loss'),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
        ]
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1
        )
        
        # 3. Evaluation & Deliverables
        print("[INFO] Generating Advanced Deliverables...")
        
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
        
        # A. Confusion Matrices
        cm = confusion_matrix(y_true, y_pred_class)
        cm_norm = confusion_matrix(y_true, y_pred_class, normalize='true')
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix (Counts)')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1])
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrix_advanced.png')
        print("Saved confusion_matrix_advanced.png")
        
        # B. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig('roc_curve.png')
        print("Saved roc_curve.png")
        
        # C. Classification Report
        report = classification_report(y_true, y_pred_class, target_names=['False Positive', 'Planet'])
        print("\n" + report)
        
        with open("classification_report.txt", "w") as f:
            f.write(report)
        print("Saved classification_report.txt")
        
        break

if __name__ == "__main__":
    main()

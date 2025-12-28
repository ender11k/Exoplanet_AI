import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# Configuration
# ==========================================
DATA_DIR = "notebooks/results_koi"
METADATA_PATH = "data/all_df.csv"
BATCH_SIZE = 32
EPOCHS = 50
IMG_SHAPE_GLOBAL = (2001, 1)
IMG_SHAPE_LOCAL = (201, 1)
SCALAR_FEATURES = ['period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_srad', 'koi_steff', 'koi_slogg']

# ==========================================
# Data Generator
# ==========================================
class ExoplanetDataGenerator(Sequence):
    def __init__(self, file_paths, labels, scalar_data, batch_size=32, shuffle=True):
        self.file_paths = file_paths
        self.labels = labels
        self.scalar_data = scalar_data # Dictionary or DataFrame indexed by ID
        self.batch_size = batch_size
        self.shuffle = shuffle
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
                # Extract ID from filename (e.g., "KIC_10342097.npz" -> 10342097)
                filename = os.path.basename(path)
                target_id = int(filename.split('_')[1].split('.')[0])
                
                # Load Light Curves
                with np.load(path) as data:
                    g_view = data['X_global']
                    l_view = data['X_local']
                    
                    # Validate shapes
                    if g_view.shape != IMG_SHAPE_GLOBAL or l_view.shape != IMG_SHAPE_LOCAL:
                        # Resize or skip? For now, let's skip if shape is wrong to avoid crashing
                        # But resizing is safer for minor mismatches. 
                        # Assuming strict shape for now as per requirements.
                        raise ValueError(f"Shape mismatch: {g_view.shape} vs {IMG_SHAPE_GLOBAL}")

                # Load Scalars
                if target_id in self.scalar_data.index:
                    scalars = self.scalar_data.loc[target_id].values
                else:
                    # Fallback if ID not found in CSV (should not happen with proper filtering)
                    scalars = np.zeros(len(SCALAR_FEATURES))

                X_global.append(g_view)
                X_local.append(l_view)
                X_scalar.append(scalars)
                y.append(batch_labels[i])

            except Exception as e:
                print(f"[Warning] Corrupt file or error processing {path}: {e}")
                # Fallback: Use the previous valid sample to maintain batch size
                # If it's the first sample, we might have an issue, but rare.
                if len(X_global) > 0:
                    X_global.append(X_global[-1])
                    X_local.append(X_local[-1])
                    X_scalar.append(X_scalar[-1])
                    y.append(y[-1])
                else:
                    # If the very first sample fails, push zeros (safe fallback)
                    X_global.append(np.zeros(IMG_SHAPE_GLOBAL))
                    X_local.append(np.zeros(IMG_SHAPE_LOCAL))
                    X_scalar.append(np.zeros(len(SCALAR_FEATURES)))
                    y.append(0)

        return [np.array(X_global), np.array(X_local), np.array(X_scalar)], np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ==========================================
# Model Architecture
# ==========================================
def build_hybrid_model():
    # Branch 1: Global Morphological Tower
    input_global = layers.Input(shape=IMG_SHAPE_GLOBAL, name='global_input')
    x1 = layers.Conv1D(16, 3, padding='same')(input_global)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.MaxPooling1D(2)(x1)
    
    x1 = layers.Conv1D(32, 3, padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.MaxPooling1D(2)(x1)
    
    x1 = layers.Conv1D(64, 3, padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.MaxPooling1D(2)(x1)
    
    x1 = layers.GlobalAveragePooling1D()(x1) # Flattening

    # Branch 2: Local Morphological Tower
    input_local = layers.Input(shape=IMG_SHAPE_LOCAL, name='local_input')
    x2 = layers.Conv1D(16, 3, padding='same')(input_local)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.MaxPooling1D(2)(x2)
    
    x2 = layers.Conv1D(32, 3, padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.MaxPooling1D(2)(x2)
    
    x2 = layers.GlobalAveragePooling1D()(x2)

    # Branch 3: Physical Context Tower
    input_scalar = layers.Input(shape=(len(SCALAR_FEATURES),), name='scalar_input')
    x3 = layers.Dense(16)(input_scalar)
    x3 = layers.ReLU()(x3)
    x3 = layers.Dropout(0.3)(x3)

    # Fusion Block
    concatenated = layers.Concatenate()([x1, x2, x3])
    fusion = layers.Dense(64)(concatenated)
    fusion = layers.ReLU()(fusion)
    fusion = layers.Dropout(0.4)(fusion)
    
    output = layers.Dense(1, activation='sigmoid', name='output')(fusion)

    model = models.Model(inputs=[input_global, input_local, input_scalar], outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

# ==========================================
# Main Training Pipeline
# ==========================================
def main():
    print("[INFO] Loading Metadata...")
    if not os.path.exists(METADATA_PATH):
        print(f"[Error] Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH, low_memory=False)
    
    # Filter for existing files
    print("[INFO] Indexing .npz files...")
    all_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    
    valid_files = []
    valid_labels = []
    valid_ids = []

    for f in all_files:
        try:
            # Filename format: KIC_10342097.npz
            tid = int(os.path.basename(f).split('_')[1].split('.')[0])
            
            # Check if ID exists in dataframe
            row = df[df['target_id'] == tid]
            if not row.empty:
                valid_files.append(f)
                valid_ids.append(tid)
                # Label: 1 if CONFIRMED, 0 otherwise (CANDIDATE might be ambiguous, assuming binary 0/1 from download script logic)
                # Actually, download_data.py saved 'y' in the npz. 
                # But for speed, let's trust the CSV 'label' column if it exists, or re-derive it.
                # The download script derived label from 'koi_disposition' == 'CONFIRMED' -> 1, else 0.
                # Let's replicate that logic to be safe.
                disp = row.iloc[0]['koi_disposition']
                label = 1 if disp == 'CONFIRMED' else 0
                valid_labels.append(label)
        except Exception:
            continue

    print(f"[INFO] Found {len(valid_files)} valid samples aligned with metadata.")
    
    if len(valid_files) == 0:
        print("[Error] No valid data found. Exiting.")
        return

    # Prepare Scalar Data
    print("[INFO] Preprocessing Scalar Features...")
    scalar_df = df[df['target_id'].isin(valid_ids)].set_index('target_id')[SCALAR_FEATURES]
    
    # Handle NaNs
    scalar_df = scalar_df.fillna(scalar_df.median())
    
    # Normalize
    scaler = StandardScaler()
    scalar_scaled = pd.DataFrame(scaler.fit_transform(scalar_df), index=scalar_df.index, columns=SCALAR_FEATURES)

    # Convert to numpy arrays for splitting
    X_paths = np.array(valid_files)
    y = np.array(valid_labels)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # We will train on the first fold for this script to produce the deliverables
    # (Training 5 models sequentially would take too long for a demo script, but the structure supports it)
    print("[INFO] Starting Stratified K-Fold (Fold 1/5)...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_paths, y)):
        if fold > 0: break # Run only first fold
        
        print(f"Training on {len(train_idx)} samples, Validating on {len(val_idx)} samples.")
        
        # Compute Class Weights
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y[train_idx])
        class_weights = dict(enumerate(weights))
        print(f"Class Weights: {class_weights}")

        # Generators
        train_gen = ExoplanetDataGenerator(
            X_paths[train_idx], 
            y[train_idx], 
            scalar_scaled, 
            batch_size=BATCH_SIZE
        )
        val_gen = ExoplanetDataGenerator(
            X_paths[val_idx], 
            y[val_idx], 
            scalar_scaled, 
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        # Build Model
        model = build_hybrid_model()
        
        # Callbacks
        checkpoint = callbacks.ModelCheckpoint(
            f"hybrid_model_fold_{fold+1}.h5", 
            save_best_only=True, 
            monitor='val_loss', 
            mode='min'
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-6,
            verbose=1
        )
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=8, 
            restore_best_weights=True,
            verbose=1
        )

        # Train
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=[checkpoint, reduce_lr, early_stop],
            class_weight=class_weights,
            verbose=1
        )

        # Evaluation & Visualization
        print("[INFO] Generating Deliverables...")
        
        # 1. Training History
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss Curve')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.savefig('training_history.png')
        print("Saved training_history.png")

        # 2. Predictions
        y_pred_prob = model.predict(val_gen)
        # Align labels with generator output (generator might drop last partial batch if not handled, 
        # but our __len__ uses floor, so we must match exactly what generator produced)
        # Safest way: iterate generator to get true labels
        y_true = []
        y_pred_aligned = []
        
        # Re-run val_gen to get aligned labels
        for i in range(len(val_gen)):
            _, batch_y = val_gen[i]
            y_true.extend(batch_y)
            y_pred_aligned.extend(y_pred_prob[i*BATCH_SIZE : (i+1)*BATCH_SIZE])
            
        y_true = np.array(y_true)
        y_pred_aligned = np.array(y_pred_aligned)
        y_pred_class = (y_pred_aligned > 0.5).astype(int)

        # 3. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_class)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False Positive', 'Planet'], yticklabels=['False Positive', 'Planet'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        print("Saved confusion_matrix.png")

        # 4. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_aligned)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        print("Saved roc_curve.png")

        # 5. Metrics Report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_class, target_names=['False Positive', 'Planet']))

if __name__ == "__main__":
    main()

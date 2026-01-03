import os
import glob
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Try importing XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

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
    fusion = layers.Dense(64, activation='relu', kernel_regularizer=l2_reg, name='fusion_layer')(concatenated)
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
# Feature Extraction Helper
# ==========================================
def extract_features(model, generator):
    """Extracts embeddings from the fusion layer and scalar inputs."""
    feature_extractor = models.Model(inputs=model.input, outputs=model.get_layer('fusion_layer').output)
    
    embeddings = []
    scalars = []
    labels = []
    
    # Note: Generator must be unshuffled to align with indices
    for i in range(len(generator)):
        (X_global, X_local, X_scalar), y = generator[i]
        batch_embeddings = feature_extractor.predict_on_batch((X_global, X_local, X_scalar))
        
        embeddings.append(batch_embeddings)
        scalars.append(X_scalar)
        labels.append(y)
        
    embeddings = np.vstack(embeddings)
    scalars = np.vstack(scalars)
    labels = np.concatenate(labels)
    
    features = np.hstack([scalars, embeddings])
    return features, labels

# ==========================================
# XGBoost Tuning Helper
# ==========================================
def tune_xgboost(X_train, y_train):
    print("[INFO] Tuning XGBoost Hyperparameters...")
    param_dist = {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
    }
    
    clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', use_label_encoder=False)
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, scoring='roc_auc', cv=3, verbose=1, n_jobs=-1)
    random_search.fit(X_train, y_train)
    
    print(f"[INFO] Best Params: {random_search.best_params_}")
    return random_search.best_params_

# ==========================================
# Evaluation Helper
# ==========================================
def evaluate_thresholds(y_true, y_pred_prob, fold_num, model_name="CNN"):
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
            
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='.', label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(DATA_DIR, f'pr_curve_{model_name}_fold_{fold_num}.png'))
    plt.close()
    
    return best_thresh, best_f1

# ==========================================
# Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Exoplanet AI Hybrid Training Script")
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'weighted_bce', 'focal'], help='Loss function type')
    parser.add_argument('--oversample', action='store_true', help='Enable oversampling of positive class')
    parser.add_argument('--augment_positive_only', action='store_true', help='Augment only positive samples')
    parser.add_argument('--num_conv_blocks', type=int, default=3, help='Number of CNN blocks')
    parser.add_argument('--num_transformer_blocks', type=int, default=1, help='Number of Transformer blocks')
    parser.add_argument('--folds', type=int, default=5, help='Number of K-Folds')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    
    # Hybrid Arguments
    parser.add_argument('--hybrid', action='store_true', help='Enable Hybrid CNN + XGBoost training')
    parser.add_argument('--ensemble_method', type=str, default='average', choices=['average', 'stacking', 'weighted_average', 'cnn_only', 'xgboost_only'], help='Ensemble method')
    parser.add_argument('--meta_learner', type=str, default='logistic', choices=['logistic', 'ridge'], help='Meta learner type for stacking')
    parser.add_argument('--tune_xgboost', action='store_true', help='Run RandomizedSearchCV for XGBoost')
    parser.add_argument('--xgb_rounds', type=int, default=100, help='XGBoost boosting rounds')
    parser.add_argument('--xgb_early_stopping', type=int, default=10, help='XGBoost early stopping rounds')
    parser.add_argument('--xgb_max_depth', type=int, default=6, help='XGBoost max depth')
    parser.add_argument('--xgb_eta', type=float, default=0.1, help='XGBoost learning rate')
    parser.add_argument('--cnn_weight', type=float, default=0.5, help='Weight for CNN in weighted average')
    parser.add_argument('--xgb_weight', type=float, default=0.5, help='Weight for XGBoost in weighted average')
    
    args = parser.parse_args()

    if args.hybrid and not XGB_AVAILABLE:
        print("[ERROR] XGBoost is not installed. Please install it via 'pip install xgboost' or disable --hybrid.")
        return

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
    
    # OOF Prediction Arrays
    oof_cnn = np.zeros(len(valid_files))
    oof_xgb = np.zeros(len(valid_files))
    oof_y = np.zeros(len(valid_files))
    
    # To track if we actually filled the arrays (in case of skipped files in generator)
    # But generator logic is robust enough. We assume alignment.
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(valid_files, valid_labels)):
        print(f"\n[INFO] Starting Fold {fold+1}/{args.folds}")
        
        X_train_paths, y_train = valid_files[train_idx], valid_labels[train_idx]
        X_val_paths, y_val = valid_files[val_idx], valid_labels[val_idx]

        # Compute Class Weights
        cw = None
        if args.loss == 'weighted_bce':
            if args.oversample:
                cw = None
            else:
                weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                cw = dict(enumerate(weights))

        # Generators
        train_gen = ExoplanetDataGenerator(
            X_train_paths, y_train, 
            batch_size=32, shuffle=True, augment=True, 
            augment_positive_only=args.augment_positive_only,
            oversample=args.oversample
        )
        # Validation Generator MUST NOT SHUFFLE to align with OOF indices
        val_gen = ExoplanetDataGenerator(
            X_val_paths, y_val, 
            batch_size=32, shuffle=False, augment=False
        )

        # Build & Train CNN
        model = build_transformer_model(
            num_conv_blocks=args.num_conv_blocks,
            num_transformer_blocks=args.num_transformer_blocks,
            loss_type=args.loss
        )
        
        callbacks_list = [
            callbacks.ModelCheckpoint(f"best_model_fold_{fold+1}.keras", save_best_only=True, monitor='val_recall', mode='max'),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            callbacks.EarlyStopping(monitor='val_recall', mode='max', patience=15, restore_best_weights=True, verbose=1)
        ]
        
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.epochs,
            callbacks=callbacks_list,
            class_weight=cw,
            verbose=1
        )
        
        # --- OOF Predictions: CNN ---
        # Predict on validation set
        # Note: val_gen yields batches. We need to flatten.
        # The generator length might be slightly less than len(val_idx) due to batch flooring.
        # To be precise, we should iterate carefully or use predict_generator with steps.
        # However, our generator drops the last partial batch. This causes misalignment with val_idx.
        # FIX: We must ensure we predict on ALL validation samples.
        # We will manually load the validation files for prediction to ensure 1:1 mapping.
        
        print("[INFO] Generating OOF Predictions for Fold...")
        
        # Helper to predict on list of files without dropping last batch
        def predict_on_files(file_paths, model_func):
            preds = []
            actuals = []
            # Process in chunks to avoid OOM
            chunk_size = 32
            for i in range(0, len(file_paths), chunk_size):
                batch_files = file_paths[i:i+chunk_size]
                X_g, X_l, X_s = [], [], []
                y_b = []
                for p in batch_files:
                    try:
                        with np.load(p) as d:
                            X_g.append(d['global_view'])
                            X_l.append(d['local_view'])
                            X_s.append(d['scalars'])
                            if 'label' in d: y_b.append(d['label'])
                            else: y_b.append(0) # Should not happen
                    except:
                        # If file fails, append zeros to keep alignment
                        X_g.append(np.zeros(IMG_SHAPE_GLOBAL))
                        X_l.append(np.zeros(IMG_SHAPE_LOCAL))
                        X_s.append(np.zeros(SCALAR_SHAPE))
                        y_b.append(0)
                
                if len(X_g) > 0:
                    p_batch = model_func((np.array(X_g), np.array(X_l), np.array(X_s)))
                    preds.extend(p_batch.flatten())
                    actuals.extend(y_b)
            return np.array(preds), np.array(actuals)

        # CNN OOF
        cnn_preds_fold, y_true_fold = predict_on_files(X_val_paths, lambda x: model.predict_on_batch(x))
        
        # Store in global OOF arrays
        # Note: predict_on_files returns exactly len(X_val_paths) predictions
        oof_cnn[val_idx] = cnn_preds_fold
        oof_y[val_idx] = y_true_fold

        # --- OOF Predictions: XGBoost ---
        if args.hybrid:
            # Extract Features for Train (using generator is fine for training as dropping few samples is ok)
            X_train_xgb, y_train_xgb = extract_features(model, train_gen)
            
            # Extract Features for Val (Must use all files)
            # We need a custom extractor that doesn't drop samples
            def extract_features_all(file_paths):
                feats = []
                # Use the sub-model
                extractor = models.Model(inputs=model.input, outputs=model.get_layer('fusion_layer').output)
                
                chunk_size = 32
                for i in range(0, len(file_paths), chunk_size):
                    batch_files = file_paths[i:i+chunk_size]
                    X_g, X_l, X_s = [], [], []
                    for p in batch_files:
                        try:
                            with np.load(p) as d:
                                X_g.append(d['global_view'])
                                X_l.append(d['local_view'])
                                X_s.append(d['scalars'])
                        except:
                            X_g.append(np.zeros(IMG_SHAPE_GLOBAL))
                            X_l.append(np.zeros(IMG_SHAPE_LOCAL))
                            X_s.append(np.zeros(SCALAR_SHAPE))
                    
                    if len(X_g) > 0:
                        emb = extractor.predict_on_batch((np.array(X_g), np.array(X_l), np.array(X_s)))
                        batch_feats = np.hstack([np.array(X_s), emb])
                        feats.append(batch_feats)
                return np.vstack(feats)

            X_val_xgb = extract_features_all(X_val_paths)
            
            # Tune or Train
            xgb_params = {
                'max_depth': args.xgb_max_depth,
                'eta': args.xgb_eta,
                'objective': 'binary:logistic',
                'eval_metric': 'auc'
            }
            
            if args.tune_xgboost:
                best_params = tune_xgboost(X_train_xgb, y_train_xgb)
                xgb_params.update(best_params)
            
            dtrain = xgb.DMatrix(X_train_xgb, label=y_train_xgb)
            dval = xgb.DMatrix(X_val_xgb, label=y_true_fold)
            
            bst = xgb.train(xgb_params, dtrain, num_boost_round=args.xgb_rounds, 
                            evals=[(dval, 'eval')], early_stopping_rounds=args.xgb_early_stopping, verbose_eval=False)
            
            xgb_preds_fold = bst.predict(dval)
            oof_xgb[val_idx] = xgb_preds_fold
            
    # --- End of CV Loop ---
    
    # --- Meta-Learner & Final Evaluation ---
    print("\n[INFO] Training Meta-Learner on OOF Predictions...")
    
    # Filter out any indices that might not have been filled (e.g. if file load failed completely)
    # But we initialized with zeros, so it's fine.
    
    X_meta = np.column_stack((oof_cnn, oof_xgb))
    y_meta = oof_y
    
    final_preds = np.zeros_like(y_meta)
    
    if args.ensemble_method == 'cnn_only':
        final_preds = oof_cnn
    elif args.ensemble_method == 'xgboost_only':
        final_preds = oof_xgb
    elif args.ensemble_method == 'average':
        final_preds = (oof_cnn + oof_xgb) / 2.0
    elif args.ensemble_method == 'weighted_average':
        final_preds = (args.cnn_weight * oof_cnn + args.xgb_weight * oof_xgb) / (args.cnn_weight + args.xgb_weight)
    elif args.ensemble_method == 'stacking':
        # Train Meta Learner on OOF
        # To evaluate the STACKER itself, we technically need nested CV or just report OOF score.
        # Standard practice: Train on OOF, report score on OOF (optimistic) or use cross_val_predict again.
        # We will use cross_val_predict to get unbiased predictions from the meta-learner
        from sklearn.model_selection import cross_val_predict
        
        if args.meta_learner == 'logistic':
            meta = LogisticRegression()
            # Get unbiased predictions from the meta-learner
            final_preds = cross_val_predict(meta, X_meta, y_meta, cv=5, method='predict_proba')[:, 1]
        else:
            meta = RidgeClassifier()
            # RidgeClassifier does not support predict_proba, use decision_function
            final_preds = cross_val_predict(meta, X_meta, y_meta, cv=5, method='decision_function')
            # Normalize decision function to 0-1 range for consistency
            final_preds = (final_preds - final_preds.min()) / (final_preds.max() - final_preds.min())
        
        # Fit final meta learner for saving
        meta.fit(X_meta, y_meta)
        print(f"[INFO] Meta-Learner Coefficients: {meta.coef_}")

    # Final Metrics
    print("\n[INFO] Final Ensemble Performance (OOF):")
    best_thresh, best_f1 = evaluate_thresholds(y_meta, final_preds, "Global_Ensemble", "Ensemble")
    print(f"[INFO] Best Threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")
    
    y_pred_class = (final_preds >= best_thresh).astype(int)
    print(classification_report(y_meta, y_pred_class, target_names=['False Positive', 'Planet']))
    
    cm = confusion_matrix(y_meta, y_pred_class)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Aggregate Confusion Matrix (OOF)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(DATA_DIR, 'confusion_matrix_final.png'))

if __name__ == "__main__":
    main()

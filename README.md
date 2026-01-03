# Exoplanet AI: SOTA Training Pipeline

This repository contains a state-of-the-art (SOTA) hybrid CNN-Transformer model for exoplanet detection using Kepler and TESS data.

## Key Features

*   **Hybrid Architecture**: Combines Global and Local view CNNs with a Transformer Encoder to capture both morphological shapes and long-range temporal dependencies (transit dips).
*   **Robust Stacking Ensemble**: Implements **Out-of-Fold (OOF)** prediction collection to train a meta-learner (Logistic Regression) without data leakage.
*   **XGBoost Integration**: Uses learned embeddings from the Deep Learning model as features for Gradient Boosting.
*   **Advanced Balancing**: Supports both Undersampling and Oversampling strategies.
*   **Custom Loss Functions**: Includes Focal Loss ($\gamma=2.0$) to focus learning on hard-to-classify examples.

## Usage

The main training script is `scripts/train_model_sota.py`.

### Basic Run (Default)
Runs with standard Binary Cross Entropy, 5-Fold CV, and no oversampling.

```bash
python scripts/train_model_sota.py
```

### Recommended SOTA Configuration (Hybrid Stacking)
To achieve the best results, use the **Hybrid Stacking** pipeline. This trains the CNN/Transformer on 5 folds, collects OOF predictions, trains an XGBoost model on the embeddings, and finally trains a Meta-Learner to combine them.

```bash
python scripts/train_model_sota.py --hybrid --ensemble_method stacking --meta_learner logistic --loss focal --oversample --augment_positive_only --folds 5
```

### Hyperparameter Tuning
To automatically tune XGBoost hyperparameters (Random Search) inside each fold:

```bash
python scripts/train_model_sota.py --hybrid --tune_xgboost ...
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--hybrid` | `flag` | `False` | Enables the Hybrid pipeline (CNN + XGBoost). |
| `--ensemble_method` | `str` | `average` | Method to combine predictions: `average`, `stacking`, `weighted_average`, `cnn_only`, `xgboost_only`. |
| `--meta_learner` | `str` | `logistic` | Meta-learner for stacking: `logistic` or `ridge`. |
| `--tune_xgboost` | `flag` | `False` | Runs RandomizedSearchCV for XGBoost hyperparameters. |
| `--loss` | `str` | `bce` | Loss function: `bce`, `weighted_bce`, or `focal`. |
| `--oversample` | `flag` | `False` | If set, oversamples the positive class to match the negative class count (1:1 ratio). |
| `--augment_positive_only` | `flag` | `False` | If set, applies augmentations (flip, roll, jitter) ONLY to positive samples. |
| `--xgb_rounds` | `int` | `100` | XGBoost boosting rounds. |
| `--xgb_early_stopping` | `int` | `10` | XGBoost early stopping rounds. |

## Evaluation Outputs

Results are saved in `notebooks/results_koi/`:
*   `best_model_fold_X.keras`: Best model checkpoint for each fold.
*   `pr_curve_Global_Ensemble_fold_Ensemble.png`: Precision-Recall curve for the final ensemble (OOF).
*   `confusion_matrix_final.png`: Aggregate confusion matrix for the final ensemble.

## Dependencies

*   TensorFlow 2.x
*   XGBoost (`pip install xgboost`)
*   NumPy, Pandas, Matplotlib, Seaborn
*   Scikit-learn
*   Tqdm

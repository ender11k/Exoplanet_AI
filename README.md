# Exoplanet AI: SOTA Training Pipeline

This repository contains a state-of-the-art (SOTA) hybrid CNN-Transformer model for exoplanet detection using Kepler and TESS data.

## Key Features

*   **Hybrid Architecture**: Combines Global and Local view CNNs with a Transformer Encoder to capture both morphological shapes and long-range temporal dependencies (transit dips).
*   **Advanced Balancing**: Supports both Undersampling and Oversampling strategies to handle severe class imbalance.
*   **Custom Loss Functions**: Includes Focal Loss ($\gamma=2.0$) to focus learning on hard-to-classify examples.
*   **Robust Evaluation**: Uses Stratified K-Fold Cross-Validation and threshold optimization to maximize Recall.

## Usage

The main training script is `scripts/train_model_sota.py`. It supports various CLI arguments for customization.

### Basic Run (Default)
Runs with standard Binary Cross Entropy, 5-Fold CV, and no oversampling (uses undersampling or raw data depending on generator defaults, but note: the script defaults to `oversample=False` which means raw data unless modified).

```bash
python scripts/train_model_sota.py
```

### Recommended SOTA Configuration
To achieve the best results (High Recall), use **Focal Loss**, **Oversampling**, and **Positive-Only Augmentation**:

```bash
python scripts/train_model_sota.py --loss focal --oversample --augment_positive_only --folds 5
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--loss` | `str` | `bce` | Loss function: `bce`, `weighted_bce`, or `focal`. |
| `--oversample` | `flag` | `False` | If set, oversamples the positive class to match the negative class count (1:1 ratio). |
| `--augment_positive_only` | `flag` | `False` | If set, applies augmentations (flip, roll, jitter) ONLY to positive samples to increase their diversity without distorting negatives. |
| `--num_conv_blocks` | `int` | `3` | Number of CNN blocks in the Global branch. |
| `--num_transformer_blocks` | `int` | `1` | Number of Transformer Encoder blocks. |
| `--folds` | `int` | `5` | Number of folds for Stratified K-Fold Cross-Validation. |
| `--epochs` | `int` | `50` | Number of training epochs per fold. |

## Evaluation Outputs

Results are saved in `notebooks/results_koi/`:
*   `best_model_fold_X.keras`: Best model checkpoint for each fold.
*   `pr_curve_fold_X.png`: Precision-Recall curve for each fold.
*   `confusion_matrix_aggregate.png`: Aggregated confusion matrix across all folds.

## Dependencies

*   TensorFlow 2.x
*   NumPy, Pandas, Matplotlib, Seaborn
*   Scikit-learn
*   Tqdm

No extra installation is required for Focal Loss as it is implemented as a custom class within the script.

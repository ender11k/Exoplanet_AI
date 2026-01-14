# NASA-Level Optimal Exoplanet Detection Pipeline
## IEEE Research Paper Quality Implementation Plan

**Project:** Exoplanet Transit Signal Classification using Deep Learning  
**Target:** IEEE/ApJ Publication Standard  
**Author:** NASA Internship Project  
**Date:** January 2026

---

## Executive Summary

This document outlines a comprehensive, production-grade approach to building a state-of-the-art (SOTA) exoplanet detection system based on:

1. **ExoMiner** (Valizadegan et al., 2021) - NASA Ames Research Center
2. **AstroNet** (Shallue & Vanderburg, 2018) - Google Brain / Harvard
3. **Astronet-Triage** (Yu et al., 2019) - NASA/Caltech

These papers achieved **>98% precision** and have been used to **validate 301+ new exoplanets**.

---

## Table of Contents

1. [Architecture Design](#1-architecture-design)
2. [Data Pipeline](#2-data-pipeline)
3. [Training Strategy](#3-training-strategy)
4. [Evaluation Framework](#4-evaluation-framework)
5. [Explainability Module](#5-explainability-module)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Expected Results](#7-expected-results)

---

## 1. Architecture Design

### 1.1 ExoMiner-Inspired Multi-Branch Architecture

The key insight from ExoMiner is **mimicking how domain experts vet transit signals**. Each branch processes a different diagnostic view:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXOPLANET CLASSIFIER v2.0                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   GLOBAL    │  │   LOCAL     │  │  SECONDARY  │  │   STELLAR   │        │
│  │   VIEW      │  │   VIEW      │  │   ECLIPSE   │  │  CENTROID   │        │
│  │  (Full LC)  │  │ (Transit)   │  │  (Phase     │  │   OFFSET    │        │
│  │             │  │             │  │   0.5)      │  │             │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         ▼                ▼                ▼                ▼               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  1D-CNN +   │  │  1D-CNN +   │  │  1D-CNN +   │  │  Dense +    │        │
│  │  SE Block   │  │  SE Block   │  │  SE Block   │  │  BN         │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         └────────────────┼────────────────┼────────────────┘               │
│                          ▼                                                  │
│                 ┌─────────────────┐                                        │
│                 │   ATTENTION     │                                        │
│                 │   FUSION        │                                        │
│                 │   (Cross-View)  │                                        │
│                 └────────┬────────┘                                        │
│                          │                                                  │
│                          ▼                                                  │
│                 ┌─────────────────┐                                        │
│                 │  SCALAR FEATS   │◄── [Period, Depth, Duration, Prad,     │
│                 │  INTEGRATION    │     Srad, Teff, SNR, MES, ...]         │
│                 └────────┬────────┘                                        │
│                          │                                                  │
│                          ▼                                                  │
│                 ┌─────────────────┐      ┌─────────────────┐               │
│                 │   FINAL HEAD   │─────►│ P(Planet) ∈[0,1]│               │
│                 │   (Dense+Sig)   │      └─────────────────┘               │
│                 └─────────────────┘                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Architectural Components

#### A. Squeeze-and-Excitation (SE) Blocks
```python
# Channel attention for light curve features
SE_ratio = 16  # Reduction ratio
# Recalibrates channel-wise feature responses
```

#### B. Multi-Scale Convolutions
```python
# Capture transit signatures at different scales
kernel_sizes = [3, 5, 7, 11]  # Multi-resolution
# Small kernels → Sharp transit edges
# Large kernels → Broad stellar variability
```

#### C. Residual Connections
```python
# Gradient flow preservation for deeper networks
# x_out = F(x) + x  # Skip connection
```

### 1.3 Input Specifications

| View | Shape | Description | Purpose |
|------|-------|-------------|---------|
| Global | (2001, 1) | Full phase-folded LC | Overall transit morphology |
| Local | (201, 1) | 4× transit duration | Transit shape details |
| Secondary | (201, 1) | Phase 0.5 centered | Detect secondary eclipses (EB indicator) |
| Odd Transit | (201, 1) | Odd-numbered transits | Transit timing variations |
| Even Transit | (201, 1) | Even-numbered transits | Eclipsing binary detection |
| Scalars | (15,) | Catalog parameters | Physical constraints |
| Centroid | (2, 201) | Row/Col pixel shift | Background contamination |

---

## 2. Data Pipeline

### 2.1 Multi-Stage Data Acquisition

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA ACQUISITION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Stage 1: Catalog Collection                                           │
│  ├── NASA Exoplanet Archive (TAP Query)                                │
│  │   ├── Kepler: cumulative (KOI) table                               │
│  │   ├── K2: k2candidates table                                       │
│  │   └── TESS: TOI table                                              │
│  └── Kepler TCE table (34,000+ candidates)                            │
│                                                                         │
│  Stage 2: Light Curve Download                                         │
│  ├── MAST Portal via lightkurve                                       │
│  │   ├── Kepler: kplr*_llc.fits (Long Cadence)                       │
│  │   ├── K2: ktwo*_llc.fits                                          │
│  │   └── TESS: tess*_lc.fits (2-min cadence)                         │
│  └── Caching: D:\.lightkurve2 (persistent storage)                    │
│                                                                         │
│  Stage 3: Preprocessing                                                │
│  ├── Detrending: Savitzky-Golay (window=101)                          │
│  ├── Outlier removal: 3σ clipping                                     │
│  ├── Gap filling: Linear interpolation                                │
│  └── Phase folding: Period & epoch alignment                          │
│                                                                         │
│  Stage 4: View Generation                                              │
│  ├── Global view: 2001-bin phase fold                                 │
│  ├── Local view: 201-bin transit window                               │
│  ├── Secondary eclipse view: Phase 0.5 centered                       │
│  ├── Odd/Even views: Transit parity splitting                         │
│  └── Centroid time series: Motion test extraction                     │
│                                                                         │
│  Stage 5: Feature Extraction                                           │
│  ├── Catalog scalars: Period, Duration, Depth, Prad, etc.            │
│  ├── Derived features: SNR, MES, transit count                        │
│  └── Quality flags: Crowding, contamination                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Class Balance Strategy

**Problem:** Extreme imbalance (5-10% positive class)

| Strategy | Pros | Cons | Recommended |
|----------|------|------|-------------|
| **Oversampling** | Simple | Overfitting risk | ✓ (with augmentation) |
| **SMOTE** | Synthetic diversity | Not for time series | ✗ |
| **Focal Loss** | Hard example mining | Hyperparameter sensitive | ✓ |
| **Class Weights** | No data change | May underperform | ✓ (baseline) |
| **Threshold Tuning** | Post-hoc adjustment | Doesn't fix learning | ✓ (evaluation) |

**Our Approach: Combined Strategy**
```python
# 1. Oversample positive class to 1:2 ratio (not 1:1 to preserve learning signal)
# 2. Apply focal loss (γ=2.0, α=0.25)
# 3. Augment ONLY positive samples (physics-preserving transforms)
# 4. Optimize threshold on validation PR curve
```

### 2.3 Data Augmentation (Physics-Preserving)

```python
class PhysicsPreservingAugmentation:
    """
    Augmentations that preserve astrophysical validity
    """
    
    # ✓ VALID AUGMENTATIONS
    def time_reversal(self, flux):
        """Mirror flip - physically valid (time symmetry)"""
        return np.flip(flux)
    
    def phase_shift(self, flux, max_shift=0.02):
        """Small roll - simulates epoch uncertainty"""
        shift = int(len(flux) * max_shift * np.random.uniform(-1, 1))
        return np.roll(flux, shift)
    
    def flux_scaling(self, flux, range=(0.98, 1.02)):
        """Global scaling - simulates calibration variance"""
        return flux * np.random.uniform(*range)
    
    def gaussian_noise(self, flux, snr_factor=0.1):
        """Add realistic photon noise"""
        noise = np.random.normal(0, snr_factor * np.std(flux), flux.shape)
        return flux + noise
    
    def transit_depth_jitter(self, flux, depth_var=0.05):
        """Slight depth variation - simulates limb darkening uncertainty"""
        transit_mask = flux < (flux.mean() - 2*flux.std())
        depth_factor = 1 + np.random.uniform(-depth_var, depth_var)
        flux[transit_mask] *= depth_factor
        return flux
    
    # ✗ INVALID AUGMENTATIONS (Don't use)
    # - Random cropping (destroys phase alignment)
    # - Vertical flip (inverts transit → unphysical)
    # - Large time stretching (changes period)
```

---

## 3. Training Strategy

### 3.1 Two-Stage Learning

**Stage 1: Pre-training on Synthetic Data**
```python
# Generate synthetic transits with known parameters
# Use BLS (Box Least Squares) transit model
# Inject into real stellar variability backgrounds
# Train model to convergence (warm start)
```

**Stage 2: Fine-tuning on Real Labels**
```python
# Load pre-trained weights
# Fine-tune on KOI/TOI labels
# Use smaller learning rate (1e-5)
# Focus on hard examples via focal loss
```

### 3.2 Cross-Validation Protocol

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NESTED CROSS-VALIDATION (NCv)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  OUTER LOOP: 5-Fold Stratified (Model Evaluation)                      │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Fold 1: Train[2,3,4,5] → Test[1] → OOF_pred[1]                  │ │
│  │  Fold 2: Train[1,3,4,5] → Test[2] → OOF_pred[2]                  │ │
│  │  Fold 3: Train[1,2,4,5] → Test[3] → OOF_pred[3]                  │ │
│  │  Fold 4: Train[1,2,3,5] → Test[4] → OOF_pred[4]                  │ │
│  │  Fold 5: Train[1,2,3,4] → Test[5] → OOF_pred[5]                  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                           ↓                                            │
│  INNER LOOP: 3-Fold (Hyperparameter Tuning) - Optional                 │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Within each outer train set:                                     │ │
│  │  - Grid search over learning_rate, dropout, etc.                 │ │
│  │  - Select best hyperparameters                                    │ │
│  │  - Retrain on full outer training set                            │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                           ↓                                            │
│  AGGREGATION:                                                          │
│  - Concatenate all OOF predictions                                     │
│  - Report metrics on ENTIRE dataset (no leakage)                       │
│  - Statistical significance via bootstrap CI                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Ensemble Architecture

```python
# Level 1: Base Models (trained on CV folds)
base_models = [
    'CNN_Global_Local',       # Your current architecture
    'Transformer_Global',     # Self-attention on global view
    'XGBoost_Scalars',        # Gradient boosting on features
    'LightGBM_Embeddings',    # GB on CNN embeddings
]

# Level 2: Meta-Learner (trained on OOF predictions)
meta_learner = LogisticRegression(C=1.0)  # Simple, interpretable

# Level 3: Threshold Optimization
# Use PR curve on held-out set to find optimal threshold
# Target: Maximize F1 or fix precision at 0.99
```

### 3.4 Learning Rate Schedule

```python
# Cosine Annealing with Warm Restarts
schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-3,
    first_decay_steps=1000,
    t_mul=2.0,    # Double period after each restart
    m_mul=0.9,    # Reduce max LR by 10% each restart
    alpha=1e-6    # Minimum LR
)

# Or: OneCycleLR (proven effective for image classification)
# Peak at epoch 30% → gradual decay
```

---

## 4. Evaluation Framework

### 4.1 Metrics Suite (NASA Standard)

| Metric | Formula | Target | Priority |
|--------|---------|--------|----------|
| **Precision** | TP / (TP + FP) | ≥ 0.99 | Critical |
| **Recall** | TP / (TP + FN) | ≥ 0.90 | High |
| **F1 Score** | 2 × P × R / (P + R) | ≥ 0.85 | High |
| **PR-AUC** | Area under PR curve | ≥ 0.95 | Primary |
| **ROC-AUC** | Area under ROC curve | ≥ 0.98 | Secondary |
| **Reliability** | Calibration curve slope | ≈ 1.0 | Critical |

### 4.2 Ranking Metric (ExoMiner Standard)

```python
def recall_at_precision(y_true, y_pred_prob, target_precision=0.99):
    """
    The key metric from ExoMiner paper.
    At fixed precision, what recall do we achieve?
    
    ExoMiner: Recall = 93.6% @ Precision = 99%
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    
    # Find threshold that gives target precision
    valid_idx = np.where(precision >= target_precision)[0]
    if len(valid_idx) == 0:
        return 0.0, 1.0  # Can't achieve target precision
    
    best_recall = recall[valid_idx].max()
    best_thresh = thresholds[valid_idx[np.argmax(recall[valid_idx])]]
    
    return best_recall, best_thresh
```

### 4.3 Calibration (Reliability Diagram)

```python
def plot_reliability_diagram(y_true, y_pred_prob, n_bins=10):
    """
    A well-calibrated model should have predictions
    that match observed frequencies.
    
    E.g., if model predicts 0.7 → 70% should be true positives
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    observed_freq = []
    predicted_freq = []
    
    for i in range(n_bins):
        mask = (y_pred_prob >= bin_edges[i]) & (y_pred_prob < bin_edges[i+1])
        if mask.sum() > 0:
            observed_freq.append(y_true[mask].mean())
            predicted_freq.append(y_pred_prob[mask].mean())
    
    # Plot: Perfect calibration = diagonal line
    plt.plot([0, 1], [0, 1], '--', label='Perfect Calibration')
    plt.plot(predicted_freq, observed_freq, 'o-', label='Model')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title('Reliability Diagram')
```

### 4.4 Statistical Significance

```python
# Bootstrap Confidence Intervals
def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=0.95):
    scores = []
    n = len(y_true)
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        score = metric_fn(y_true[idx], y_pred[idx])
        scores.append(score)
    
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    
    return np.mean(scores), (lower, upper)

# Report: F1 = 0.87 (95% CI: 0.84 - 0.90)
```

---

## 5. Explainability Module

### 5.1 Why Explainability Matters for NASA

1. **Scientific Validation**: Domain experts must verify model decisions
2. **Discovery Trust**: New planet candidates need justification
3. **Publication Requirements**: Journals require interpretable results
4. **Error Analysis**: Understand failure modes (eclipsing binaries, systematics)

### 5.2 Integrated Gradients (Primary Method)

```python
def integrated_gradients(model, input_data, baseline=None, steps=50):
    """
    Attributes each input feature's contribution to the prediction.
    
    For transit detection:
    - High attribution at transit dip → Model learned correct feature
    - High attribution at stellar variability → Potential overfitting
    """
    if baseline is None:
        baseline = np.zeros_like(input_data)
    
    # Interpolate between baseline and input
    alphas = np.linspace(0, 1, steps)
    interpolated = [baseline + alpha * (input_data - baseline) for alpha in alphas]
    
    # Compute gradients
    with tf.GradientTape() as tape:
        inputs = tf.convert_to_tensor(interpolated)
        tape.watch(inputs)
        outputs = model(inputs)
    
    gradients = tape.gradient(outputs, inputs)
    
    # Average gradients and multiply by (input - baseline)
    avg_gradients = tf.reduce_mean(gradients, axis=0)
    attributions = (input_data - baseline) * avg_gradients
    
    return attributions
```

### 5.3 Branch Contribution Analysis

```python
def analyze_branch_contributions(model, sample):
    """
    ExoMiner-style analysis: Which branch contributed most?
    
    Interpretation:
    - Global view dominant → Full LC shape is key
    - Local view dominant → Transit morphology matters
    - Scalars dominant → Physical parameters are decisive
    """
    # Create modified inputs (zero out each branch)
    contributions = {}
    
    baseline_pred = model.predict(sample)
    
    for branch_name, branch_idx in [('global', 0), ('local', 1), ('scalar', 2)]:
        modified = list(sample)
        modified[branch_idx] = np.zeros_like(sample[branch_idx])
        modified_pred = model.predict(modified)
        
        contributions[branch_name] = float(baseline_pred - modified_pred)
    
    return contributions
```

---

## 6. Implementation Roadmap

### Phase 1: Data Foundation (Week 1-2)

| Task | Status | Priority |
|------|--------|----------|
| Move lightkurve cache to D: drive | 🔴 Pending | Critical |
| Retry failed downloads (899 targets) | 🔴 Pending | High |
| Process light curves to .npz format | 🔴 Pending | High |
| Implement secondary eclipse view | 🔴 Pending | Medium |
| Implement odd/even transit views | 🔴 Pending | Medium |
| Extract centroid time series | 🔴 Pending | Low |

### Phase 2: Architecture Upgrade (Week 2-3)

| Task | Status | Priority |
|------|--------|----------|
| Implement SE blocks | 🔴 Pending | High |
| Add multi-scale convolutions | 🔴 Pending | High |
| Implement attention fusion | 🔴 Pending | Medium |
| Add residual connections | 🔴 Pending | Medium |
| Implement physics-preserving augmentation | 🔴 Pending | High |

### Phase 3: Training Pipeline (Week 3-4)

| Task | Status | Priority |
|------|--------|----------|
| Implement focal loss with auto-tuning | 🟡 Partial | High |
| Add cosine annealing scheduler | 🔴 Pending | Medium |
| Implement synthetic pre-training | 🔴 Pending | Medium |
| Set up proper OOF collection | 🟢 Done | - |
| Add XGBoost hyperparameter tuning | 🟢 Done | - |

### Phase 4: Evaluation & Explainability (Week 4-5)

| Task | Status | Priority |
|------|--------|----------|
| Implement recall@precision metric | 🔴 Pending | Critical |
| Add reliability diagram | 🔴 Pending | High |
| Implement integrated gradients | 🔴 Pending | High |
| Branch contribution analysis | 🔴 Pending | Medium |
| Bootstrap confidence intervals | 🔴 Pending | Medium |

### Phase 5: Documentation & Reproducibility (Week 5-6)

| Task | Status | Priority |
|------|--------|----------|
| Full experiment logging (MLflow/W&B) | 🔴 Pending | High |
| Hyperparameter configuration files | 🔴 Pending | High |
| Results summary & figures | 🔴 Pending | Critical |
| IEEE paper draft | 🔴 Pending | Critical |
| Code cleanup & comments | 🔴 Pending | High |

---

## 7. Expected Results

### 7.1 Target Performance (Based on Literature)

| Metric | Current | Target | ExoMiner Benchmark |
|--------|---------|--------|-------------------|
| Precision | ~0.50 | ≥0.95 | 0.99 |
| Recall | ~0.30 | ≥0.85 | 0.936 |
| F1 Score | ~0.02 | ≥0.80 | ~0.96 |
| PR-AUC | ~0.60 | ≥0.92 | 0.98 |
| Recall@P=0.99 | N/A | ≥0.70 | 0.936 |

### 7.2 Ablation Study Plan

| Experiment | Purpose |
|------------|---------|
| Baseline CNN (no augmentation) | Lower bound |
| + Focal Loss | Measure imbalance handling |
| + Oversampling | Compare to focal loss |
| + SE Blocks | Measure channel attention impact |
| + Multi-scale Conv | Measure feature extraction improvement |
| + Attention Fusion | Measure cross-view learning |
| + XGBoost Ensemble | Measure hybrid boost |
| + Stacking Meta-learner | Final performance |

### 7.3 Novel Contributions for IEEE Paper

1. **Multi-view attention fusion** for transit signal classification
2. **Physics-preserving augmentation** framework for light curves
3. **Comprehensive benchmark** across Kepler, K2, and TESS missions
4. **Explainability framework** for scientific validation
5. **Production-ready pipeline** with NASA data standards

---

## Appendix A: File Structure

```
Exoplanet_AI/
├── data/
│   ├── raw/                    # Original catalog CSVs
│   ├── processed/              # Processed .npz files
│   └── splits/                 # Train/val/test indices
├── models/
│   ├── checkpoints/            # Model weights
│   └── configs/                # Hyperparameter configs
├── scripts/
│   ├── download_data.py        # Data acquisition
│   ├── process_data.py         # Preprocessing
│   ├── train_exominer.py       # Main training script
│   └── evaluate.py             # Evaluation suite
├── notebooks/
│   ├── EDA.ipynb              # Exploratory analysis
│   ├── Results.ipynb          # Final results
│   └── Explainability.ipynb   # Attribution analysis
├── docs/
│   └── NASA_OPTIMAL_PLAN.md   # This document
├── tests/
│   └── test_*.py              # Unit tests
└── paper/
    ├── figures/               # Publication figures
    └── draft.tex              # IEEE paper draft
```

---

## Appendix B: References

1. **ExoMiner**: Valizadegan et al. (2021). "ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier that Validates 301 New Exoplanets." ApJ, 926, 120.

2. **AstroNet**: Shallue & Vanderburg (2018). "Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90." AJ, 155, 94.

3. **Astronet-Triage**: Yu et al. (2019). "Identifying Exoplanets with Deep Learning III: Automated Triage and Vetting of TESS Candidates." AJ, 158, 25.

4. **Focal Loss**: Lin et al. (2017). "Focal Loss for Dense Object Detection." ICCV.

5. **SE-Net**: Hu et al. (2018). "Squeeze-and-Excitation Networks." CVPR.

---

*Document Version: 1.0*  
*Last Updated: January 12, 2026*

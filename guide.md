# Feature Engineering Guide

**Project:** Net Zero Tracker — Scope 3 Coverage Classification  
**Last Updated:** 21 December 2025

---

## 1. Project Overview

This project develops a multiclass classification model to predict `Scope_3_coverage` (Yes, No, Partial, Not Specified) for companies based on their climate commitments and organizational characteristics. The dataset originates from the Net Zero Tracker initiative.

### Dataset Summary

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 1,129 | Model training and encoder fitting |
| Validation | 377 | Hyperparameter tuning |
| Test | 377 | Final evaluation (held out) |

### Target Variable Distribution

| Class | Proportion |
|-------|------------|
| Yes | ~40% |
| Not Specified | ~22% |
| Partial | ~19% |
| No | ~18% |

---

## 2. Pipeline Architecture

The feature engineering pipeline consists of two stages. **All learnable transformations (imputation statistics, scalers, encoders) are fitted exclusively on the training set** to prevent data leakage.

```
Raw Data (data/raw/)
       │
       ▼
┌───────────────────────────────────────────────────────┐
│  Stage 1: preprocessing_pipeline.py                  │
│  1. Stratified train/val/test split                  │
│  2. Calculate imputation stats (FIT on train only)   │
│  3. Apply imputation to all splits                   │
│  4. Log transformations                              │
│  5. Standard scaling (FIT on train only)             │
│  6. Missingness indicators                           │
└───────────────────────────────────────────────────────┘
       │
       ▼
Interim Data (data/interim/)
       │
       ▼
┌───────────────────────────────────────────────────────┐
│  Stage 2: pipeline.py                                │
│  1. Ordinal encoding (FIT on train only)             │
│  2. One-hot encoding (FIT on train only)             │
│  3. Country frequency grouping (FIT on train only)   │
│  4. Classifier integration                           │
└───────────────────────────────────────────────────────┘
       │
       ▼
Encoded Features (70 columns) → Model
```

### Rationale for Two-Stage Design

1. **Stage 1** handles complex imputation logic (group-based medians/modes) that requires custom implementation — statistics computed from training data, then applied to all splits
2. **Stage 2** uses sklearn's `ColumnTransformer` for serializable, production-ready encoding — `fit()` called on training data only, `transform()` applied to all splits
3. The intermediate CSV files (`data/interim/`) enable inspection and debugging between stages

---

## 3. Execution Order

### Step 1: Run Preprocessing Pipeline

```bash
python src/preprocessing_pipeline.py
```

**Inputs:**
- `data/raw/net_zero_tracker.csv`

**Outputs:**
- `data/interim/train.csv` (1,129 rows × 22 columns)
- `data/interim/val.csv` (377 rows × 22 columns)
- `data/interim/test.csv` (377 rows × 22 columns)
- `logs/preprocess.log`

**Key Operations:**
- Stratified train/val/test split (60/20/20)
- Industry-group imputation: statistics (medians/modes) computed on **train only**, then applied to all splits
- Log transformation for `revenue`, `Employees`
- Standard scaling: `StandardScaler` **fit on train only**, then transforms applied to val/test
- Creation of `Interim_target_year_missing` indicator

### Step 2: Run Encoding Pipeline

```bash
python src/pipeline.py
```

**Inputs:**
- `data/interim/train.csv`, `val.csv`, `test.csv`

**Outputs:**
- Encoded numpy arrays (70 features)
- `logs/feature_engineering.log`

**Key Operations:**
- Fit encoders on training data only
- Ordinal encoding for 11 ordered categorical features
- One-hot encoding for 4 nominal features + Country
- Country grouping: top 10 by frequency + "Other"

### Step 3: Model Training (Next Milestone)

```python
from src.pipeline import create_pipeline, load_data, encode_target

X_train, X_val, X_test, y_train, y_val, y_test = load_data()
y_train_enc, y_val_enc, y_test_enc, le = encode_target(y_train, y_val, y_test)

from sklearn.ensemble import RandomForestClassifier
pipeline = create_pipeline(classifier=RandomForestClassifier())
pipeline.fit(X_train, y_train_enc)
```

---

## 4. Feature Summary

### Final Feature Set (21 Features → 70 Encoded)

| Category | Count | Features |
|----------|-------|----------|
| Numeric (passthrough) | 5 | `End_target_year`, `Interim_target_year`, `log_revenue`, `log_employees`, `Interim_target_year_missing` |
| Ordinal (integer-encoded) | 11 | `Published_plan`, `Private_company`, `Race_to_zero_member`, `Separate_removal_target`, `Historical_emissions`, `Accountability_delivery`, `Carbon_credits`, `GHGs_covered`, `Reporting_mechanism`, `Status_of_end_target`, `Planning_removals` |
| Nominal (one-hot) | 4 | `Geographic_region`, `Industry`, `End_target`, `Interim_target` |
| High-cardinality (grouped + one-hot) | 1 | `Country` → 11 columns (top 10 + Other) |

### Excluded Features (Data Leakage)

| Feature | Reason for Exclusion |
|---------|---------------------|
| `Scope_1_coverage` | Determined simultaneously with target variable |
| `Scope_2_coverage` | Determined simultaneously with target variable |

---

## 5. Documentation Map

### Configuration Files

| File | Purpose | When to Consult |
|------|---------|-----------------|
| `preprocessing_config.yml` | Imputation strategies, missingness types (MCAR/MAR/MNAR), and rationale per feature | Before modifying imputation logic |
| `encoding_strategy.md` | Encoding decisions for each feature type | Before changing encoding approach |
| `final_features.txt` | Authoritative list of 21 input features | To verify feature set |

### Planning Documents

| File | Purpose | When to Consult |
|------|---------|-----------------|
| `feature_engineering_plan.md` | EDA-driven engineering decisions, priority rankings | To understand why specific features were engineered |
| `eda_reflection.md` | Key EDA findings and their implications | To review data characteristics |

### Reflection Documents

| File | Purpose | When to Consult |
|------|---------|-----------------|
| `fe_reflection.md` | Lessons learned, debugging challenges, design tradeoffs | For retrospective or knowledge transfer |
| `PIPELINE_README.md` | Technical documentation for two-stage pipeline | For onboarding new contributors |

### Audit Logs

| File | Purpose |
|------|---------|
| `logs/preprocess.log` | Timestamped record of preprocessing operations |
| `logs/feature_engineering.log` | Timestamped record of encoding operations |
| `logs/checksum.log` | SHA256 checksums for data integrity verification |

---

## 6. Data Leakage Prevention

The pipeline implements the following safeguards:

| Stage | Safeguard | Implementation |
|-------|-----------|----------------|
| Data Split | Stratified split before any processing | `train_test_split(..., stratify=y)` |
| Imputation | Group statistics computed on training set only | `preprocessing_pipeline.py` calculates medians/modes from train, applies to all splits |
| Encoding | Encoders fitted on training set only | `preprocessor.fit(X_train)` then `transform()` for all splits |
| Country Grouping | Top 10 countries determined from training set | `CountryGrouper.fit()` learns from train only |
| Unknown Handling | Unseen categories handled gracefully | `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`, `OneHotEncoder(handle_unknown='ignore')` |

---

## 7. Reproducibility

### Checksum Verification

Each interim CSV file has a corresponding `.sha256` file. To verify data integrity:

```bash
python src/generate_checksums.py --verify
```

### Environment

All dependencies are specified in `requirements.txt`. Key packages:
- pandas
- numpy
- scikit-learn
- joblib

---

## 8. Next Steps (Modeling Milestone)

1. Load preprocessed data using `load_data()` from `src/pipeline.py`
2. Create pipeline with classifier using `create_pipeline(classifier=...)`
3. Train on `X_train`, tune hyperparameters on `X_val`
4. Evaluate final model on `X_test`
5. Save fitted pipeline using `save_pipeline()` for deployment

---

## 9. Quick Reference: File Locations

```
calibration_classification/
├── src/
│   ├── preprocessing_pipeline.py   # Stage 1: Imputation & scaling
│   ├── pipeline.py                 # Stage 2: Encoding & model
│   ├── generate_checksums.py       # Data integrity verification
│   └── reproducibility.py          # Reproducibility utilities
├── data/
│   ├── raw/                        # Original dataset
│   ├── interim/                    # Preprocessed CSVs (train/val/test)
├── logs/
│   ├── preprocess.log              # Preprocessing audit trail
│   ├── feature_engineering.log     # Encoding audit trail
│   └── checksum.log                # Data integrity log
├── preprocessing_config.yml        # Imputation configuration
├── encoding_strategy.md            # Encoding decisions
├── feature_engineering_plan.md     # Engineering rationale
├── final_features.txt              # Feature list
└── guide.md                        # This document
```

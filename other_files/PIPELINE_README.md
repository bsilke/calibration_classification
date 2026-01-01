# Pipeline Documentation

## Overview

This project uses a **two-stage pipeline** for feature engineering:

| Stage | File | Purpose |
|-------|------|---------|
| **Stage 1** | `src/preprocessing_pipeline.py` | Data cleaning, imputation, log transforms, scaling → saves CSVs |
| **Stage 2** | `src/pipeline.py` | Categorical encoding, model integration → serializable sklearn Pipeline |

## Why Two Stages?

**Stage 1 (preprocessing_pipeline.py):**
- Complex group-based imputation (Industry median/mode) that doesn't fit sklearn's SimpleImputer
- Creates train/val/test splits BEFORE imputation to prevent leakage
- Adds missingness indicators
- Saves intermediate CSVs for EDA and debugging

**Stage 2 (pipeline.py):**
- sklearn-compatible Pipeline for encoding + model
- Serializable with `joblib.dump()` for deployment
- Fits on training data, transforms val/test consistently

## Pipeline Components

### Numeric Features (5)
| Feature | Handling |
|---------|----------|
| `End_target_year` | Passthrough (already scaled) |
| `Interim_target_year` | Passthrough (already scaled) |
| `log_revenue` | Passthrough (already scaled) |
| `log_employees` | Passthrough (already scaled) |
| `Interim_target_year_missing` | Passthrough (binary 0/1) |

### Ordinal Features (11) → `OrdinalEncoder`
Features with meaningful order:
- Binary: `Published_plan`, `Private_company`, `Race_to_zero_member`, `Separate_removal_target`, `Historical_emissions`
- Ternary: `Accountability_delivery`, `Carbon_credits`
- Multi-level: `GHGs_covered`, `Reporting_mechanism`, `Status_of_end_target`, `Planning_removals`

### Nominal Features (4) → `OneHotEncoder`
- `Geographic_region` (8 categories)
- `Industry` (13 categories)
- `End_target` (7 categories)
- `Interim_target` (7 categories)

### High Cardinality Feature (1) → Frequency Grouping + OneHot
- `Country` (61 unique → top 10 + "Other")
- Rationale: Prevents sparse 61-column one-hot; retains most common countries

## Usage

```python
from src.pipeline import create_pipeline, load_data, encode_target

# Load preprocessed data
X_train, X_val, X_test, y_train, y_val, y_test = load_data()

# Encode target
y_train_enc, y_val_enc, y_test_enc, le = encode_target(y_train, y_val, y_test)

# Create pipeline with model
from sklearn.ensemble import RandomForestClassifier
pipeline = create_pipeline(classifier=RandomForestClassifier(n_estimators=100, random_state=42))

# Fit on training data only
pipeline.fit(X_train, y_train_enc)

# Predict
predictions = pipeline.predict(X_test)

# Save for deployment
from src.pipeline import save_pipeline
save_pipeline(pipeline, le, 'models/pipeline.pkl')
```

## Testing

Run the pipeline module directly to verify:

```bash
python src/pipeline.py
```

Expected output:
- Data loaded with correct shapes
- Top 10 countries identified
- Verification that train/val/test have same number of encoded columns

## Dropped Features

| Feature | Reason |
|---------|--------|
| `Scope_1_coverage` | Target leakage |
| `Scope_2_coverage` | Target leakage |
| `GHG_emissions` | >50% missing (MNAR) |

## Files

| File | Purpose |
|------|---------|
| `src/preprocessing_pipeline.py` | Stage 1: Data prep → CSVs |
| `src/pipeline.py` | Stage 2: Encoding + Model |
| `preprocessing_config.yml` | Imputation strategies & rationale |
| `encoding_strategy.md` | Encoding decisions |
| `final_features.txt` | Feature list for reproducibility |
| `models/pipeline.pkl` | Saved fitted pipeline (after training) |
| `models/scaler.joblib` | Saved StandardScaler (from Stage 1) |

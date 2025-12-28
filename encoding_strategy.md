# Categorical Feature Encoding Strategy

## Dataset Overview

- **Total Categorical Features:** 16 (in `data/interim/train.csv`, excluding target and leakage columns)
- **Target Variable:** `Scope_3_coverage` (4-class: Yes, Not Specified, Partial, No)
- **Numeric Features:** 4 (`log_revenue`, `log_employees`, `End_target_year`, `Interim_target_year`)
- **Dropped (Leakage):** 2 (`Scope_1_coverage`, `Scope_2_coverage`)

---

## Encoding Summary

| Category | Count | Encoding Method | Rationale |
|----------|-------|-----------------|-----------|
| **ORDINAL** | 11 | `OrdinalEncoder` | Features with meaningful natural ordering |
| **NOMINAL** | 5 | `OneHotEncoder` | Features with no natural ordering |
| **TARGET** | 1 | `LabelEncoder` | Multi-class target variable |
| **DROPPED (Leakage)** | 2 | N/A | `Scope_1_coverage`, `Scope_2_coverage` |

**Note:** Identifier columns (`ID_Code`, `Name`) were already dropped during preprocessing and do not exist in `data/interim/train.csv`.

---

## ⚠️ DROPPED FEATURES - Target Leakage

The following features were **removed** from the dataset due to **target leakage**:

| Feature | Reason for Removal |
|---------|--------------------|
| `Scope_1_coverage` | Decided simultaneously with Scope_3_coverage during corporate emissions reporting |
| `Scope_2_coverage` | Decided simultaneously with Scope_3_coverage during corporate emissions reporting |

**Why this matters:**
- Companies decide on Scope 1, 2, and 3 coverage *together* as part of their emissions reporting strategy
- A company reporting Scope 1 & 2 has already committed to comprehensive reporting → Scope 3 is highly likely
- Including these features would create **circular reasoning** and **artificially inflate model performance**
- Feature selection analysis showed suspiciously high MI scores (0.276, 0.237) confirming leakage

**Decision Date:** 2025-12-20

---

## Ordinal Features (11) → `OrdinalEncoder`

Ordinal encoding maps categories to integers while preserving natural order.

### Binary Yes/No Features (6)

| Feature | Order | Reasoning |
|---------|-------|-----------|
| `Published_plan` | No → Yes | Having a plan is "higher" than not having one |
| `Private_company` | No → Yes | Binary indicator (public vs private) |
| `Race_to_zero_member` | No → Yes | Binary membership status |
| `Separate_removal_target` | No → Yes | Having separate target is more specific |
| `Historical_emissions` | No → Yes | Includes historical emissions or not |

### Ternary Features (2)

| Feature | Order | Reasoning |
|---------|-------|-----------|
| `Accountability_delivery` | Not Specified → No → Yes | Ordered by accountability level |
| `Carbon_credits` | No → Not Specified → Yes | Ordered by commitment to carbon credits |

### ~~Coverage Features~~ — REMOVED (Target Leakage)

| Feature | Status |
|---------|--------|
| `Scope_1_coverage` | ❌ **DROPPED** - Target leakage |
| `Scope_2_coverage` | ❌ **DROPPED** - Target leakage |

See "DROPPED FEATURES - Target Leakage" section above for details.

### Multi-level Ordinal Features (4)

| Feature | Order | Reasoning |
|---------|-------|-----------|
| `GHGs_covered` | Not Specified → Carbon dioxide only → Carbon dioxide and other GHGs | Comprehensiveness progression |
| `Reporting_mechanism` | No reporting mechanism → Less than annual reporting → Annual reporting | Frequency progression |
| `Status_of_end_target` | Proposed/in discussion → Declaration/pledge → In corporate strategy → Achieved (self-declared) → Achieved (externally validated) | Commitment progression |
| `Planning_removals` | No → Not Specified → Yes (unspecified) → Yes (nature-based) → Yes (CCS-based) → Yes (both) | Specificity progression |

---

## Nominal Features (5) → `OneHotEncoder`

One-hot encoding creates binary columns for each category. Used when no natural ordering exists.

| Feature | Unique Values | Notes |
|---------|---------------|-------|
| `Country` | 61 | **High cardinality** → Use **Target Encoding** with smoothing (see Feature Engineering section) |
| `Geographic_region` | 8 | Moderate cardinality - suitable for one-hot |
| `Industry` | 13 | Moderate cardinality - suitable for one-hot |
| `End_target` | 16 | Target types have no inherent ranking |
| `Interim_target` | 7 | Low cardinality - suitable for one-hot |

### Reasoning for Nominal Classification

- **Country:** Geographic identifiers cannot be ranked meaningfully (USA is not "greater than" JPN)
- **Geographic_region:** Regional groupings have no inherent hierarchy
- **Industry:** Business sectors are qualitative categories without ranking
- **End_target/Interim_target:** Different target types represent strategies, not progressions

---

## Special Considerations for Random Forest

### 1. High Cardinality Features

`Country` (61 unique values) creates sparse features with one-hot encoding.

**DECISION:** Use **Target Encoding with Smoothing** (see `feature_engineering_plan.md`)

```python
from category_encoders import TargetEncoder

country_encoder = TargetEncoder(
    cols=['Country'],
    smoothing=10,  # Blend rare categories toward global mean
    min_samples_leaf=5
)
# CRITICAL: Fit only on training data
country_encoder.fit(X_train, y_train_encoded)
```

**Rationale:**
- One-hot would create 61 sparse columns
- Target encoding captures country-target relationships
- Smoothing prevents overfitting on rare countries (e.g., countries with <10 samples)
- `Geographic_region` serves as fallback grouping if needed

### 2. Ordinal Encoding Advantages

Tree-based models can learn thresholds on ordinal-encoded features:
- Reduces dimensionality compared to one-hot encoding
- Preserves ordering information for splits
- More efficient for features with many ordered categories

### 3. Missing Indicator Features

Binary `_was_missing` columns were created during imputation to capture missingness patterns. These complement imputed values and may contain predictive signal.

### 4. Interaction Features

**DECISION:** ~~Create `Industry × End_target` interaction feature~~ **DEFERRED**

```python
# NOT IMPLEMENTED - Random Forest learns interactions automatically
# X_train['Industry_EndTarget'] = X_train['Industry'] + '_' + X_train['End_target']
```

**Original Rationale:**
- `Industry` (Cramér's V = 0.22) and `End_target` (Cramér's V = 0.31) both highly predictive
- Combination may capture sector-specific patterns (e.g., "Net Zero" may mean different Scope 3 coverage in Energy vs Financials)
- Apply frequency threshold: keep interactions with ≥10 samples, group others as "Other"

**Why Deferred:**
- Random Forest (planned model) learns feature interactions automatically through tree splits
- Adding explicit interaction would create high cardinality (13 × 7 = 91 combinations)
- Can revisit if linear models are used or if RF feature importance suggests interaction value

### 5. Numerical Feature Binning

**DECISION:** Bin `End_target_year` into categorical groups (optional experiment)

```python
def bin_target_year(year):
    if year <= 2030: return 'Near-term (≤2030)'
    elif year <= 2040: return 'Medium-term (2031-2040)'
    elif year <= 2050: return 'Paris-aligned (2041-2050)'
    else: return 'Long-term (>2050)'
```

**Rationale:**
- EDA showed weak linear correlation (-0.03) between year and target
- Categorical treatment may capture 2050 Paris Agreement threshold significance
- Lower priority; test incrementally

---

## Data Leakage Prevention

**CRITICAL:** All encoders must be **fit only on training data** to prevent data leakage.

### Workflow

1. **Fit encoders** on `data/interim/train.csv` only
2. **Transform** training data using fitted encoders
3. **Transform** validation data using the SAME fitted encoders (no re-fitting)
4. **Transform** test data using the SAME fitted encoders (no re-fitting)

### Why This Matters

- Fitting on val/test would leak information about categories not seen during training
- One-hot encoding must know all possible categories from training data only
- Target encoding (if used) must calculate means from training labels only
- Ensures model evaluation reflects true generalization performance

### Handling Unknown Categories

Categories in val/test that don't exist in training:
- **OrdinalEncoder:** Set `handle_unknown='use_encoded_value'` with `unknown_value=-1`
- **OneHotEncoder:** Set `handle_unknown='ignore'` (creates zero vector)

### Saving Fitted Encoders

```python
import joblib

# After fitting on training data
joblib.dump(preprocessor, 'models/fitted_encoder.joblib')

# When applying to new data
preprocessor = joblib.load('models/fitted_encoder.joblib')
X_new_encoded = preprocessor.transform(X_new)
```

---

## Implementation Notes

```python
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Define column groups (all columns exist in data/interim/train.csv)
ORDINAL_COLS = ['Published_plan', 'Private_company', 'Race_to_zero_member', 
                'Separate_removal_target', 'Historical_emissions', 'Accountability_delivery',
                'Scope_1_coverage', 'Scope_2_coverage', 'Carbon_credits', 'GHGs_covered',
                'Reporting_mechanism', 'Status_of_end_target', 'Planning_removals']

NOMINAL_COLS = ['Country', 'Geographic_region', 'Industry', 'End_target', 'Interim_target']

# Note: ID_Code and Name were already dropped during preprocessing

# Example preprocessor - FIT ONLY ON TRAINING DATA
preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value', 
                                   unknown_value=-1), ORDINAL_COLS),
        ('nominal', OneHotEncoder(handle_unknown='ignore', sparse_output=False), NOMINAL_COLS),
    ],
    remainder='passthrough'  # Keeps numeric columns: log_revenue, log_employees, End_target_year, Interim_target_year
)

# Correct usage:
# preprocessor.fit(X_train)           # Fit on training only
# X_train_enc = preprocessor.transform(X_train)
# X_val_enc = preprocessor.transform(X_val)    # Transform only
# X_test_enc = preprocessor.transform(X_test)  # Transform only
```

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-12-20 | Classify 13 features as ordinal | Domain knowledge indicates natural ordering |
| 2025-12-20 | Classify 5 features as nominal | No inherent ranking between categories |
| 2025-12-20 | Drop ID_Code, Name | Unique identifiers with no predictive value |
| 2025-12-20 | Flag Country as high-cardinality | 61 unique values may cause sparse encoding issues |
| 2025-12-20 | Fit encoders on training data only | Prevent data leakage into validation/test sets |
| 2025-12-20 | Use frequency grouping for Country | Top 10 + "Other" reduces 61 to 11 categories |
| 2025-12-20 | ~~Create Industry × End_target interaction~~ DEFERRED | Random Forest learns interactions automatically |
| 2025-12-20 | Optional: Bin End_target_year | Categorical treatment for Paris-aligned thresholds |

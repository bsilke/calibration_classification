# Preprocessing Workflow

1. Preprocessing Workflow
2. Load & clean data (drop columns, handle missing values, create indicators) ✅ done
3. Train-test split ← next step
4. Fit encoders on training data only
5. Transform both train and test sets
6. Train model on encoded training data
7. Evaluate on encoded test data

---

# Data Quality Reflection: Most Significant Risk

## Top Risk: Missing Values (~40%+ in Critical Columns)

The most significant data-quality risk in this classification project is **the large number of missing values**. The raw dataset contains over 40% missing values in many columns, including business-critical features like `Company_annual_revenue`, `Employees`, and `GHG_emissions`. The `GHG_emissions` column was so incomplete (>50% missing) that it had to be dropped entirely.

This risk matters because missing data forces difficult trade-offs during preprocessing. Dropping rows with missing values would reduce the dataset from 2,086 companies to a much smaller subset, sacrificing statistical power and potentially introducing selection bias if missingness is not random. Conversely, imputation introduces assumptions about the data distribution that may not hold—for example, imputing revenue by industry median assumes companies within an industry have similar financials, which may not be true for outliers or niche players.

The imputation strategy chosen (group-based median/mode by Industry) was fitted on training data only to prevent data leakage, but the assumptions embedded in this approach propagate through to model predictions.

**Mitigation Strategy:**
1. **Document imputation statistics** — The pipeline logs all imputation values for reproducibility and auditing
2. **Monitor imputation rates** — Track what percentage of predictions rely on imputed values
3. **Group-based imputation** — Using Industry-specific medians reduces distortion compared to global imputation
4. **Flag high-imputation records** — In production, records with many imputed features could be flagged for manual review

---
# Top 3 Data Quality Issues

Based on the Data Quality Scorecard results, the following are the most significant data quality issues:

---

## 1. 🔴 Missing Values (~40%+ in many columns)

**Impact on Model Performance:**
- Reduces available training data significantly when rows are dropped
- May introduce bias if missingness is not random (MNAR/MAR)
- Imputation can distort feature distributions and relationships

**Difficulty to Fix:** ⚠️ Medium-High
- Simple imputation (median/mode) is straightforward but may not be appropriate
- Multiple imputation or domain-specific strategies require more effort
- Some columns have >50% missing — may need to be dropped entirely

**System/Team Impact:**
- Affects feature engineering decisions downstream
- Requires documentation of imputation strategy for reproducibility
- Stakeholders may question predictions for records with imputed values

---

## 2. 🔴 Outliers (>10% in numeric columns)

**Impact on Model Performance:**
- Skews mean-based statistics and standardization
- Tree-based models may create splits on extreme values
- Affects distance-based algorithms (KNN, clustering)
- May represent legitimate extreme cases (large corporations)

**Difficulty to Fix:** ⚠️ Medium
- Log transformation works well for revenue/employee counts
- Winsorization preserves data while limiting extremes
- Requires domain knowledge to distinguish errors from valid extremes

**System/Team Impact:**
- Transformation decisions affect model interpretability
- Need to apply same transformation to new predictions
- Business users may find log-transformed values harder to interpret

---

## 3. 🟡 Data Type Inconsistencies (e.g., Company_annual_revenue as object)

**Impact on Model Performance:**
- Prevents numeric operations and statistical analysis
- May cause silent failures in sklearn pipelines
- Values like "$1,000,000" need parsing before use

**Difficulty to Fix:** ✅ Low-Medium
- String cleaning with regex is straightforward
- Need to handle edge cases (currency symbols, different formats)
- Requires validation after conversion

**System/Team Impact:**
- Upstream data collection process may need improvement
- ETL pipelines should standardize formats at ingestion
- Documentation needed for expected formats

---

## Summary Table

| Issue | Severity | Fix Difficulty | Downstream Impact |
|-------|----------|----------------|-------------------|
| Missing Values | 🔴 High | Medium-High | Feature selection, model training |
| Outliers | 🔴 High | Medium | Scaling, model interpretation |
| Data Type Issues | 🟡 Medium | Low-Medium | Pipeline reliability |

**Recommended Priority:** Address data type issues first (quick win), then handle missing values strategy, finally apply outlier treatment after train-test split to avoid data leakage.

---

# Random Forest Model: Data Quality Sensitivities

For a **Random Forest model**, the data quality issues have different impacts:

| Issue | Impact on Random Forest |
|-------|------------------------|
| **Missing Values** | 🔴 **MOST CRITICAL** - scikit-learn's `RandomForestClassifier` will throw an error with NaN values. Must be handled before training. |
| **Data Type Issues** | 🟡 Medium - Columns must be numeric or properly encoded. String values cause failures. |
| **Outliers** | 🟢 **LOW IMPACT** - Random Forest is inherently robust to outliers because splits are based on ranking/thresholds, not distances. |

## Why Missing Values Matter Most for Random Forest:

1. **Hard Constraint**: Unlike some implementations (XGBoost, LightGBM), sklearn RF cannot handle NaN values at all
2. **Your dataset has ~40%+ missing in many columns** - this forces either:
   - Dropping rows → significant data loss
   - Imputation → introduces assumptions into your features
3. **Imputation strategy affects model** - median vs. mode vs. model-based imputation will give different results

## Important to consider for Random Forest:

- **Outliers are NOT a concern** - the algorithm's tree-based splits are unaffected by extreme values
- **No scaling required** - unlike logistic regression or SVM, RF doesn't need standardization
- **Handles mixed feature types** well (after encoding categoricals)

**Remark**: Focus preprocessing effort on handling missing values properly, rather than outlier treatment. The outliers in the revenue/employee columns won't hurt Random Forest performance.

---

# Data Leakage Analysis

## Overview

Data leakage occurs when information from outside the training dataset is used to create the model, leading to overly optimistic performance estimates that don't generalize to new data.

## Flagged Features (8 total)

### 🔴 HIGH RISK - Scope-Related Features

| Feature | Correlation with Target | Risk Level | Issue |
|---------|------------------------|------------|-------|
| `Scope_1_coverage` | ~0.4-0.6 | **HIGH** | Companies reporting Scope 1 emissions are more likely to report Scope 3 |
| `Scope_2_coverage` | ~0.4-0.6 | **HIGH** | Same pattern - correlated reporting behavior |

**Why this is leakage:**
- Scope 1, 2, and 3 coverage are often reported together as part of GHG reporting
- Using Scope 1/2 to predict Scope 3 is predicting "does company report emissions?" rather than learning meaningful patterns
- In production, if you already know Scope 1/2 status, you likely already know Scope 3 status

**Decision needed:**
- **Option A: Drop Scope 1 & 2** - Predict Scope_3 without knowing other scope reporting (harder, more useful)
- **Option B: Keep them** - Predict Scope_3 given current reporting behavior (easier, less actionable)

### 🟢 LOW RISK - False Positives (Feature Names Contain "target")

| Feature | Risk Level | Why It's Safe |
|---------|------------|---------------|
| `End_target` | LOW | Refers to climate target type (Net Zero, Carbon Neutral, etc.) |
| `Interim_target` | LOW | Refers to interim climate target type |
| `End_target_year` | LOW | Year of climate target commitment |
| `Interim_target_year` | LOW | Year of interim target commitment |
| `Status_of_end_target` | LOW | Progress status of climate target |
| `Separate_removal_target` | LOW | Whether company has separate carbon removal target |

**Why these are NOT leakage:**
- "Target" in these features refers to **climate/emissions targets**, not the ML target variable
- These are legitimate predictive features about company climate commitments
- No direct relationship with `Scope_3_coverage` beyond valid correlations

## Recommendations

### If Goal is: "Predict which companies will report Scope 3"
→ **Drop `Scope_1_coverage` and `Scope_2_coverage`**
- More challenging prediction task
- Model learns from company characteristics (industry, size, country, targets)
- Useful for identifying potential reporters

### If Goal is: "Predict Scope 3 status given all available reporting info"
→ **Keep all features**
- Easier prediction task
- Scope 1/2 are strong predictors
- Less useful for business decisions (you'd already know the answer)

## Leakage Check Function

```python
def check_for_leakage(df, target_col, feature_cols):
    """Check for potential data leakage"""
    leakage_report = {}
    
    for col in feature_cols:
        # Check for perfect correlation with target
        if df[col].dtype in ['int64', 'float64']:
            correlation = df[col].corr(df[target_col].astype('category').cat.codes)
            if abs(correlation) > 0.95:
                leakage_report[col] = f"High correlation with target: {correlation:.3f}"
        
        # Check for target-derived features (common naming patterns)
        if any(word in col.lower() for word in ['target', 'label', 'outcome', 'result']):
            leakage_report[col] = "Potentially target-derived feature name"
        
        # Check for future-looking features in time series
        if any(word in col.lower() for word in ['future', 'next', 'after', 'lag_-']):
            leakage_report[col] = "Potentially future-looking feature"
    
    return leakage_report
```

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-12-20 | **Drop Scope_1_coverage and Scope_2_coverage** | Companies that report Scope 3 ALWAYS also report Scope 1 and 2. This logical dependency creates data leakage - the model could learn this rule instead of meaningful patterns. |
| 2025-12-20 | **Fix imputation leakage - created data_preprocessing_v2.ipynb** | Original preprocessing did imputation before train/test split, causing test data to influence imputation statistics. New version splits FIRST, then calculates imputation stats from training data only. |

---

# Preprocessing Pipeline Fix (Data Leakage)

## Original Issue

The original `data_preprocessing.ipynb` had a data leakage issue:

```
❌ WRONG ORDER (v1):
Raw Data → Impute (using ALL data) → Split → train/val/test
```

This means imputation statistics (medians, modes) were calculated using test data, which leaks information.

## Corrected Pipeline

Created `data_preprocessing_v2.ipynb` with the correct order:

```
✅ CORRECT ORDER (v2):
Raw Data → Split → Impute (using TRAINING data only) → train/val/test
```

### Key Changes:
1. Split happens on raw data (with only minimal cleaning)
2. Imputation statistics (group medians/modes by Industry) calculated from TRAINING set only
3. Same statistics applied to val/test sets
4. No information from test set influences any preprocessing step

### Impact Assessment:
- **Severity**: Low-Medium (large industry groups mean train-only stats ≈ full stats)
- **Fix effort**: Created new preprocessing notebook
- **Recommendation**: Use `data_preprocessing_v2.ipynb` for rigorous methodology

---

# SMOTE Analysis for Class Imbalance

## Overview

Analyzed whether SMOTE (Synthetic Minority Over-sampling Technique) would improve model performance for the Scope_3_coverage classification task.

## Class Distribution (Training Set)

| Class | Count | Percentage |
|-------|-------|------------|
| Yes | 456 | 40.3% |
| Not Specified | 247 | 21.8% |
| Partial | 220 | 19.5% |
| No | 208 | 18.4% |

**Imbalance Ratio:** 2.19:1 (majority/minority)

## SMOTE Threshold Check

- **Minority class (No):** 18.4%
- **SMOTE threshold:** 10%
- **Result:** Minority class is **above** the 10% threshold → SMOTE not strictly required

## Methods Tested

1. **Original (no resampling):** 1,131 samples
2. **SMOTE:** 1,824 samples (all classes balanced to 456)
3. **SMOTETomek:** 1,768 samples (SMOTE + Tomek link removal for cleaner boundaries)

## Results

| Method | Training Samples | Balanced Accuracy |
|--------|------------------|-------------------|
| **Original** | 1,131 | **0.466** |
| SMOTE | 1,824 | 0.448 |
| SMOTETomek | 1,768 | 0.442 |

## Key Findings

1. **SMOTE did NOT improve performance** - original data performs best
2. The "Partial" class has consistently low recall (8-11%) across all methods - this is an inherently difficult class to predict
3. SMOTE's synthetic samples appear to add noise rather than useful signal in this case

## Decision

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-12-20 | **Do NOT use SMOTE** | Minority class (18.4%) is above 10% threshold. Testing showed original data achieves higher balanced accuracy (0.466) than SMOTE (0.448) or SMOTETomek (0.442). |

## Recommendation

Use the **original (unbalanced) training data** for the final model. The class imbalance is relatively mild (2.19:1) and SMOTE does not provide benefit for this dataset.

**Notebook:** `jupyter_notebooks/smote_resampling.ipynb`  
**Artifacts:** `models/smote_artifacts.pkl`

---

# Date-Time Feature Engineering & Business Insights

## Decision: Keep Original Year Columns

After exploratory analysis, the decision was made to **retain the original year columns** (`End_target_year`, `Interim_target_year`) rather than engineered temporal features.

### Features Evaluated

| Feature | Description | Formula | Final Decision |
|---------|-------------|---------|----------------|
| `End_target_year` | Raw end target year | Original column | ✅ **Kept** |
| `Interim_target_year` | Raw interim target year | Original column | ✅ **Kept** |
| `years_to_end_target` | Years until end target | End_target_year - 2025 | ❌ Not used |
| `years_to_interim_target` | Years until interim target | Interim_target_year - 2025 | ❌ Not used |
| `target_gap` | Gap between targets | End_target_year - Interim_target_year | ❌ Not used |

### Rationale

1. **Redundancy** — `years_to_end_target` and `years_to_interim_target` are perfectly correlated with the raw year columns (linear shift by reference year). Tree-based models handle either representation equally well.

2. **No performance improvement** — Model performance with temporal features (accuracy: 0.52, balanced: 0.43) was slightly worse than with original features (accuracy: 0.54, balanced: 0.46).

3. **Reference date dependency** — Temporal features require a hardcoded reference date (e.g., 2025), creating maintenance burden and potential for model drift if not updated during retraining.

4. **Interpretability** — Raw year values (e.g., "2030") are more universally interpretable than relative values (e.g., "5 years to target").

### Model Performance Comparison

| Configuration | Features | Test Accuracy | Balanced Accuracy |
|---------------|----------|---------------|-------------------|
| **Original (raw years)** | 72 | **0.54** | **0.46** |
| +3 temporal features | 75 | 0.52 | 0.43 |
| +target_gap only | 73 | 0.52 | 0.43 |

The original configuration with raw year columns achieved the best performance.

---

## Exploratory Analysis: Business Insights

Despite not being added to the model, the exploratory analysis revealed valuable business insights.

## Key Finding: Near-Term Targets Have Lowest Scope 3 Coverage

| Target Urgency | Yes (Scope_3) | No (Scope_3) |
|----------------|---------------|--------------|
| Near-term (≤2030) | **26.5%** | 31.8% |
| Mid-term (2031-2040) | 50.8% | 18.6% |
| Long-term (>2040) | 42.4% | 13.7% |

**Insight:** Companies with near-term targets (≤2030) have the **lowest** Scope_3 coverage rate (26.5%), while mid-term targets (2031-2040) have the highest (50.8%).

## Business Interpretation: Support Needed for Near-Term Pledgers

Companies with **near-term targets + low Scope 3 coverage** are priority candidates for support:

### Why This Matters

1. **High ambition, incomplete reporting** — Committed to aggressive climate goals but may lack infrastructure/expertise to measure Scope 3 emissions (supply chain, business travel, etc.)

2. **Regulatory risk** — As Scope 3 reporting becomes mandatory (EU CSRD, SEC climate rules), these companies face compliance gaps

3. **Credibility gap** — A 2030 net-zero target without Scope 3 coverage may be seen as "greenwashing" since Scope 3 typically accounts for 70-90% of total emissions

### Potential Use Cases

| Stakeholder | Action |
|-------------|--------|
| **Consulting firms** | Target these companies for Scope 3 measurement support |
| **Regulators** | Focus enforcement/guidance on near-term pledgers |
| **Investors** | Flag as ESG risk (ambitious claims, incomplete data) |
| **NGOs** | Prioritize engagement with companies that have committed but haven't fully measured |

### Model Application

The classification model could be used to **predict** which companies are likely to have incomplete Scope 3 reporting, helping stakeholders:
- Proactively identify where support is needed before targets come due
- Prioritize outreach to companies at highest risk of non-compliance
- Allocate resources efficiently for climate disclosure assistance

**Notebook:** `jupyter_notebooks/datetime_features.ipynb`

---

# Target Encoding for Classification (LabelEncoder)

## Question: Does encoding the target as 0, 1, 2, 3 assume ordinality?

**No.** Integer encoding of categorical targets does **not** assume ordinality for classification models.

## Why It's Safe

| Model | How it treats encoded target |
|-------|------------------------------|
| Logistic Regression (multinomial) | Learns separate coefficients per class; predicts probability distribution over discrete classes |
| Decision Tree | Treats as categorical labels; splits to maximize purity, no ordering assumed |
| k-NN | Counts neighbor labels; no distance between class values |
| Random Forest | Same as Decision Tree — majority vote |

These classifiers treat the integers as **class identifiers**, not ordered values. They never compute things like "class 2 is between class 1 and 3" or interpolate between classes.

## When Ordinality WOULD Matter

- **Regression** (predicting continuous values)
- **Ordinal regression** (e.g., ratings: 1 < 2 < 3 < 4 < 5)

## Our Target (`Scope_3_coverage`)

- Classes: Yes, No, Partial, Not Specified — **no natural ordering**
- Standard multiclass classification is correct
- `LabelEncoder` converts strings to integers for computational efficiency only

---
- **Interpretability** — "5 years to target" is more intuitive than "2030"
- **Model drift detection** — Can monitor if target urgency distributions change over time
- **Future models** — May help linear models or for time-series approaches

Keep the features in the pipeline for documentation/monitoring purposes.

---
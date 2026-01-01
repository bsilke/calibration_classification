# Feature Engineering Plan

**Project:** Net Zero Tracker – Scope 3 Coverage Classification  
**Date:** 20 December 2025  
**Based on:** EDA findings from `jupyter_notebooks/eda.ipynb`

---

## ⚠️ CRITICAL: Dropped Features (Target Leakage)

| Feature | Status | Reason |
|---------|--------|--------|
| `Scope_1_coverage` | ❌ **DROPPED** | Target leakage — decided simultaneously with Scope_3_coverage |
| `Scope_2_coverage` | ❌ **DROPPED** | Target leakage — decided simultaneously with Scope_3_coverage |

**Why removed:** Companies decide on Scope 1, 2, and 3 coverage *together* as part of their emissions reporting strategy. Including Scope 1/2 creates circular reasoning and artificially inflates model performance. Initial feature selection showed suspiciously high MI scores (0.276, 0.237) confirming the leakage.

**Decision Date:** 2025-12-20

---

## EDA-Driven Feature Engineering Summary

| EDA Finding | Feature Engineering Action |
|-------------|---------------------------|
| Numeric features show weak target correlation | Minimal transformation; focus on categorical features |
| `End_target`, `Industry`, `Published_plan` most predictive | Prioritize proper encoding; consider interactions |
| `Country` has 61 unique values (high cardinality) | Apply target encoding or regional grouping |
| Missing values imputed by Industry group | Already handled; missingness indicators optional |
| `log_revenue`, `log_employees` already normalized | No further transformation needed |
| ~5% outliers detected by Isolation Forest | Retained as legitimate edge cases |

---

## Feature Engineering Pipeline

```python
def plan_feature_engineering(df, target_col):
    """Planning function for feature engineering based on EDA insights."""
    plan = {
        'numerical_features': [],
        'categorical_features': [],
        'datetime_features': [],
        'missing_features': []
    }
    
    for col in df.columns:
        if col == target_col:
            continue
            
        if df[col].dtype in ['int64', 'float64']:
            plan['numerical_features'].append(col)
        elif df[col].dtype == 'object':
            plan['categorical_features'].append(col)
        elif 'datetime' in str(df[col].dtype):
            plan['datetime_features'].append(col)
            
        # Check for missing values
        if df[col].isnull().sum() > 0:
            plan['missing_features'].append(col)
    
    return plan
```

### Current Feature Inventory (from `data/interim/train.csv`)

| Type | Count | Features |
|------|-------|----------|
| **Numerical** | 4 | `log_revenue`, `log_employees`, `End_target_year`, `Interim_target_year` |
| **Categorical** | 18 | `Country`, `Geographic_region`, `Industry`, `End_target`, `Interim_target`, `Private_company`, `Published_plan`, `Status_of_end_target`, `GHGs_covered`, `Scope_1_coverage`, `Scope_2_coverage`, `Reporting_mechanism`, `Accountability_delivery`, `Carbon_credits`, `Separate_removal_target`, `Planning_removals`, `Historical_emissions`, `Race_to_zero_member` |
| **DateTime** | 0 | None (years are numeric) |
| **Missing** | 0 | All imputed during preprocessing |

---

## Practice Activity: Three Key Engineering Decisions

### 1. Numerical Feature for Transformation

**Selected Feature:** `End_target_year`

**Current State:** Raw year values (2017–2070), mean ~2043

**Proposed Transformation:** Binning into categorical groups

```python
def bin_target_year(year):
    """Convert end target year to meaningful business categories."""
    if year <= 2030:
        return 'Near-term (≤2030)'
    elif year <= 2040:
        return 'Medium-term (2031-2040)'
    elif year <= 2050:
        return 'Paris-aligned (2041-2050)'
    else:
        return 'Long-term (>2050)'
```

**Rationale:**
- EDA showed weak linear correlation between `End_target_year` and target (-0.03)
- However, the *category* of commitment timeline may matter more than exact year
- 2050 is a critical threshold (Paris Agreement alignment)
- Business interpretation: Companies with near-term targets may have different Scope 3 behaviors than those with distant commitments
- Tree-based models may benefit from explicit categorical splits rather than finding year thresholds

**Alternative Considered:** StandardScaler normalization — rejected because year values are already interpretable and tree-based models don't require scaling.

---

### 2. Categorical Feature Needing Advanced Encoding

**Selected Feature:** `Country`

**Current State:** 61 unique values with highly imbalanced distribution
- Top 3: United States (267), United Kingdom (134), Japan (98)
- Many countries have <10 samples

**Proposed Encoding:** Target Encoding with Smoothing

```python
from category_encoders import TargetEncoder

# Target encoding with smoothing to prevent overfitting on rare categories
country_encoder = TargetEncoder(
    cols=['Country'],
    smoothing=10,  # Blend with global mean for rare categories
    min_samples_leaf=5  # Minimum samples before using category mean
)

# CRITICAL: Fit only on training data
country_encoder.fit(X_train, y_train_encoded)
X_train['Country_encoded'] = country_encoder.transform(X_train)['Country']
X_val['Country_encoded'] = country_encoder.transform(X_val)['Country']
```

**Rationale:**
- One-hot encoding would create 61 sparse columns, reducing model efficiency
- Target encoding captures the relationship between country and Scope 3 coverage
- Smoothing prevents overfitting on countries with few samples (e.g., if "Brazil" has only 5 companies with 100% "Yes", raw target encoding would assign 1.0, likely overfitting)
- `Geographic_region` (8 values) provides a fallback grouping if target encoding proves unstable

**Data Leakage Prevention:**
- Target encoder MUST be fit on training data only
- Smoothing blends rare category means toward global mean, reducing leakage risk
- Cross-validation during encoding can further reduce leakage

**Alternative Considered:** Grouping by `Geographic_region` only — rejected because country-level patterns may exist (e.g., regulatory differences between UK and France despite both being in Europe).

---

### 3. Interaction Feature Worth Testing — **DEFERRED**

**Selected Interaction:** `Industry` × `End_target`

> **Status:** DEFERRED — Random Forest captures interactions automatically through tree splits. Explicit interaction feature adds complexity without guaranteed benefit. Revisit if switching to logistic regression or linear SVM.

**Current State:**
- `Industry`: 11 categories, Cramér's V = 0.22 with target
- `End_target`: 5 main categories, Cramér's V = 0.31 with target
- Both independently show moderate-high predictive power

**Proposed Feature:** Concatenated interaction term

```python
# Create interaction feature
X_train['Industry_EndTarget'] = X_train['Industry'] + '_' + X_train['End_target']

# Example values:
# "Industrials_Net Zero"
# "Financials_Carbon Neutral"
# "Energy_No target"
```

**Rationale:**
- Different industries may have different relationships between target type and Scope 3 coverage
- Example hypothesis: "Net Zero" commitment in Energy sector may be more likely to include Scope 3 than "Net Zero" in Financials (due to supply chain emissions relevance)
- EDA showed class overlap in numeric space — categorical interactions may provide better separation
- Tree-based models can capture interactions implicitly, but explicit features can help linear models and speed up tree learning

**Cardinality Consideration:**
- 11 Industries × 5 End_targets = 55 potential combinations
- Some combinations may be rare or non-existent
- Apply frequency threshold: only keep interactions with ≥10 samples, group others as "Other"

**Validation Approach:**
- Compare model performance with and without interaction feature
- Use feature importance to assess if interaction is being used
- If multicollinearity concerns arise, consider dropping original features

---

## Additional Engineering Considerations

### Features NOT Requiring Transformation

| Feature | Reason |
|---------|--------|
| `log_revenue` | Log-transformed; shows negative skew (-2.91) but acceptable for tree models |
| `log_employees` | Log-transformed; mild negative skew (-0.87) is good |
| `Interim_target_year` | Moderate skew (1.42); year values shouldn't be transformed |
| Binary features (13) | Already ordinal-encoded as Yes/No; no transformation needed |

### Handling Skewness Analysis

| Feature | Raw Skewness | After Transform | Decision |
|---------|--------------|-----------------|----------|
| `Revenue` | +6.23 (right) | -2.91 (left) | ✅ Accept: Log over-corrected but trees don't care |
| `Employees` | +9.18 (right) | -0.87 (left) | ✅ Good: Nearly symmetric |
| `Interim_target_year` | +1.42 (right) | N/A | ✅ Leave as-is: Year semantics matter |

**Why accept negative skew for `log_revenue`:**
1. RandomForest is **scale-invariant** — doesn't assume normality
2. Transformation stabilized variance (large values no longer dominate)
3. Alternative transforms (sqrt, Box-Cox) add complexity without benefit for trees
4. If using linear models later, consider `PowerTransformer(method='yeo-johnson')`

### Potential Future Features (Domain-Driven)

| Feature Idea | Data Required | Rationale |
|--------------|---------------|-----------|
| `years_until_target` | Reference date | Time pressure may influence reporting behavior |
| `target_ambition_score` | Expert mapping | Combine target type + timeline + coverage into single score |
| `industry_avg_coverage` | Aggregation | Compare company to industry peers |
| `company_age` | External data | Mature companies may have different reporting patterns |

---

## Implementation Priority

| Priority | Feature | Complexity | Expected Impact | Status |
|----------|---------|------------|-----------------|--------|
| **1** | Country frequency grouping | Medium | High — reduces cardinality (top 10 + Other) | ✅ Done |
| **2** | ~~Industry × End_target interaction~~ | Low | Medium — may capture conditional patterns | ⏸️ DEFERRED |
| **3** | End_target_year binning | Low | Low-Medium — categorical interpretation vs continuous | Not started |

---

## Validation Strategy

1. **Baseline Model:** Train RandomForest on current features (no new engineering)
2. **Incremental Testing:** Add one engineered feature at a time, measure balanced accuracy change
3. **Feature Importance:** Check if engineered features rank in top 10
4. **Overfitting Check:** Compare train vs validation performance — large gap indicates overengineering

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-12-20 | Selected `End_target_year` for binning | Weak linear correlation suggests categorical treatment may help |
| 2025-12-20 | Selected `Country` for target encoding | High cardinality (61) makes one-hot impractical |
| 2025-12-20 | Selected `Industry × End_target` interaction | Both features highly predictive; combination may capture sector-specific patterns |
| 2025-12-20 | Deprioritized numeric transformations | EDA showed numeric features have weak predictive power |
| 2025-12-21 | Changed Country to frequency grouping | Target encoding adds leakage risk; top 10 + Other is simpler |
| 2025-12-21 | **DEFERRED** interaction feature | Random Forest learns interactions via tree splits; explicit feature adds complexity without proven benefit |

---

## Next Steps

1. ~~Implement target encoding for `Country` in feature engineering script~~ → ✅ Done (frequency grouping instead)
2. ~~Create `Industry_EndTarget` interaction feature~~ → ⏸️ DEFERRED
3. Test `End_target_year` binning as optional experiment
4. ~~Build sklearn `ColumnTransformer` pipeline combining all transformations~~ → ✅ Done (`src/pipeline.py`)
5. Validate on holdout data before final model training

## Missing Data Handling - Column Exclusion

### Threshold Decision: **50% Missing**

A **50% missing value threshold** was applied to determine which columns to exclude from the dataset. This is a standard threshold that balances data retention with data quality.

### Rationale for 50% Threshold

| Threshold Option | Trade-off |
|------------------|-----------|
| **>70-80%** | Conservative - keeps more columns but may include unreliable features |
| **>50%** ✅ | **Chosen** - balanced approach, excludes columns where majority is missing |
| **>30-40%** | Aggressive - higher data quality but loses potentially useful features |

**Why 50%?**
- Columns with >50% missing have **more gaps than data**, making imputation unreliable
- Statistical power is compromised when majority of observations require imputation
- Risk of introducing imputation bias increases with missing percentage
- For classification tasks, features with sparse data contribute more noise than signal

### Variables Excluded (24 columns with >50% missing)

| Category | Variables | Missing % | Reason |
|----------|-----------|-----------|--------|
| **Borderline (50-70%)** | `Interim_target_percentage_reduction` | 52.4% | Exceeds threshold; imputation would dominate |
| | `Interim_target_baseline_year` | 50.2% | Exceeds threshold; majority missing |
| **High Missing (70-90%)** | `End_target_percentage_reduction` | 86.1% | Too sparse for reliable analysis |
| | `End_target_baseline_year` | 73.9% | Majority missing |
| | `Carbon_credits_conditions` | 70.5% | Limited reporting |
| **Very High Missing (>90%)** | `GHG_emissions`, `GHG_emissions_year`, etc. | 98-100% | Essentially empty columns |
| | `Territorial_emissions`, `Consumption_emissions` | 100% | No data available |
| | Various `_intensity_unit`, `_baseyear_emissions` | 93-99% | Too sparse |

### Full List of Excluded Columns

```
Interim_target_percentage_reduction (52.4%)
Interim_target_baseline_year (50.2%)
End_target_percentage_reduction (86.1%)
End_target_baseline_year (73.9%)
Carbon_credits_conditions (70.5%)
Max_emission_offset (98.5%)
Carbon_credits_conditions_other (93.5%)
Territorial_emissions (100.0%)
Consumption_emissions (100.0%)
International_aviation (100.0%)
International_shipping (100.0%)
GHG_emissions (99.7%)
GHG_emissions_year (98.6%)
GHG_emissions_source (100.0%)
End_target_intensity_unit (98.0%)
end_target_baseyear_emissions (99.7%)
End_target_target_emissions (99.7%)
End_target_bau_emissions (100.0%)
End_target_other (98.4%)
Interim_target_intensity_unit (96.1%)
Interim_target_baseyear_emissions (98.9%)
Interim_target_target_emissions (98.8%)
Interim_target_bau_emissions (100.0%)
Interim_target_other (93.6%)
```

### Variables Retained (25 columns with ≤50% missing)

These columns have sufficient data for reliable analysis and imputation:

| Missing Range | Count | Examples |
|---------------|-------|----------|
| **0% (Complete)** | 7 | `Name`, `Country`, `Company_annual_revenue`, `ID_Code` |
| **1-30% (Low)** | 17 | `Scope_3_coverage` (target), `Employees`, `Industry`, `End_target_year` |
| **30-50% (Moderate)** | 1 | `Interim_target_year` (38.7%) |

### Impact on Modeling

| Aspect | Effect |
|--------|--------|
| **Feature Count** | Reduced from ~49 to ~25 usable features |
| **Data Quality** | ✅ Higher - all retained features have majority non-missing |
| **Imputation Reliability** | ✅ Improved - imputation on ≤50% missing is more robust |
| **Model Performance** | ✅ Expected improvement - less noise from sparse features |

### Note on GHG_emissions

Although `GHG_emissions` was initially considered for log transformation, its **99.7% missing rate** makes it unsuitable for analysis. The `log_ghg_emissions` transformation created earlier will be based on essentially no data and should not be used as a model feature.

---

## Target Variable Handling - Missing Labels

### Decision: Remove Rows with Missing `Scope_3_coverage`

**Action**: Dropped all rows where the target variable `Scope_3_coverage` is missing.

| Metric | Value |
|--------|-------|
| Rows before | 2,076 |
| Missing labels | 193 (9.3%) |
| Rows after | 1,883 |

### Rationale

Supervised machine learning requires labeled examples to learn patterns. Rows without a target label cannot contribute to model training:

1. **No learning signal**: The model cannot learn from unlabeled examples in supervised classification
2. **Cannot evaluate**: These rows cannot be used for validation or testing either
3. **Imputation inappropriate**: Unlike features, target variables should **never** be imputed - this would create artificial/fabricated labels

### Alternative Approaches (Not Used)

| Approach | Why Not Applicable |
|----------|-------------------|
| **Impute target** | ❌ Creates fake labels - fundamentally flawed |
| **Semi-supervised learning** | ❌ Out of scope for this project |
| **Keep for prediction** | ❌ These are training data, not new unlabeled data |

### Final Target Distribution

After removing missing labels, the dataset has a reasonably balanced 4-class distribution:

| Class | Count | Percentage |
|-------|-------|------------|
| Yes | 762 | 40.5% |
| Not Specified | 411 | 21.8% |
| Partial | 363 | 19.3% |
| No | 347 | 18.4% |

> **Note**: "Not Specified" is a valid class label (indicating companies that haven't specified their Scope 3 coverage), distinct from missing data.

---

## Feature Imputation Strategy - Group-Based Imputation

### Decision: Use Industry-Based Group Imputation

Instead of global imputation (using overall median/mode), we apply **group-based imputation** using `Industry` as the grouping variable.

### Implementation

| Feature Type | Strategy | Fallback |
|--------------|----------|----------|
| **Numerical** | Industry-specific median | Global median |
| **Categorical** | Industry-specific mode | Global mode |

### Rationale

Companies within the same industry share similar characteristics:

1. **Revenue patterns**: Tech companies have different typical revenues than retail or utilities
2. **Employee counts**: Service industries differ from manufacturing in workforce size
3. **Reporting behaviors**: Some industries have stricter reporting requirements

Using industry-specific values provides **more accurate imputation** than global statistics.

### Additional Feature: Missing Indicators

Before imputation, binary indicator columns (`_was_missing`) are created for columns with missing values. This captures the **missingness pattern** as a potential predictive feature - e.g., companies that don't report certain metrics may systematically differ from those that do.

### Code Implementation

```python
def group_imputation(df, target_col, group_col):
    # Fill missing values with group median (numerical) or mode (categorical)
    df[target_col] = df.groupby(group_col)[target_col].transform(
        lambda x: x.fillna(x.median())  # or x.mode() for categorical
    )
    return df
```

### Why Not Simple Global Imputation?

| Method | Pros | Cons |
|--------|------|------|
| **Global median/mode** | Simple, fast | Ignores industry differences |
| **Group-based** ✅ | Context-aware, more accurate | Slightly more complex |
| **KNN imputation** | Uses multiple features | Computationally expensive |
| **MICE** | Iterative, sophisticated | Overkill for this dataset size |

Group-based imputation provides a good balance between accuracy and simplicity for this dataset.

### Note: Imputation Requirements by Model Type

Whether imputation is **strictly necessary** depends on the model implementation:

| Library/Model | Handles NaN? | Imputation Required? |
|---------------|--------------|---------------------|
| **scikit-learn RandomForest** | ❌ No | ✅ Yes - throws `ValueError: Input contains NaN` |
| **XGBoost** | ✅ Yes | ❌ No - learns optimal split direction for missing values |
| **LightGBM** | ✅ Yes | ❌ No - efficient native handling |
| **CatBoost** | ✅ Yes | ❌ No - also handles categoricals natively |
| **HistGradientBoostingClassifier** | ✅ Yes | ❌ No - scikit-learn's native option (v1.0+) |

**For this project**: Since we're using scikit-learn's `RandomForestClassifier`, imputation is **required**. However, for tree-based models:

- The exact imputed value matters less than for linear models
- Trees split on thresholds, so imputed values simply get grouped together
- Even simple median/mode imputation works well

**Alternative approach**: If imputation is undesirable, consider switching to XGBoost or LightGBM, which:
- Learn the optimal direction for missing values during training
- Implicitly capture missingness patterns
- Often outperform manual imputation strategies
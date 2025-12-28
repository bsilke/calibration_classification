# Exploratory Data Analysis Report
## Net Zero Tracker – Scope 3 Coverage Classification

**Report Date:** 20 December 2025  
**Analyst:** S. Bumann
**Version:** 1.0

---

## 1. Executive Summary

The analysis examined corporate net-zero commitment data from the Net Zero Tracker to predict Scope 3 emissions coverage status. The cleaned dataset contains **1,883 companies** across 11 industries, with a 4-class target variable (`Yes`, `No`, `Partial`, `Not Specified`). Key findings indicate that categorical features—particularly `End_target`, `Industry`, and `Published_plan`—demonstrate the strongest predictive association with the target, while numeric features show weak correlations. The dataset exhibits moderate class imbalance (2.2:1 ratio), which was addressed through stratified splitting rather than SMOTE resampling (tested but degraded performance). Two data quality issues were identified and resolved: an invalid year value (1000) and an extreme revenue outlier ($2.35 trillion). The data is considered ready for modeling with tree-based classifiers recommended due to the mixed feature types and categorical dominance.

---

## 2. Data Provenance

| Attribute | Details |
|-----------|---------|
| **Source** | Net Zero Tracker (https://zerotracker.net/) |
| **Raw File** | `data/raw/net_zero_tracker.csv` |
| **Version** | Downloaded November 2025 |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Access Restrictions** | None – publicly available dataset |
| **Entity Types** | Filtered to companies only (excluded countries, regions, cities) |
| **Original Records** | 4,183 entities (all types) |
| **Filtered Records** | 2,086 companies → 1,883 after cleaning |
| **Data Integrity** | SHA-256 checksum verified via `src/generate_checksums.py` |

**Preprocessing Pipeline:** All transformations are documented and reproducible via `src/preprocessing_pipeline.py`, with configuration stored in `RUN_CONFIG.json`.

---

## 3. Class Distribution

The target variable `Scope_3_coverage` contains four classes representing whether companies report their Scope 3 (supply chain) emissions:

| Class | Count | Percentage | Description |
|-------|-------|------------|-------------|
| **Yes** | 760 | 40.4% | Full Scope 3 coverage reported |
| **Not Specified** | 412 | 21.9% | Coverage status not disclosed |
| **Partial** | 364 | 19.3% | Some Scope 3 categories reported |
| **No** | 347 | 18.4% | No Scope 3 emissions reported |

**Imbalance Assessment:**
- Majority class: "Yes" (40.4%)
- Minority class: "No" (18.4%)
- Imbalance ratio: **2.2:1**
- Classification: Moderate imbalance

**Visualization:** Target distribution bar chart saved to `other_files/eda_target_distribution.png`

---

## 4. Variable Summaries

### 4.1 Feature Overview

The cleaned dataset contains **22 features** (after dropping leakage columns):

| Type | Count | Features |
|------|-------|----------|
| **Numeric** | 4 | `log_revenue`, `log_employees`, `End_target_year`, `Interim_target_year` |
| **Categorical** | 18 | `Industry`, `Country`, `End_target`, `Published_plan`, etc. |

### 4.2 Numeric Feature Statistics

| Feature | Min | Max | Mean | Std | Notes |
|---------|-----|-----|------|-----|-------|
| `log_revenue` | 0.00 | 27.35 | 23.45 | 2.31 | Log-transformed; approximately normal |
| `log_employees` | 0.00 | 14.51 | 10.12 | 1.89 | Log-transformed; slight left skew |
| `End_target_year` | 2017 | 2070 | 2043 | 8.2 | Most targets set for 2050 |
| `Interim_target_year` | 2020 | 2050 | 2031 | 4.8 | Typically 15-20 years before end target |

### 4.3 Key Categorical Features

| Feature | Unique Values | Most Common | Notes |
|---------|---------------|-------------|-------|
| `Industry` | 11 | Industrials (201) | Imbalanced; some industries <30 samples |
| `End_target` | 5 | Net Zero (892) | Strong target-association |
| `Country` | 67 | United States (267) | High cardinality; grouped in modeling |
| `Published_plan` | 4 | In Progress (498) | Ordinal encoding recommended |

**Visualizations:**
- `other_files/eda_log_revenue_distribution.png`
- `other_files/eda_industry_distribution.png`

---

## 5. Key Correlations & Feature Insights

### 5.1 Numeric Feature Correlations

Pearson correlation analysis between numeric features and the encoded target variable revealed weak linear relationships:

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| `log_revenue` | +0.08 | Negligible positive |
| `log_employees` | +0.06 | Negligible positive |
| `End_target_year` | -0.03 | No relationship |
| `Interim_target_year` | +0.02 | No relationship |

**Key Finding:** Numeric features alone have limited predictive power. Company size and target timing are not strong indicators of Scope 3 coverage.

### 5.2 Categorical Feature Associations

Cramér's V statistic was used to measure categorical-target associations:

| Feature | Cramér's V | Predictive Potential |
|---------|------------|---------------------|
| `End_target` | 0.31 | **High** |
| `Industry` | 0.22 | **High** |
| `Published_plan` | 0.19 | **Moderate-High** |
| `Scope_1_coverage` | 0.58 | DROPPED (leakage) |
| `Scope_2_coverage` | 0.54 | DROPPED (leakage) |

### 5.3 Top 3 Predictive Features

1. **`End_target`** – Companies committing to "Net Zero" targets are more likely to report comprehensive Scope 3 coverage than those with "Carbon Neutral" or no specified targets.

2. **`Industry`** – Sectors with complex supply chains (Consumer Discretionary, Industrials) show different Scope 3 reporting patterns than service-based industries (Financials, Communication Services).

3. **`Published_plan`** – Companies with published climate action plans demonstrate higher rates of Scope 3 disclosure, reflecting organizational maturity in sustainability reporting.

**Visualization:** `other_files/eda_correlation_heatmap.png`

---

## 6. Missing Values & Outliers

### 6.1 Missing Value Analysis

| Stage | Missing Values | Action |
|-------|----------------|--------|
| Raw data | 151,072 across all fields | Filtered to relevant columns |
| `GHG_emissions` | >50% missing | Column dropped |
| `Scope_3_coverage` (target) | 193 rows | Rows dropped |
| `Industry` | 8 rows | Rows dropped (needed for imputation) |
| Remaining features | Variable | Group-median/mode imputation by Industry |

**Imputation Strategy:**
- Numeric features: Group median by Industry, with global median fallback
- Categorical features: Group mode by Industry, with global mode fallback
- Imputation statistics calculated on training set only to prevent data leakage

### 6.2 Outlier Detection

**Isolation Forest Results (5% contamination):**
- Total samples analyzed: 1,129
- Outliers detected: ~57 (5%)
- Outliers retained in dataset (legitimate edge cases)

**Data Errors Removed:**

| Issue | Value | Company | Resolution |
|-------|-------|---------|------------|
| Invalid year | `End_target_year = 1000` | BAIC Motor | Row dropped |
| Extreme revenue | $2.35 trillion | Unknown | Row dropped |

**Year Validation:** All `End_target_year` and `Interim_target_year` values now fall within 2000-2100.

**Visualizations:** `other_files/eda_isolation_forest_outliers.png`

---

## 7. Ethical & Licensing Notes

### 7.1 Data Usage Rights

The Net Zero Tracker data is released under **CC BY 4.0**, permitting:
- ✅ Sharing and redistribution
- ✅ Adaptation and derivative works
- ✅ Commercial use
- ⚠️ Attribution required

### 7.2 Ethical Considerations

| Concern | Assessment | Mitigation |
|---------|------------|------------|
| **Selection Bias** | Dataset includes only companies with public commitments; may not represent all corporations | Results should not be generalized to companies without net-zero commitments |
| **Temporal Validity** | Corporate commitments change over time | Model should be periodically retrained with updated data |
| **Label Quality** | "Not Specified" category may include companies that report but were not captured | Conservative interpretation of predictions for this class |
| **Regional Bias** | Overrepresentation of companies from developed economies | May underperform for emerging market companies |

### 7.3 Data Leakage Prevention

The following features were identified as **high leakage risk** and dropped:
- `Scope_1_coverage`
- `Scope_2_coverage`

**Rationale:** Companies that report Scope 1/2 emissions almost always report Scope 3, creating near-perfect correlation that would not generalize to prediction scenarios.

---

## 8. Next-Step Recommendations

### 8.1 Feature Engineering

| Recommendation | Priority | Rationale |
|----------------|----------|-----------|
| Create `target_gap` feature | Low | Difference between end and interim target years; tested but showed no improvement |
| Group rare industries | Medium | Industries with <30 samples may be combined into "Other" |
| Encode `Country` by region | Medium | Reduce cardinality while preserving geographic signal |
| Interaction features | Low | `Industry × End_target` may capture sector-specific commitment patterns |

### 8.2 Modeling Recommendations

1. **Algorithm Selection:** Tree-based models (Random Forest, XGBoost) are recommended due to:
   - Native handling of mixed feature types
   - Robustness to feature scaling
   - Ability to capture non-linear relationships

2. **Evaluation Strategy:**
   - Primary metric: **Balanced Accuracy** (accounts for class imbalance)
   - Secondary metrics: Per-class precision, recall, F1-score
   - Special attention to "Partial" class (historically low recall)

3. **Resampling Decision:**
   - SMOTE was tested and **not recommended** (degraded balanced accuracy from 0.466 to 0.448)
   - Stratified splitting preserves class proportions across train/val/test

### 8.3 Data Improvements

| Improvement | Feasibility | Impact |
|-------------|-------------|--------|
| Incorporate temporal features (commitment date) | Medium | May capture reporting trends over time |
| Add external financial data | Low | Could strengthen revenue/employee features |
| Include company sustainability ratings | Medium | Third-party ESG scores may improve predictions |

---

## Appendix: Saved Artifacts

### Visualizations (`other_files/`)
- `eda_log_revenue_distribution.png`
- `eda_industry_distribution.png`
- `eda_correlation_heatmap.png`
- `eda_isolation_forest_outliers.png`
- `eda_pairplot.png`
- `eda_multivariate.png`
- `eda_target_distribution.png`

### Data Files (`data/interim/`)
- `train.csv` – 1,129 samples (60%)
- `val.csv` – 377 samples (20%)
- `test.csv` – 377 samples (20%)

### Configuration
- `RUN_CONFIG.json` – All preprocessing parameters
- `logs/preprocess.log` – Pipeline execution log

---

*Report generated from EDA notebook: `jupyter_notebooks/eda.ipynb`*

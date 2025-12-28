# Feature Engineering Lessons Learned

**Project:** Net Zero Tracker – Scope 3 Coverage Classification  
**Date:** 20 December 2025

---

## 1. Which engineered feature added the greatest predictive value, and why?

**`Interim_target_year_missing`** — the missingness indicator for interim target year.

With 33.5% of companies not specifying an interim target year, this binary feature captures a potential MNAR (Missing Not At Random) pattern. Companies with vague, non-specific climate commitments may be less likely to comprehensively report Scope 3 emissions. The missingness itself signals something about corporate transparency and commitment level.

**Why it matters:** Unlike imputed values that estimate what the year *might* be, the indicator directly captures *whether the company bothered to specify* — a behavioral signal that correlates with our target.

---

## 2. What transformation had the most surprising impact?

**Dropping `Scope_1_coverage` and `Scope_2_coverage`.**

Initially, feature selection showed these had the highest mutual information scores (0.276 and 0.237). This seemed like great predictive power — until it was realized it was **target leakage**. Companies decide on Scope 1, 2, and 3 coverage simultaneously during emissions reporting.

**The surprise:** The "best" features were actually the most dangerous. Removing them likely reduced our apparent accuracy but gave us a model that will actually generalize to new data.

---

## 3. Which step took the longest to debug, and what was the challenge?

**Industry-group imputation with proper data leakage prevention.**

The challenge was ensuring imputation statistics were calculated **only on training data** and then applied consistently to validation and test sets. The initial implementation risked using global statistics that included test data.

**Solution:** Split data BEFORE imputation, calculate group medians/modes on training set only, then apply those learned values to all splits. This required restructuring the pipeline from a single-pass approach to a multi-stage process.

---

## 4. How will feature engineering choices affect model interpretability?

| Choice | Interpretability Impact |
|--------|------------------------|
| **Ordinal encoding** (11 features) | ✅ Preserves meaning — "Achieved" > "Proposed" is intuitive |
| **Country grouping** (top 10 + Other) | ✅ Reduces from 61 to 11 categories — easier to explain "USA vs Other" |
| **Log transforms** (revenue, employees) | ⚠️ Coefficients need back-transformation for business interpretation |
| **One-hot encoding** | ⚠️ Creates 70 features — feature importance spreads across dummies |
| **Missingness indicator** | ✅ Directly interpretable — "company didn't specify" is a clear concept |

**Overall:** The pipeline prioritizes interpretability by using ordinal encoding where order exists and reducing high-cardinality features. The tradeoff is 70 encoded features, which may require aggregation (e.g., summing importance of all country dummies) for stakeholder communication.

---

## 5. What would I do differently next time?

1. **Start with leakage analysis earlier.**  Scope 1/2 leakage was discovered during feature selection — should have caught this during EDA by asking "when is this information known?"

2. **Consider target encoding for Country.** Frequency-based grouping is chosen (top 10 + Other) for simplicity, but target encoding with smoothing might capture more nuanced country-target relationships without 11 one-hot columns.

3. **Add more missingness indicators.** Only `Interim_target_year_missing` was added (33.5% missing). Could experiment with indicators for `End_target_year` (12.7%) and `Employees` (5.8%) to see if missingness patterns add predictive value.

4. **Document assumptions earlier.** The `preprocessing_config.yml` was created mid-process. Starting with a configuration file from day one would have made decisions more traceable.

---

## Summary

The most valuable lessons from this feature engineering process:

- **Leakage detection is critical** — high mutual information can be a warning sign, not just good news
- **Missingness is information** — indicator variables can capture behavioral patterns
- **Two-stage pipelines work well** — complex imputation in preprocessing, simple encoding in sklearn Pipeline
- **Document everything** — `preprocessing_config.yml`, `encoding_strategy.md`, and logs enable reproducibility

---

## Files Produced

| File | Purpose |
|------|---------|
| `src/preprocessing_pipeline.py` | Data cleaning, imputation, scaling |
| `src/pipeline.py` | Encoding + model integration |
| `preprocessing_config.yml` | Imputation strategies with rationale |
| `encoding_strategy.md` | Encoding decisions |
| `feature_engineering_plan.md` | EDA-driven engineering decisions |
| `final_features.txt` | Final feature list |
| `logs/preprocess.log` | Preprocessing audit trail |
| `logs/feature_engineering.log` | Encoding audit trail |
| `PIPELINE_README.md` | Pipeline documentation |

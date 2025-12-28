# Quality Assurance Checklist

**Dataset:** Net Zero Tracker - Scope_3_coverage Classification  
**Date:** 2025-12-20  
**Author:** [Your Name]  
**QA Notebook:** `jupyter_notebooks/qa_checks.ipynb`

---

## 1. Schema Validation

| Check | Status | Notes |
|-------|--------|-------|
| Are all expected columns present? | ✅ | 23 columns as expected |
| Do column data types match what you planned? | ✅ | Numeric cols: float64, Categorical: object |
| Are there any unexpected or unused columns? | ✅ | Schema consistent across train/val/test |

**Fix Plan (if any):**
> N/A - All checks passed

---

## 2. Missingness Audit

| Check | Status | Notes |
|-------|--------|-------|
| Are missing values within acceptable thresholds (<10%)? | ✅ | 0% missing after imputation |
| Have you documented whether missingness is random or systematic? | ✅ | See `data_preprocessing_v2.ipynb` |
| Have imputation strategies been applied and noted? | ✅ | Group-based (Industry) median/mode |

**Fix Plan (if any):**
> N/A - Imputation applied correctly with training-only statistics

---

## 3. Outlier Review

| Check | Status | Notes |
|-------|--------|-------|
| Have you run outlier detection on all numeric features? | ✅ | IQR method applied to 4 numeric features |
| Are outliers flagged, capped, removed or explained? | ✅ | Log transforms applied to revenue/employees |
| Have extreme values been reviewed with domain context? | ✅ | Year columns reviewed, log features handled |

**Fix Plan (if any):**
> N/A - Log transforms address skewness in revenue/employee data

---

## 4. Class Balance Check

| Check | Status | Notes |
|-------|--------|-------|
| Have you calculated and visualized class distribution? | ✅ | Yes: 40% / 22% / 19% / 18% |
| Do all classes meet a minimum threshold for samples (≥50)? | ✅ | Min class "No" has 208 samples |
| Have you documented any resampling decisions (SMOTE, undersampling)? | ✅ | Currently using 'none' - imbalance ratio 2.2:1 |

**Fix Plan (if any):**
> May apply SMOTE/undersampling if model performance on minority classes is poor

---

## 5. Encoding Validation

| Check | Status | Notes |
|-------|--------|-------|
| Are all categorical features encoded as planned? | ✅ | 11 ordinal, 5 nominal features |
| Have you checked for category collapse (different labels encoded the same)? | ✅ | No collapse detected |
| Do the dimensions of your encoded matrix match expectations? | ✅ | ~85 dimensions after encoding |

**Fix Plan (if any):**
> N/A - Encoding plan documented in `modeling_pipeline.ipynb`

---

## 6. Scaling Verification

| Check | Status | Notes |
|-------|--------|-------|
| Is scaling applied only to numeric features? | ✅ | StandardScaler on 4 numeric features |
| Was the scaler fit only on the training set? | ✅ | See DataFrameEncoder in pipeline |
| Did you visualize feature distributions before and after scaling? | ✅ | Visualized in qa_checks.ipynb |

**Fix Plan (if any):**
> N/A - Scaling correctly applied

---

## 7. Leakage Prevention

| Check | Status | Notes |
|-------|--------|-------|
| Have you run a leakage detection check (future or label-derived features)? | ✅ | Scope_1/2 identified as leakage |
| Are any time-based features filtered correctly (no future info)? | ✅ | Year columns reviewed - no future leakage |
| Are preprocessing steps cleanly separated by train/test? | ✅ | Split BEFORE imputation in v2 |

**Fix Plan (if any):**
> Scope_1_coverage and Scope_2_coverage dropped in modeling pipeline (logical dependency)

---

## 8. Reproducibility

| Check | Status | Notes |
|-------|--------|-------|
| Is your preprocessing script version-controlled? | ✅ | `src/preprocessing_pipeline.py` |
| Is there a config file with parameters? | ✅ | `RUN_CONFIG.json` |
| Have you saved your environment details (requirements.txt)? | ✅ | `requirements.txt` generated |

**Fix Plan (if any):**
> N/A - Full reproducibility infrastructure in place

---

## Summary

| Section | Pass | Fail | Total |
|---------|------|------|-------|
| Schema Validation | 3 | 0 | 3 |
| Missingness Audit | 3 | 0 | 3 |
| Outlier Review | 3 | 0 | 3 |
| Class Balance Check | 3 | 0 | 3 |
| Encoding Validation | 3 | 0 | 3 |
| Scaling Verification | 3 | 0 | 3 |
| Leakage Prevention | 3 | 0 | 3 |
| Reproducibility | 3 | 0 | 3 |
| **TOTAL** | **24** | **0** | **24** |

---

## Reproducibility Artifacts

| File | Purpose |
|------|---------|
| `RUN_CONFIG.json` | All configuration parameters and random seeds |
| `requirements.txt` | Python environment packages |
| `logs/preprocess.log` | Complete preprocessing log |
| `data/raw/*.sha256` | Data integrity checksums |
| `src/preprocessing_pipeline.py` | Reproducible preprocessing script |
| `src/reproducibility.py` | Logging and checksum utilities |

---

## Sign-off

- [x] All critical checks passed
- [x] Fix plans documented for failed checks
- [x] Ready to proceed with modeling

**Reviewer:** _______________  
**Date:** 2025-12-20

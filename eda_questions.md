# EDA Questions: Critical Questions Before Modeling

The following three questions must be answered through exploratory data analysis before proceeding to model training.

---

## 1. What is the class distribution of the target variable, and are any classes underrepresented?

**Why this matters:**  
The target variable `Scope_3_coverage` has four classes (Yes, No, Partial, Not Specified). If any class represents less than 10-15% of the data, the model may struggle to learn patterns for that class, leading to poor recall. Additionally, understanding the distribution informs decisions about stratified splitting, class weights, and whether techniques like SMOTE should be considered.

**What to examine:**
- Count and percentage of each class
- Whether the minority class has sufficient samples for learning (rule of thumb: >100 samples)
- Whether class proportions are preserved across train/val/test splits

**Finding from this project:**  
The distribution is moderately imbalanced: Yes (40%), Not Specified (22%), Partial (19%), No (18%). The minority class (No) at 18% is above the 10% threshold, suggesting SMOTE may not be necessary—and testing confirmed this.

---

## 2. Which features have high missingness, and is the missingness random or systematic?

**Why this matters:**  
Features with >40% missing values require careful handling. If missingness is systematic (e.g., smaller companies don't report emissions), imputation may introduce bias. If missingness is random, imputation is safer but still affects model reliability for records with imputed values.

**What to examine:**
- Percentage of missing values per column
- Whether missingness correlates with the target variable (potential leakage)
- Whether missingness clusters by industry, region, or company size
- Columns with >50% missing that should be dropped entirely

**Finding from this project:**  
`GHG_emissions` had >50% missing and was dropped. `Company_annual_revenue` and `Employees` had ~30-40% missing and were imputed using industry-group medians fitted on training data only.

---

## 3. Are there features with unexpected data types or encoding inconsistencies?

**Why this matters:**  
Numeric fields stored as strings (e.g., "$1,000,000" or "1.5M") will fail silently in sklearn pipelines or be treated as categorical. Categorical fields with inconsistent encoding (e.g., "Yes", "yes", "Y") create artificial cardinality. These issues must be resolved before encoding and modeling.

**What to examine:**
- Data types of all columns (`df.dtypes`)
- Unique values in categorical columns for inconsistencies
- Numeric columns stored as `object` type
- Special characters or formatting in numeric fields

**Finding from this project:**  
`Company_annual_revenue` and `Employees` were stored as strings with currency symbols and commas. These were converted to numeric types during preprocessing. Categorical columns like `Scope_3_coverage` had consistent encoding (Yes/No/Partial/Not Specified).

---

## Summary Checklist

| Question | Status | Key Finding |
|----------|--------|-------------|
| Target class distribution | ✅ Answered | Moderate imbalance, minority class at 18% |
| Feature missingness patterns | ✅ Answered | `GHG_emissions` dropped, others imputed by industry |
| Data type inconsistencies | ✅ Answered | Revenue/Employees converted from string to numeric |

These questions were addressed during the preprocessing phase. The findings informed decisions about imputation strategy, feature dropping, and the decision not to use SMOTE.

# EDA Reflection

**Project:** Net Zero Tracker – Scope 3 Coverage Classification  
**Date:** 20 December 2025

---

## Which EDA visualization surfaced the most surprising or unexpected pattern?

The **pairplot of numeric features colored by target class** revealed the most surprising insight: there was virtually no visual separation between the four Scope 3 coverage classes across any numeric feature combination. It was expected that larger companies (higher `log_revenue`, `log_employees`) would show meaningfully different Scope 3 reporting patterns—perhaps with bigger firms more likely to report comprehensively due to regulatory pressure and resources. Instead, the classes overlapped almost entirely in the numeric feature space.

This finding directly contradicted the initial hypothesis that company size would be a strong predictor. The Cramér's V analysis confirmed this, showing that categorical features like `End_target` (V=0.31) and `Industry` (V=0.22) carried far more predictive signal than any numeric feature (all correlations <0.10).

---

## How will that insight influence the approach to feature engineering?

Given the weak predictive power of numeric features, the feature engineering strategy will prioritize:

1. **Categorical encoding optimization** – Since `End_target`, `Industry`, and `Published_plan` drive predictions, careful attention will be given to encoding strategies. One-hot encoding for nominal features and ordinal encoding for features with natural ordering (e.g., `Published_plan`: "No" < "In Progress" < "Yes") will be tested.

2. **Deprioritizing numeric transformations** – Further numeric feature engineering (e.g., polynomial features, binning, PCA) is unlikely to improve model performance given the poor class separability observed. No additional numeric-based features will be engineered.

3. **Interaction features** – An interaction term between `Industry × End_target` will be explored, as different industries may have varying relationships between target commitment type and Scope 3 reporting behavior.

4. **Preserving original year columns** – Engineered temporal features (`years_to_end_target`, `target_gap`) were tested but showed no improvement over raw year values, so the original columns will be retained without transformation.

---

## Which external resource supported the learning the most and why?

The **scikit-learn documentation on Isolation Forest** was the most valuable external resource during this EDA. While typical statistical outlier detection methods (IQR, Z-scores) are useful, Isolation Forest provided a multivariate approach that could identify outliers based on combinations of features rather than examining each feature in isolation.

The documentation's explanation of the `contamination` parameter was particularly helpful—setting it to 0.05 allowed detection of the most anomalous 5% of observations without requiring assumptions about data distribution. This revealed outliers that univariate methods would have missed, such as companies with unusual combinations of small revenue but very large employee counts.

Additionally, the scikit-learn user guide's discussion of when tree-based anomaly detection outperforms distance-based methods (like Local Outlier Factor) helped justify the choice for this mixed-type dataset where categorical features dominate.

---

## Key Takeaways

1. **Don't assume numeric features will dominate** – In business datasets, categorical variables often carry more signal than continuous measurements.

2. **Visualize before engineering** – The pairplot saved significant time by revealing that complex numeric transformations would be pointless.

3. **Document data quality decisions** – Removing the year=1000 outlier and $2.35T revenue error was straightforward, but documenting the rationale ensures reproducibility and transparency for future collaborators.

4. **Test assumptions empirically** – SMOTE was initially assumed necessary for the 2.2:1 class imbalance, but empirical testing showed it degraded performance. The EDA process reinforced the importance of validating intuitions with data.

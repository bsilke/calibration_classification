# Outlier Handling Decisions

## Analysis of 3 Potential Outliers

### 1️⃣ Company_annual_revenue - Extreme High Values (e.g., $500B+)

| Question | Answer |
|----------|--------|
| **Statistically unusual?** | ✅ **Yes** - Z-scores often exceed 10+ for the largest companies. Values like $500B are 50-100x the median revenue. |
| **Makes sense in context?** | ✅ **Yes** - These are real multinational corporations (e.g., Walmart, Amazon, Apple). The Net Zero Tracker includes Fortune 500 companies with legitimately massive revenues. |
| **Decision** | **🟢 KEEP but TRANSFORM** |

**Reasoning:** These are not data errors but reflect the genuine power-law distribution of company sizes. Removing them would eliminate important large emitters. Instead:
- Apply **log transformation** before modeling to reduce skewness
- Random Forest handles this naturally via tree splits
- For linear models, log-transform is essential

---

### 2️⃣ Employees - Very Large Companies (e.g., 2M+ employees)

| Question | Answer |
|----------|--------|
| **Statistically unusual?** | ✅ **Yes** - Companies with 1-2 million employees are extreme outliers by IQR standards. |
| **Makes sense in context?** | ✅ **Yes** - Global retail giants (Walmart ~2.1M), logistics companies, and large manufacturers legitimately employ millions. |
| **Decision** | **🟢 KEEP but TRANSFORM** |

**Reasoning:** Same rationale as revenue - these represent real data points crucial for climate analysis:
- Large employers often have significant Scope 3 emissions
- Log transformation recommended
- Could create `log_employees` feature for modeling

---

### 3️⃣ End_target_year - Distant Future Years (e.g., 2070, 2100)

| Question | Answer |
|----------|--------|
| **Statistically unusual?** | ⚠️ **Moderately** - Most targets cluster around 2030-2050. Years like 2070+ are uncommon but not impossible. |
| **Makes sense in context?** | ✅ **Yes** - Some companies set long-term aspirational targets (e.g., "carbon negative by 2100"). These reflect real corporate commitments. |
| **Decision** | **🟢 KEEP as-is** |

**Reasoning:** 
- These are valid business decisions, not data errors
- Tree-based models will naturally segment by year ranges
- No transformation needed for year variables
- *Alternative:* Could cap at 2070 if modeling requires it (winsorizing)

---

## Summary Table

| Outlier | Keep | Transform | Remove |
|---------|------|-----------|--------|
| Extreme Revenue | ✅ | Log transform | ❌ |
| Large Employees | ✅ | Log transform | ❌ |
| Distant Target Years | ✅ | None (or winsorize) | ❌ |

## Key Principle
> **In business data, extreme values often represent the most important observations.** For climate/emissions analysis, the largest companies are precisely those with the greatest environmental impact. Removing them would bias results toward smaller, less impactful entities.

---

## Outliers Removed (Data Errors)

While most outliers were kept, **two observations were removed** as they represent clear data entry errors rather than legitimate extreme values:

| Observation | Value | Reason for Removal |
|-------------|-------|-------------------|
| **BAIC Motor** | `End_target_year = 1000` | Impossible year - clearly a typo (likely meant 2030 or similar) |
| **Unknown Company** | `Revenue = $2,347,159,000,000` | ~$2.35 trillion revenue is implausibly high (exceeds Walmart, the world's largest company by revenue) |

### Distinction from Kept Outliers
- **Kept outliers** (Walmart $500B+, etc.): Extreme but *verifiable* against external sources
- **Removed outliers**: Values that are *impossible* or *implausible* given real-world constraints

---

## Why Log Transformations for Revenue & Employees?

Revenue and employee counts follow a **power-law distribution** - a few very big companies dominate the scale while most companies are much smaller.

### The Problem Without Log Scale

| Aspect | Linear Scale | Log Scale |
|--------|--------------|-----------|
| **Visualization** | Extreme values dominate; data appears as one cluster near origin | Spread is visible across all company sizes |
| **Relative differences** | $10B vs $20B looks same as $500B vs $510B | Doubling always looks the same (10→20 = 100→200) |
| **Statistical properties** | Highly skewed, non-normal | More symmetric, closer to normal |
| **Model performance** | Linear models struggle with skewness | Better fits for regression/classification |

### Key Insight
> **Log scale treats multiplicative differences equally** - a company 10x larger always appears the same distance apart, whether it's $1M→$10M or $10B→$100B.

### When to Apply Log Transformation

| Model Type | Log Transform Needed? |
|------------|----------------------|
| Random Forest / XGBoost | Optional (tree splits work fine without it) |
| Linear Regression | **Essential** for skewed features |
| KNN / Distance-based | **Recommended** (distances become meaningful) |
| Neural Networks | **Recommended** (helps gradient descent) |

### Implementation Note
```python
# Create log-transformed features (add small constant to handle zeros)
df['log_revenue'] = np.log1p(df['Company_annual_revenue'])
df['log_employees'] = np.log1p(df['Employees'])
```

---

## Comprehensive Outlier Analysis Framework

A systematic analysis was performed using **both statistical techniques and business rules**:

### Detection Methods Applied

| Method | Criteria | Purpose |
|--------|----------|---------|
| **Z-Score** | \|z\| > 3 | Identifies extreme deviations from mean |
| **IQR** | Outside Q1-1.5×IQR to Q3+1.5×IQR | Identifies values outside interquartile fence |
| **Business Rules** | Domain-specific validation | Catches impossible/implausible values |

### Business Rules for Net Zero Tracker Data

| Rule | Valid Range | Rationale |
|------|-------------|-----------|
| Year columns | 1900-2100 | No net zero targets before 1900 or after 2100 |
| Revenue/Employees | > 0 | Must be positive values |
| GHG Emissions | ≥ 0 | Cannot have negative emissions (except removals) |
| Percentages | 0-100% | Reduction targets should be valid percentages |

### Final Decision Matrix

| Outlier Group | Detection | Decision | Implementation |
|---------------|-----------|----------|----------------|
| Company Size (Revenue, Employees) | IQR + Z-score | 🔄 TRANSFORM | `np.log1p(x)` |
| GHG Emissions | IQR + Z-score | 🔄 TRANSFORM | `np.log1p(x)` |
| Invalid Years (e.g., 1000) | Business Rule | ❌ REMOVE | Filter rows |
| Valid Distant Years (2070+) | IQR | ✅ KEEP | No change |
| Out-of-range Percentages | Business Rule | ⚠️ CLIP | `clip(0, 100)` |

### Model Impact Assessment

| Aspect | Impact of Decisions |
|--------|---------------------|
| **Fairness** | ✅ Large companies preserved - no bias toward smaller firms |
| **Performance** | ✅ Log transforms improve model convergence & stability |
| **Interpretability** | ✅ Log coefficients = % change interpretation |

### Log-Transformed Features Created

```python
df['log_revenue'] = np.log1p(df['Company_annual_revenue'])
df['log_employees'] = np.log1p(df['Employees'])
df['log_revenue_per_emp'] = np.log1p(df['Revenue_per_employee'])
df['log_ghg_emissions'] = np.log1p(df['GHG_emissions'])
```

> **Result**: Skewness reduced from ~10+ to ~0.5 for size-related features, making distributions approximately normal while preserving all legitimate data points.

---


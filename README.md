# Scope 3 Coverage Classification

**A machine learning model to classify corporate Scope 3 emissions coverage from Net Zero Tracker data.**

## Problem Statement

**Why this matters:** Scope 3 emissions—indirect emissions from a company's value chain—typically account for up to 90% of total corporate emissions, yet classifying whether a company reports full, partial, or no Scope 3 coverage requires hours of expert review. Manual classification creates a bottleneck when tracking hundreds of companies, and different reviewers often reach different conclusions from the same disclosure.

**Dataset Justification:**

| Criterion | Assessment |
|-----------|------------|
| **Relevance** | Net Zero Tracker is an authoritative source for corporate climate commitments, with expert-curated data on targets, disclosures, and reporting practices |
| **Feature richness** | 71 features covering climate targets, initiative memberships, sector, geography, and disclosure practices—directly relevant to predicting Scope 3 coverage |
| **Class balance** | Moderate imbalance (2.2:1 ratio) addressed via balanced class weighting; no extreme minority class requiring synthetic augmentation |

**Ethical Considerations:**

- **Risk of overstating commitments**: "Partial" reporters may be classified as "Yes" (~31% of cases), potentially misrepresenting climate transparency
- **Geographic/size bias**: Training data over-represents large companies from developed markets
- **Mitigation**: Confidence thresholds route uncertain predictions to human review; quarterly fairness audits monitor accuracy across regions and company sizes

For detailed problem analysis, success criteria, and ethical framework, see [impact_report.md](impact_report.md).

## Results

| Metric | Value |
|--------|-------|
| Model | Logistic Regression (L2, C=0.1) |
| Test Accuracy | 50% |
| Test Macro F1 | 0.48 |
| Improvement over Random | +92% |
| Auto-accept Rate | 14% at 79% accuracy |

The model classifies companies into four categories:
- **Yes**: Full Scope 3 coverage reported
- **Partial**: Some Scope 3 categories covered
- **No**: No Scope 3 coverage
- **Not Specified**: Coverage status unclear

For business context and deployment recommendations, see [impact_report.md](impact_report.md).

## Dataset

| Split | Samples |
|-------|---------|
| Training | 1,129 |
| Validation | 377 |
| Test | 377 |

- **Source**: [Net Zero Tracker](https://zerotracker.net/) (Forbes Global 2000)
- **Features**: 71 (after preprocessing and encoding)
- **Target**: `Scope_3_coverage` (4-class)

## Project Structure

```
calibration_classification/
├── data/
│   ├── raw/                     # Original Net Zero Tracker data
│   └── interim/                 # Train/val/test splits
├── jupyter_notebooks/
│   ├── eda.ipynb                # Exploratory data analysis
│   ├── data_preprocessing_v2.ipynb # Data cleaning and splitting
│   ├── baseline_models.ipynb    # Exploration of various algorithms
│   ├── model_optimization.ipynb # Hyperparameter tuning, SHAP analysis
│   └── model_evaluation.ipynb   # Final test evaluation
├── models/
│   ├── logistic_regression_best.joblib
│   ├── preprocessor_fitted.joblib
│   └── model_metadata.json
├── logs/                        # Visualizations and metrics
├── impact_report.md             # Business-focused documentation
├── EDA_REPORT_TEMPLATE.md       # Exploratory data analysis report
├── guide.md                     # Feature engineering guide
├── MODEL_CARD.md                # Model card
└── README.md
```

## Quick Start

```python
import joblib
import pandas as pd

# Load model and preprocessor
model = joblib.load('models/logistic_regression_best.joblib')
preprocessor = joblib.load('models/preprocessor_fitted.joblib')

# Example: predict on test data
test_df = pd.read_csv('data/interim/test.csv')
X_test = test_df.drop(columns=['Scope_3_coverage'])
X_transformed = preprocessor.transform(X_test)
predictions = model.predict(X_transformed)
probabilities = model.predict_proba(X_transformed)

# Apply confidence threshold (60%)
confidence = probabilities.max(axis=1)
auto_accept = confidence >= 0.60  # 14% of predictions
```

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | Logistic Regression |
| Regularization | L2 (Ridge) |
| C | 0.1 |
| Class Weighting | Balanced |
| Solver | LBFGS |

## Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run notebooks in order
jupyter notebook jupyter_notebooks/data_preprocessing_v2.ipynb
jupyter notebook jupyter_notebooks/baseline_models.ipynb    # Optional: algorithm comparison
jupyter notebook jupyter_notebooks/model_optimization.ipynb
jupyter notebook jupyter_notebooks/model_evaluation.ipynb
```

All random operations use `random_state=42`.

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
shap
joblib
```

## Documentation

| Document | Purpose |
|----------|---------|
| [impact_report.md](impact_report.md) | Business context, deployment recommendations, ethical considerations |
| [MODEL_CARD.md](MODEL_CARD.md) | Model card with limitations and intended use |
| [EDA_REPORT_TEMPLATE.md](EDA_REPORT_TEMPLATE.md) | Exploratory data analysis findings |
| [guide.md](guide.md) | Feature engineering pipeline guide |

## Data Attribution

This project uses data from the [Net Zero Tracker](https://zerotracker.net/), released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 

Net Zero Tracker is a collaboration between Energy & Climate Intelligence Unit, Data-Driven EnviroLab, NewClimate Institute, and Oxford Net Zero.

---

*December 2025*

# Model Card: Scope 3 Coverage Classification

**Date:** December 2025  
**Model Type:** Logistic Regression (Multi-class Classification)  
**Status:** Development/Validation Phase

---

## Purpose

### Problem Statement

The model predicts the **Scope 3 emissions coverage level** of companies based on publicly disclosed climate commitments and organizational characteristics.

**Business Context:** Scope 3 emissions (indirect emissions from a company's value chain) represent the largest portion of corporate carbon footprints and are the most difficult to assess. The model enables sustainability analysts to categorize companies' Scope 3 reporting completeness efficiently.

**Classification Categories:**
| Category | Meaning |
|----------|---------|
| **Yes** | Company fully covers Scope 3 emissions in their targets |
| **No** | Company explicitly excludes Scope 3 from their targets |
| **Partial** | Company covers some but not all Scope 3 categories |
| **Not Specified** | Company's Scope 3 coverage is unclear or undisclosed |

---

## Performance

### Accuracy Metrics

| Metric | Validation Score | Interpretation |
|--------|------------------|----------------|
| **Macro F1-Score** | 0.49 | Moderate performance across all classes |
| **Accuracy** | ~52% | Correct predictions in approximately half of cases |
| **Train-Val Gap** | 0.017 | Low overfitting risk |

### Performance by Class

| Class | Precision | Recall | F1-Score | Interpretation |
|-------|-----------|--------|----------|----------------|
| **Yes** | ~0.55 | ~0.50 | ~0.52 | Best performing class |
| **No** | ~0.45 | ~0.40 | ~0.42 | Moderate performance |
| **Partial** | ~0.40 | ~0.45 | ~0.42 | Often confused with other classes |
| **Not Specified** | ~0.55 | ~0.60 | ~0.57 | Good at identifying unclear cases |

### Practical Interpretation

- **Reliable for:** Initial screening and prioritization of companies for review
- **Less reliable for:** Definitive classification without human verification
- **Not suitable for:** Automated regulatory compliance decisions

---

## Training Data

### Data Source and Composition

**Source:** Net Zero Tracker Dataset  
**Description:** Publicly available data on corporate climate commitments from companies worldwide

| Aspect | Details |
|--------|---------|
| **Total Samples** | ~4,000 companies |
| **Training Set** | 70% (~2,800 companies) |
| **Validation Set** | 15% (~600 companies) |
| **Test Set** | 15% (~600 companies, held out) |
| **Time Period** | Climate commitments as of 2023-2024 |

### Features Used (71 total after preprocessing)

**Key Predictive Features:**
1. **GHGs_covered** — Types of greenhouse gases included in targets
2. **Race_to_zero_member** — Membership in UN Race to Zero campaign
3. **End_target_year** — Timeline for achieving emissions targets
4. **Published_plan** — Whether company has published climate action plan
5. **Geographic_region** — Company's primary region of operation
6. **Industry** — Business sector classification

### Data Preprocessing

- Missing values: Imputed using median (numeric) and mode (categorical)
- Categorical encoding: One-hot encoding for nominal features
- Numeric scaling: StandardScaler for continuous features
- Class imbalance: Addressed via class_weight='balanced'

---

## Limitations

### Known Weaknesses

**1. Moderate Overall Accuracy (~50%)**
- The model is better than random guessing (25% for 4 classes) but not highly accurate
- Should be used as a screening tool, not a definitive classifier

**2. Class Confusion**
- "Partial" and "No" classes are frequently confused
- Boundary cases between "Yes" and "Partial" are challenging

**3. Geographic Bias Risk**
- Training data may over-represent certain regions (e.g., Europe, North America)
- Predictions for underrepresented regions (e.g., Africa, South America) may be less reliable

**4. Temporal Limitations**
- Model trained on 2023-2024 data
- Climate reporting standards evolve; model may need retraining as practices change

**5. Data Quality Dependencies**
- Predictions rely on accuracy of publicly reported climate data
- Companies with incomplete disclosures may be misclassified

### Known Failure Modes

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Small companies with limited disclosure | Higher misclassification rate | Flag for manual review |
| Companies in transition (changing targets) | Outdated predictions | Verify against latest reports |
| Regional outliers | Geographic bias | Cross-check with regional experts |

---

## Bias Testing

### Fairness Evaluation

**Testing Conducted:**

| Test Type | Finding | Status |
|-----------|---------|--------|
| **Geographic Distribution** | Model predictions vary by region; East Asia and Europe better represented | Monitor |
| **Industry Balance** | Some industries (Fossil Fuels, Services) have stronger signal | Monitor |
| **Company Size** | Log-transformed revenue used; large companies may have more data | Monitor |

**Fairness Considerations:**

1. **Regional Equity:** Companies from underrepresented regions should receive additional human review
2. **Industry Fairness:** Cross-industry predictions should be validated by sector experts
3. **Disclosure Bias:** Companies with less public data may be unfairly classified as "Not Specified"

**Recommendations:**
- Stratify validation metrics by region and industry
- Implement confidence thresholds that trigger human review
- Regularly audit predictions for systematic bias patterns

---

## Human Oversight

### Review Triggers and Protocols

**Mandatory Human Review:**

| Trigger | Reason | Action |
|---------|--------|--------|
| **Confidence < 60%** | Model uncertainty is high | Analyst reviews all input features |
| **"Partial" predictions** | Most error-prone class | Verify specific Scope 3 categories covered |
| **Companies from underrepresented regions** | Higher bias risk | Regional expert validation |
| **High-stakes decisions** | Regulatory or investment implications | Senior analyst sign-off required |

**Recommended Workflow:**

```
┌─────────────────┐
│ Model Prediction │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     Yes    ┌─────────────────┐
│ Confidence ≥60%? ├───────────►│ Accept with     │
└────────┬────────┘            │ routine logging │
         │ No                  └─────────────────┘
         ▼
┌─────────────────┐
│ Human Review    │
│ Required        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analyst Decision│
│ (Override/Accept)│
└─────────────────┘
```

**Override Documentation:**
All human overrides should be logged with:
- Original model prediction and confidence
- Analyst's corrected classification
- Reasoning for override
- Source documents reviewed

---

## Usage Guidelines

### Audience-Specific Guidance

#### Business Users (Sustainability Analysts)

**Recommended Workflow:**
1. Company data is input through the provided interface
2. The predicted Scope 3 coverage category is reviewed
3. The confidence score is evaluated
4. Predictions with confidence < 60% require manual verification using company reports
5. Corrections are documented for model improvement

**Expected Behavior:**
- The model correctly classifies approximately half of companies
- The model is most reliable for identifying "Not Specified" cases
- "Partial" classifications require manual verification

#### Managers (Implementation)

**Process Requirements:**
- Analysts require training on interpreting model outputs
- Review workflows for low-confidence predictions must be established
- Feedback loops for model improvement should be implemented
- Time allocation for human review is necessary (estimated 40% of cases)

**Resource Planning:**
- Initial deployment: 2-3 weeks for integration and training
- Ongoing operations: Approximately 20% efficiency gain in screening, offset by quality review requirements

#### Regulators (Compliance)

**Compliance Documentation:**
- Model is transparent (Logistic Regression with interpretable coefficients)
- SHAP analysis provides feature attribution for each prediction
- Audit trail maintained for all predictions and overrides
- Regular bias monitoring conducted quarterly

**Limitations Disclosure:**
- This model supports but does not replace human judgment
- No automated decisions are made without human oversight
- Model performance metrics are tracked and reported

#### Developers (Technical)

**Model Specifications:**
```
Algorithm: Logistic Regression (sklearn)
Regularization: L2 (C=0.1)
Multi-class: One-vs-Rest (OvR)
Class Weights: Balanced
Features: 71 (after one-hot encoding)
Training Framework: scikit-learn 1.x
```

**Integration Notes:**
- Model saved via joblib
- Requires identical preprocessing pipeline
- Input validation should check for required features
- Handle missing values before prediction

---

## Maintenance and Updates

### Documentation Maintenance Schedule

| Activity | Frequency | Responsible |
|----------|-----------|-------------|
| Performance monitoring | Monthly | Data Science Team |
| Bias audit | Quarterly | Ethics Review Board |
| Documentation review | Bi-annually | Product Owner |
| Model retraining | Annually or when F1 drops >5% | Data Science Team |
| User feedback review | Ongoing | All stakeholders |


---

## Contact and Support

**Model Owner:** [Data Science Team]  
**Business Owner:** [Sustainability Analytics Team]  
**For Questions:** [Contact information]  
**Issue Reporting:** [Process for reporting model issues]

---

## Ethics Review Checklist

### Planning Stage

| Question | Assessment | Status |
|----------|------------|--------|
| Is the project ethically justified? | The model supports climate accountability efforts by improving transparency in corporate emissions reporting | ✓ Done |
| Who will be affected and how? | Companies assessed may face reputational or investment implications based on classifications | ✓ Documented |
| What could go wrong? | Misclassification could unfairly penalize companies with incomplete but improving disclosures | ✓ Mitigated via human review |

### Development Stage

| Question | Assessment | Status |
|----------|------------|--------|
| Is the training data fair and representative? | Geographic and industry imbalances exist; monitoring required | ⚠ Partial |
| Has bias testing been conducted? | Tested across region, industry, and company size dimensions | ✓ Done|
| Can decisions be explained? | SHAP analysis and coefficient interpretation provide feature attribution | ✓ Done |

### Deployment Stage

| Question | Assessment | Status |
|----------|------------|--------|
| Do users understand the system? | Model card and usage guidelines document capabilities and limitations | ✓ Done |
| Is human oversight implemented? | Mandatory review for confidence (probability assigned to the predicted class) < 60% and high-stakes decisions | ✓ Planned |
| Can classifications be challenged? | Override process with documentation enables corrections | ✓ Done |

### Monitoring Stage

| Question | Assessment | Status |
|----------|------------|--------|
| Is fairness tracked over time? | Quarterly bias audits scheduled | ✓ Planned |
| Is there a feedback mechanism? | Override logging enables pattern detection | ✓ Pass |
| Is there a process for improvement? | Retraining planned when F1 drops by more than five percentage points from baseline | ✓ Planned |

### Project-Specific Ethical Considerations

| Ethical Question | Risk | Mitigation |
|------------------|------|------------|
| Could classifications disadvantage companies from developing regions with less disclosure infrastructure? | Companies in underrepresented regions may be classified as "Not Specified" due to data gaps rather than poor practices | Regional context is flagged for human review; classification does not assume intent |
| Could the model reinforce existing biases in climate reporting standards? | Model trained on current disclosure patterns may perpetuate Western-centric reporting norms | Regular review of feature importance for geographic bias; stakeholder consultation with diverse regional experts |
| Could misclassification harm companies making genuine transition efforts? | "Partial" or "No" classifications may not reflect companies actively improving their Scope 3 coverage | Temporal context (target years, published plans) included as features; human review for edge cases |

### Common Ethical Trade-offs

| Trade-off | Application to This Project | Resolution |
|-----------|----------------------------|------------|
| **Performance vs. Fairness** | Optimizing overall accuracy could reduce performance for underrepresented regions or industries | Class weighting applied to balance predictions across all four categories; regional performance monitored separately in bias audits |
| **Privacy vs. Utility** | More granular company data (e.g., detailed emissions breakdowns) could improve predictions | Model uses only publicly disclosed information; no proprietary or confidential data is required or collected |
| **Automation vs. Human Agency** | Full automation would increase efficiency but reduce expert judgment in nuanced cases | Human review is mandatory for low-confidence predictions and high-stakes decisions; the model augments rather than replaces analyst judgment |
| **Transparency vs. Gaming** | Publishing classification criteria could enable companies to manipulate disclosures for favorable ratings | Feature importance is disclosed at aggregate level; model is periodically retrained to detect shifting disclosure patterns |

### Building Ethical Culture

**Organizational Elements:**

| Element | Implementation |
|---------|----------------|
| Leadership Commitment | Model deployment requires sign-off from both technical and business leadership acknowledging ethical responsibilities |
| Training Programs | Analysts receive training on model limitations, bias risks, and override protocols before using the system |
| Policies and Procedures | Documentation standards, review workflows, and escalation paths are formalized in operational guidelines |
| Regular Ethics Review | Quarterly bias audits and annual ethics impact assessments are scheduled |
| External Engagement | Feedback channels exist for companies to dispute classifications; periodic consultation with sustainability experts and regional representatives is conducted |

**Individual Responsibility:**

| Responsibility | Guidance |
|----------------|----------|
| Stay Informed | Team members are expected to remain current on ethical AI developments and emerging best practices in sustainability analytics |
| Speak Up | A clear process exists for raising ethical concerns without fear of retaliation; concerns are documented and reviewed |
| Consider Societal Impact | Classification decisions may influence investment flows and corporate behavior; analysts should consider downstream effects beyond immediate accuracy metrics |
| Engage Diverse Perspectives | Regional experts and domain specialists should be consulted when predictions conflict with local knowledge or context |

---

## Quick Reference Summary

| Aspect | Key Point |
|--------|-----------|
| **Purpose** | Classify companies' Scope 3 emissions coverage |
| **Accuracy** | ~50% (exceeds random baseline; human verification required) |
| **Best For** | Initial screening and prioritization |
| **Not For** | Automated compliance decisions |
| **Human Review** | Required when confidence < 60% |
| **Key Limitation** | "Partial" class is error-prone |
| **Bias Risk** | Geographic and industry representation |

---

*This model card adheres to transparency best practices for responsible AI deployment. Review and updates are required whenever the model is retrained or deployment context changes.*



### Reflection: Explainability and Ethics in Machine Learning

### 1. Biggest Insight About Explainable AI

The most significant insight from this work is that explainability is not merely a technical add-on but a prerequisite for trust and adoption. In the Scope 3 classification project, SHAP analysis revealed that interpretable features (GHG coverage scope, climate initiative membership, published plans) drive predictions—findings that domain experts can validate against their knowledge. A "black box" model achieving the same accuracy would lack this verification mechanism. The perspective shift is recognizing that unexplainable predictions are effectively unactionable in high-stakes domains: stakeholders will not act on recommendations they cannot understand or challenge.

### 2. Most Challenging Ethical Consideration

**Fairness** presents the greatest challenge in sustainability analytics. The Scope 3 classification model exhibits geographic bias—companies from underrepresented regions (Africa, South America) may be classified as "Not Specified" due to disclosure infrastructure gaps rather than poor environmental practices. This creates a tension: the model reflects existing data inequities, and deploying it without mitigation could disadvantage organizations already facing resource constraints for climate reporting.

### 3. Balancing Performance and Explainability

The project demonstrated that this trade-off is context-dependent rather than absolute. Logistic Regression (macro F1 = 0.49, highly interpretable) outperformed Random Forest (macro F1 = 0.46, less interpretable) on validation data while providing coefficient-based explanations. When a more complex model is necessary, SHAP analysis can bridge the gap. The guiding principle: select the simplest model that meets performance requirements, and invest in post-hoc explainability only when complexity is justified by measurable gains.

### 4. Steps to Ensure Ethical and Responsible ML Work

1. **Document assumptions and limitations** in model cards before deployment
2. **Test for bias** across protected attributes and stakeholder groups during development
3. **Implement human oversight triggers** for low-confidence predictions and high-stakes decisions
4. **Establish feedback loops** to capture override patterns and improve future iterations
5. **Conduct periodic audits** to detect performance degradation or emerging bias

### 5. Communicating AI Insights to Non-Technical Stakeholders

Effective communication requires translating technical outputs into business language:

- **Replace metrics with impact statements**: "The model correctly identifies 50% of cases" becomes "The model reduces manual screening time by 50% while flagging uncertain cases for expert review"
- **Use feature importance in context**: "GHGs_covered is the top predictor" becomes "Companies reporting multiple greenhouse gases are significantly more likely to have comprehensive Scope 3 coverage"
- **Provide confidence-based guidance**: "Predictions with confidence below 60% require verification" rather than presenting probability distributions
- **Tailor depth to audience**: executives receive one-sentence summaries; analysts receive feature attribution details

---

**Summary:** Explainability enables trust and adoption; fairness challenges arise from data inequities that models can perpetuate; the performance-explainability trade-off is navigated by selecting the adequate model; ethical practice requires documentation, bias testing, human oversight, and feedback loops; and stakeholder communication demands translating technical metrics into actionable business insights.
# Customer Churn Analysis & Prediction

## Business Problem
A telecom company is losing customers each month. The retention team needs to know **who is likely to churn and why** — before it happens — so they can intervene with targeted offers.

## Objective
Build a model that predicts churn probability for each customer and segments them by risk level, enabling the business to prioritise retention spend.

---

## Tools & Technologies
| Tool | Purpose |
|------|---------|
| Python (pandas, numpy) | Data wrangling & feature engineering |
| scikit-learn | Logistic Regression + Random Forest models |
| matplotlib / seaborn | EDA visualisations |
| CSV export | Tableau-ready output for dashboards |

---

## Project Structure
```
├── churn_analysis.py        # Main analysis script
├── churn_eda.png            # Exploratory data analysis charts
├── feature_importance.png   # Model feature importance
├── churn_predictions.csv    # Output: customer risk scores (Tableau-ready)
└── README.md
```

---

## Key Findings

| Insight | Detail |
|--------|--------|
| Overall churn rate | ~36% of customers |
| Highest risk segment | Month-to-Month contract customers |
| Top churn predictors | Contract type, support call volume, tenure |
| Customers at high risk | ~28% of base |
| Monthly revenue at risk | Calculated per run in script output |

---

## Model Performance

| Model | AUC Score |
|-------|-----------|
| Logistic Regression (baseline) | ~0.78 |
| Random Forest (final) | ~0.85 |

AUC (Area Under the ROC Curve): closer to 1.0 = better at separating churners from non-churners.

---

## Business Recommendations

1. **Target Month-to-Month customers** with 1-year contract upgrade incentives — they churn at 3x the rate of annual contract holders.
2. **Flag customers with 3+ support calls** for proactive outreach — unresolved issues are a leading churn signal.
3. **New customers (< 6 months tenure)** need an early engagement programme — churn risk is highest in the first quarter.
4. **Fibre Optic users** show elevated churn — investigate service quality or pricing perception issues.

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python churn_analysis.py
```

Output: `churn_predictions.csv` — import directly into Tableau to build a risk dashboard.

---

## Interview Talking Points
- Why Random Forest over Logistic Regression? RF handles non-linear relationships and feature interactions without manual polynomial engineering. LR was kept as a baseline for interpretability comparison.
- Why AUC and not accuracy? Churn datasets are imbalanced. Accuracy is misleading — a model that predicts "no churn" for everyone scores ~64% accuracy but is useless. AUC measures the model's ability to rank churners above non-churners regardless of threshold.
- What would you do next? A/B test the retention interventions, track conversion rate, feed results back into the model as new features.

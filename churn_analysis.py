"""
Customer Churn Analysis
=======================
Business Goal: Identify which customers are likely to churn
so retention teams can intervene before losing revenue.

Techniques used:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Logistic Regression (baseline model)
- Random Forest (improved model)
- Business-ready CSV exports for Tableau dashboards
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── 1. GENERATE REALISTIC SYNTHETIC DATASET ──────────────────────────────────
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'customer_id':       range(1001, 1001 + n),
    'tenure_months':     np.random.randint(1, 72, n),
    'monthly_charges':   np.round(np.random.uniform(20, 120, n), 2),
    'total_charges':     np.round(np.random.uniform(100, 8000, n), 2),
    'num_products':      np.random.randint(1, 6, n),
    'support_calls':     np.random.randint(0, 10, n),
    'contract_type':     np.random.choice(['Month-to-Month', 'One Year', 'Two Year'],
                                          n, p=[0.55, 0.25, 0.20]),
    'payment_method':    np.random.choice(['Electronic Check', 'Mailed Check',
                                           'Bank Transfer', 'Credit Card'], n),
    'internet_service':  np.random.choice(['DSL', 'Fiber Optic', 'No'], n, p=[0.4, 0.4, 0.2]),
    'senior_citizen':    np.random.choice([0, 1], n, p=[0.84, 0.16]),
    'gender':            np.random.choice(['Male', 'Female'], n),
})

# Business-logic driven churn: higher churn for M-t-M, high support calls, low tenure
churn_prob = (
    0.05
    + 0.30 * (df['contract_type'] == 'Month-to-Month')
    + 0.02 * df['support_calls']
    - 0.004 * df['tenure_months']
    + 0.10 * (df['internet_service'] == 'Fiber Optic')
    + 0.05 * df['senior_citizen']
)
churn_prob = churn_prob.clip(0.02, 0.90)
df['churn'] = (np.random.rand(n) < churn_prob).astype(int)
df['churn_label'] = df['churn'].map({1: 'Yes', 0: 'No'})

print("=" * 55)
print("CUSTOMER CHURN ANALYSIS — BUSINESS SUMMARY")
print("=" * 55)
print(f"\nTotal Customers Analysed : {len(df):,}")
print(f"Churned Customers        : {df['churn'].sum():,}  ({df['churn'].mean()*100:.1f}%)")
print(f"Retained Customers       : {(df['churn']==0).sum():,}  ({(df['churn']==0).mean()*100:.1f}%)")

# ── 2. EDA ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Customer Churn — Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Churn distribution
churn_counts = df['churn_label'].value_counts()
axes[0,0].bar(churn_counts.index, churn_counts.values, color=['#2ecc71','#e74c3c'])
axes[0,0].set_title('Churn Distribution')
axes[0,0].set_ylabel('Count')
for i, v in enumerate(churn_counts.values):
    axes[0,0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Churn by contract type
ct = df.groupby('contract_type')['churn'].mean().reset_index()
bars = axes[0,1].bar(ct['contract_type'], ct['churn']*100, color=['#3498db','#9b59b6','#1abc9c'])
axes[0,1].set_title('Churn Rate by Contract Type')
axes[0,1].set_ylabel('Churn Rate (%)')
axes[0,1].tick_params(axis='x', rotation=15)
for bar, val in zip(bars, ct['churn']*100):
    axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                   f'{val:.1f}%', ha='center', fontsize=9)

# Tenure distribution by churn
df[df['churn']==0]['tenure_months'].hist(ax=axes[0,2], alpha=0.6, color='#2ecc71',
                                          label='Retained', bins=20)
df[df['churn']==1]['tenure_months'].hist(ax=axes[0,2], alpha=0.6, color='#e74c3c',
                                          label='Churned', bins=20)
axes[0,2].set_title('Tenure Distribution by Churn')
axes[0,2].set_xlabel('Tenure (Months)')
axes[0,2].legend()

# Monthly charges vs churn
df.boxplot(column='monthly_charges', by='churn_label', ax=axes[1,0],
           boxprops=dict(color='#2c3e50'))
axes[1,0].set_title('Monthly Charges vs Churn')
axes[1,0].set_xlabel('Churned?')
plt.sca(axes[1,0])
plt.title('Monthly Charges vs Churn')

# Support calls vs churn
sc = df.groupby('support_calls')['churn'].mean().reset_index()
axes[1,1].plot(sc['support_calls'], sc['churn']*100, marker='o', color='#e67e22', linewidth=2)
axes[1,1].set_title('Support Calls vs Churn Rate')
axes[1,1].set_xlabel('Number of Support Calls')
axes[1,1].set_ylabel('Churn Rate (%)')
axes[1,1].grid(True, alpha=0.3)

# Churn by internet service
isc = df.groupby('internet_service')['churn'].mean().reset_index()
axes[1,2].bar(isc['internet_service'], isc['churn']*100, color=['#16a085','#8e44ad','#d35400'])
axes[1,2].set_title('Churn Rate by Internet Service')
axes[1,2].set_ylabel('Churn Rate (%)')

plt.tight_layout()
plt.savefig('churn_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[✓] EDA chart saved → churn_eda.png")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
df_model = df.copy()
le = LabelEncoder()
for col in ['contract_type', 'payment_method', 'internet_service', 'gender']:
    df_model[col] = le.fit_transform(df_model[col])

df_model['charges_per_month_ratio'] = df_model['total_charges'] / (df_model['tenure_months'] + 1)
df_model['is_new_customer'] = (df_model['tenure_months'] <= 6).astype(int)

features = ['tenure_months', 'monthly_charges', 'total_charges', 'num_products',
            'support_calls', 'contract_type', 'payment_method', 'internet_service',
            'senior_citizen', 'charges_per_month_ratio', 'is_new_customer']

X = df_model[features]
y = df_model['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 4. MODELS ─────────────────────────────────────────────────────────────────
# Logistic Regression (baseline)
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_s, y_train)
lr_pred = lr.predict(X_test_s)
lr_auc  = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:,1])

# Random Forest (improved)
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_auc  = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])

print("\n── MODEL PERFORMANCE ──────────────────────────────")
print(f"Logistic Regression  AUC: {lr_auc:.3f}")
print(f"Random Forest        AUC: {rf_auc:.3f}  ← selected model")
print("\nRandom Forest — Classification Report:")
print(classification_report(y_test, rf_pred, target_names=['Retained','Churned']))

# ── 5. FEATURE IMPORTANCE ─────────────────────────────────────────────────────
fi = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_})
fi = fi.sort_values('importance', ascending=True)

plt.figure(figsize=(9, 6))
plt.barh(fi['feature'], fi['importance'], color='#3498db')
plt.title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Feature importance chart saved → feature_importance.png")

# ── 6. BUSINESS OUTPUT CSV (Tableau-ready) ────────────────────────────────────
df['churn_probability'] = rf.predict_proba(df_model[features])[:,1].round(3)
df['risk_segment'] = pd.cut(df['churn_probability'],
                             bins=[0, 0.30, 0.60, 1.0],
                             labels=['Low Risk', 'Medium Risk', 'High Risk'])

output_cols = ['customer_id', 'tenure_months', 'monthly_charges', 'contract_type',
               'internet_service', 'support_calls', 'senior_citizen',
               'churn_label', 'churn_probability', 'risk_segment']
df[output_cols].to_csv('churn_predictions.csv', index=False)
print("[✓] Tableau-ready CSV saved → churn_predictions.csv")

# ── 7. BUSINESS INSIGHTS ──────────────────────────────────────────────────────
high_risk = df[df['risk_segment'] == 'High Risk']
print("\n── BUSINESS INSIGHTS ──────────────────────────────")
print(f"High-Risk Customers  : {len(high_risk):,} ({len(high_risk)/len(df)*100:.1f}% of base)")
print(f"Avg Monthly Charges  : ${high_risk['monthly_charges'].mean():.2f}")
print(f"Potential MRR at Risk: ${high_risk['monthly_charges'].sum():,.0f}/month")
print(f"\nTop churn driver     : Contract Type (Month-to-Month)")
print(f"Recommendation       : Target high-risk M-t-M customers with 1-year upgrade incentives")
print(f"                       Prioritise customers with 3+ support calls for proactive outreach")
print("\n[✓] Analysis complete.")

"""
Notebook 04: SHAP Interpretability & Business Insights
Explica as predicoes do modelo com SHAP values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sns.set_style("whitegrid")

DATA_PROCESSED = Path('data/processed')
ASSETS = Path('assets')
SRC = Path('src')

print("=" * 80)
print("NOTEBOOK 04: SHAP INTERPRETABILITY & BUSINESS INSIGHTS")
print("=" * 80)

# LOAD DATA
print("\n[1] Loading data...")
df = pd.read_csv(DATA_PROCESSED / 'features_engineered.csv')
y = df['Churn'].map({'No': 0, 'Yes': 1})
X = df.drop(['Churn', 'customerID'], axis=1, errors='ignore')

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# LOAD TUNED MODEL
print("[2] Loading tuned model...")
with open(SRC / 'xgb_model_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

# SHAP EXPLAINER
print("[3] Computing SHAP values (this may take a minute)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("SHAP computation complete")

# SHAP SUMMARY PLOT
print("\n[4] Generating SHAP summary plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(ASSETS / '13_shap_summary_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 13_shap_summary_bar.png")

# SHAP BEESWARM PLOT
print("[5] Generating SHAP beeswarm plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(ASSETS / '14_shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 14_shap_summary_beeswarm.png")

# TOP 5 FEATURES BY SHAP
print("\n[6] Top 5 features by SHAP importance...")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': X_test.columns,
    'shap_importance': mean_abs_shap
}).sort_values('shap_importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# FORCE PLOT FOR HIGH RISK CUSTOMER
print("\n[7] Generating force plots...")
y_pred_proba = model.predict_proba(X_test)[:, 1]

high_risk_idx = np.argmax(y_pred_proba)
low_risk_idx = np.argmin(y_pred_proba)

plt.figure(figsize=(14, 4))
shap.force_plot(explainer.expected_value, shap_values[high_risk_idx], X_test.iloc[high_risk_idx], matplotlib=True, show=False)
plt.tight_layout()
plt.savefig(ASSETS / '15_force_plot_high_risk.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 15_force_plot_high_risk.png (High Risk Customer)")

plt.figure(figsize=(14, 4))
shap.force_plot(explainer.expected_value, shap_values[low_risk_idx], X_test.iloc[low_risk_idx], matplotlib=True, show=False)
plt.tight_layout()
plt.savefig(ASSETS / '16_force_plot_low_risk.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 16_force_plot_low_risk.png (Low Risk Customer)")

# DEPENDENCE PLOTS FOR TOP 3 FEATURES
print("\n[8] Generating dependence plots...")
top_3_features = feature_importance.head(3)['feature'].tolist()

for i, feature in enumerate(top_3_features):
    feature_idx = X_test.columns.get_loc(feature)
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature_idx, shap_values, X_test, feature_names=X_test.columns, show=False)
    plt.tight_layout()
    plt.savefig(ASSETS / f'17_dependence_{i+1}_{feature}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: 17_dependence_{i+1}_{feature}.png")

# RISK SEGMENTATION
print("\n[9] Risk segmentation analysis...")
risk_segments = pd.DataFrame({
    'churn_probability': y_pred_proba,
    'actual_churn': y_test.values
})

risk_segments['risk_segment'] = pd.cut(risk_segments['churn_probability'], 
                                       bins=[0, 0.3, 0.5, 0.7, 1.0],
                                       labels=['Low', 'Medium', 'High', 'Very High'])

segment_analysis = risk_segments.groupby('risk_segment').agg({
    'churn_probability': ['count', 'mean'],
    'actual_churn': ['sum', 'mean']
}).round(4)

print("\nChurn Risk Segments:")
print(segment_analysis)

# RECOMMENDED ACTIONS
print("\n[10] Business recommendations...")

recommendations = """
BUSINESS INSIGHTS & RECOMMENDED ACTIONS
======================================

Risk Segment: LOW (Churn Prob < 30%)
  Count: {low_count}
  Actual Churn Rate: {low_churn:.1%}
  Action: Upsell & Cross-sell opportunities

Risk Segment: MEDIUM (Churn Prob 30-50%)
  Count: {med_count}
  Actual Churn Rate: {med_churn:.1%}
  Action: Retention outreach - discounts or service upgrades

Risk Segment: HIGH (Churn Prob 50-70%)
  Count: {high_count}
  Actual Churn Rate: {high_churn:.1%}
  Action: Immediate intervention - retention team calls

Risk Segment: VERY HIGH (Churn Prob > 70%)
  Count: {vhigh_count}
  Actual Churn Rate: {vhigh_churn:.1%}
  Action: Executive outreach - custom retention offer

TOP FACTORS DRIVING CHURN (SHAP):
  1. {f1}: {v1:.4f}
  2. {f2}: {v2:.4f}
  3. {f3}: {v3:.4f}
""".format(
    low_count=len(risk_segments[risk_segments['risk_segment'] == 'Low']),
    low_churn=risk_segments[risk_segments['risk_segment'] == 'Low']['actual_churn'].mean(),
    med_count=len(risk_segments[risk_segments['risk_segment'] == 'Medium']),
    med_churn=risk_segments[risk_segments['risk_segment'] == 'Medium']['actual_churn'].mean(),
    high_count=len(risk_segments[risk_segments['risk_segment'] == 'High']),
    high_churn=risk_segments[risk_segments['risk_segment'] == 'High']['actual_churn'].mean(),
    vhigh_count=len(risk_segments[risk_segments['risk_segment'] == 'Very High']),
    vhigh_churn=risk_segments[risk_segments['risk_segment'] == 'Very High']['actual_churn'].mean(),
    f1=feature_importance.iloc[0]['feature'],
    v1=feature_importance.iloc[0]['shap_importance'],
    f2=feature_importance.iloc[1]['feature'],
    v2=feature_importance.iloc[1]['shap_importance'],
    f3=feature_importance.iloc[2]['feature'],
    v3=feature_importance.iloc[2]['shap_importance']
)

print(recommendations)

print("\n" + "=" * 80)
print("NOTEBOOK 04 COMPLETE!")
print("PROJECT COMPLETE - ALL 4 NOTEBOOKS FINISHED")
print("=" * 80)

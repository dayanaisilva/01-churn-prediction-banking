"""
Notebook 02: Model Training & Comparison
Treina 3 modelos (Logistic Regression, Random Forest, XGBoost) e compara performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

DATA_PROCESSED = Path('data/processed')
ASSETS = Path('assets')
SRC = Path('src')

print("=" * 80)
print("NOTEBOOK 02: MODEL TRAINING & COMPARISON")
print("=" * 80)

# LOAD DATA
print("\n[1] Loading processed data...")
df = pd.read_csv(DATA_PROCESSED / 'features_engineered.csv')
print(f"Loaded: {df.shape}")

y = df['Churn'].map({'No': 0, 'Yes': 1})
X = df.drop(['Churn', 'customerID'], axis=1, errors='ignore')

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print(f"Features: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# MODEL 1: LOGISTIC REGRESSION
print("\n" + "=" * 80)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 80)

lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
f1_lr = f1_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)

print(f"AUC: {auc_lr:.3f}, F1: {f1_lr:.3f}, Precision: {precision_lr:.3f}, Recall: {recall_lr:.3f}")

# MODEL 2: RANDOM FOREST
print("\n" + "=" * 80)
print("MODEL 2: RANDOM FOREST")
print("=" * 80)

rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=20,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
f1_rf = f1_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)

print(f"AUC: {auc_rf:.3f}, F1: {f1_rf:.3f}, Precision: {precision_rf:.3f}, Recall: {recall_rf:.3f}")

# MODEL 3: XGBOOST
print("\n" + "=" * 80)
print("MODEL 3: XGBOOST")
print("=" * 80)

xgb_model = XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    random_state=42, n_jobs=-1, verbose=0
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)

print(f"AUC: {auc_xgb:.3f}, F1: {f1_xgb:.3f}, Precision: {precision_xgb:.3f}, Recall: {recall_xgb:.3f}")

# COMPARISON
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'AUC': [auc_lr, auc_rf, auc_xgb],
    'F1': [f1_lr, f1_rf, f1_xgb],
    'Precision': [precision_lr, precision_rf, precision_xgb],
    'Recall': [recall_lr, recall_rf, recall_xgb]
})

print("\n" + comparison_df.to_string(index=False))
best_model_idx = comparison_df['AUC'].idxmax()
print(f"\nBest: {comparison_df.loc[best_model_idx, 'Model']} (AUC: {comparison_df.loc[best_model_idx, 'AUC']:.3f})")
comparison_df.to_csv(ASSETS / 'model_comparison.csv', index=False)

# ROC CURVES
print("\n[2] Generating ROC curves...")
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

models = [
    ('Logistic Regression', y_pred_proba_lr, auc_lr, axes[0]),
    ('Random Forest', y_pred_proba_rf, auc_rf, axes[1]),
    ('XGBoost', y_pred_proba_xgb, auc_xgb, axes[2])
]

for name, y_proba, auc_score, ax in models:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, label=f'ROC (AUC={auc_score:.3f})', linewidth=2.5, color='#3498db')
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{name}\nAUC={auc_score:.3f}', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(ASSETS / '06_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: assets/06_roc_curves.png")

# CONFUSION MATRICES
print("[3] Generating confusion matrices...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

models_conf = [
    ('Logistic Regression', y_pred_lr, axes[0]),
    ('Random Forest', y_pred_rf, axes[1]),
    ('XGBoost', y_pred_xgb, axes[2])
]

for name, y_pred, ax in models_conf:
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Retained', 'Churned'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(name, fontweight='bold')

plt.tight_layout()
plt.savefig(ASSETS / '07_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: assets/07_confusion_matrices.png")

# FEATURE IMPORTANCE
print("[4] Feature importance...")
rf_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

xgb_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

top_15_rf = rf_importance.head(15)
axes[0].barh(range(len(top_15_rf)), top_15_rf['importance'].values, color='#2ecc71')
axes[0].set_yticks(range(len(top_15_rf)))
axes[0].set_yticklabels(top_15_rf['feature'].values)
axes[0].set_title('Random Forest: Top 15', fontweight='bold')
axes[0].invert_yaxis()

top_15_xgb = xgb_importance.head(15)
axes[1].barh(range(len(top_15_xgb)), top_15_xgb['importance'].values, color='#e74c3c')
axes[1].set_yticks(range(len(top_15_xgb)))
axes[1].set_yticklabels(top_15_xgb['feature'].values)
axes[1].set_title('XGBoost: Top 15', fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(ASSETS / '08_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: assets/08_feature_importance.png")

print(f"\nXGBoost Top 10:")
print(xgb_importance.head(10).to_string(index=False))

# CROSS-VALIDATION
print("\n[5] Cross-validation (5-fold)...")
cv_scores = cross_validate(
    xgb_model, X_train, y_train, cv=5,
    scoring=['roc_auc', 'f1', 'precision', 'recall'],
    n_jobs=-1
)

print(f"AUC: {cv_scores['test_roc_auc'].mean():.3f} +/- {cv_scores['test_roc_auc'].std():.3f}")
print(f"F1: {cv_scores['test_f1'].mean():.3f} +/- {cv_scores['test_f1'].std():.3f}")

# SAVE MODELS
print("\n[6] Saving models...")
with open(SRC / 'lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
with open(SRC / 'rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open(SRC / 'xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
with open(SRC / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Models saved")

# SUMMARY
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Best Model: XGBoost
  Test AUC: {auc_xgb:.3f}
  CV AUC: {cv_scores['test_roc_auc'].mean():.3f} +/- {cv_scores['test_roc_auc'].std():.3f}

Ranking:
  1. XGBoost: {auc_xgb:.3f}
  2. Random Forest: {auc_rf:.3f}
  3. Logistic Regression: {auc_lr:.3f}

Next: Notebook 03 - Hyperparameter Tuning
NOTEBOOK 02 COMPLETE!
""")
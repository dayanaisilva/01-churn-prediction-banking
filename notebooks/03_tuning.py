import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

sns.set_style("whitegrid")
DATA_PROCESSED = Path('data/processed')
ASSETS = Path('assets')
SRC = Path('src')

print("NOTEBOOK 03: TUNING")

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

print("GridSearchCV...")
param_grid = {'max_depth': [6, 8], 'learning_rate': [0.1, 0.2], 'gamma': [0, 1]}
xgb = XGBClassifier(n_estimators=150, subsample=0.8, random_state=42, n_jobs=-1)
grid = GridSearchCV(xgb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(f"Best: {grid.best_params_}, AUC: {grid.best_score_:.4f}")

best_xgb = grid.best_estimator_
y_pred = best_xgb.predict(X_test)
y_proba = best_xgb.predict_proba(X_test)[:, 1]
auc_new = roc_auc_score(y_test, y_proba)

print(f"Test AUC: {auc_new:.4f}")

cv = cross_validate(best_xgb, X_train, y_train, cv=5, scoring=['roc_auc', 'f1'], n_jobs=-1)
print(f"CV AUC: {cv['test_roc_auc'].mean():.4f} +/- {cv['test_roc_auc'].std():.4f}")

with open(SRC / 'xgb_model_tuned.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)

with open(SRC / 'xgb_model.pkl', 'rb') as f:
    old = pickle.load(f)

old_auc = roc_auc_score(y_test, old.predict_proba(X_test)[:, 1])
print(f"\nBefore: {old_auc:.4f}")
print(f"After:  {auc_new:.4f}")
print(f"Gain:   {(auc_new-old_auc):.4f}")

print("\nNOTEBOOK 03 COMPLETE!")
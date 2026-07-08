"""
Notebook 01: Exploratory Data Analysis (EDA)
Explore dataset, understand features, create engineered features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

DATA_RAW = Path('data/raw')
DATA_PROCESSED = Path('data/processed')
ASSETS = Path('assets')

print("=" * 80)
print("NOTEBOOK 01: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# LOAD DATA
print("\n[1] Loading dataset...")
df = pd.read_csv(DATA_RAW / 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Churn distribution
churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100

print(f"\nChurn distribution:")
print(f"  No:  {churn_counts['No']:,} ({churn_pct['No']:.1f}%)")
print(f"  Yes: {churn_counts['Yes']:,} ({churn_pct['Yes']:.1f}%)")

# Visualize churn
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2ecc71', '#e74c3c']
ax.bar(churn_counts.index, churn_counts.values, color=colors, edgecolor='black', width=0.6)
ax.set_title('Churn Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Count')
for i, v in enumerate(churn_counts.values):
    ax.text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(ASSETS / '01_churn_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: assets/01_churn_distribution.png")

# Tenure analysis
churn_no = df[df['Churn'] == 'No']['tenure']
churn_yes = df[df['Churn'] == 'Yes']['tenure']

fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot([churn_no, churn_yes], labels=['No Churn', 'Churn'], patch_artist=True)
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
ax.set_title('Tenure (months) vs Churn', fontsize=14, fontweight='bold')
ax.set_ylabel('Tenure (months)')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(ASSETS / '02_tenure_vs_churn.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: assets/02_tenure_vs_churn.png")

print(f"Tenure Stats:")
print(f"  No Churn:  Mean {churn_no.mean():.1f} months")
print(f"  Churn:     Mean {churn_yes.mean():.1f} months")

# Contract analysis
churn_by_contract = df.groupby('Contract')['Churn'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors_bar = ['#e74c3c', '#f39c12', '#2ecc71']
churn_by_contract.plot(kind='bar', ax=ax, color=colors_bar, edgecolor='black')
ax.set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Churn Rate (%)')
ax.set_xlabel('Contract Type')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(ASSETS / '03_contract_vs_churn.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: assets/03_contract_vs_churn.png")

print(f"\nChurn Rate by Contract:")
print(churn_by_contract)

# FEATURE ENGINEERING
print("\n[2] Feature engineering...")
df_processed = df.copy()

# Service count
service_cols = ['OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df_processed['service_count'] = (df_processed[service_cols] == 'Yes').sum(axis=1)

# Tenure cohort
df_processed['tenure_cohort'] = pd.cut(df_processed['tenure'], 
                                       bins=[0, 1, 3, 6, 12, 24, 73],
                                       labels=['0-1mo', '1-3mo', '3-6mo', '6-12mo', '12-24mo', '24mo+'])

# Contract risk
df_processed['contract_risk'] = (
    (df_processed['Contract'] == 'Month-to-month').astype(int) * 3 +
    (df_processed['InternetService'] == 'Fiber optic').astype(int) * 1.5
)

print("Features created: service_count, tenure_cohort, contract_risk")

# Service count analysis
churn_by_services = df_processed.groupby('service_count')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
)

fig, ax = plt.subplots(figsize=(10, 6))
churn_by_services.plot(kind='bar', ax=ax, color='#3498db', edgecolor='black')
ax.set_title('Churn Rate by Number of Services', fontsize=14, fontweight='bold')
ax.set_ylabel('Churn Rate (%)')
ax.set_xlabel('Number of Services')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(ASSETS / '04_services_vs_churn.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: assets/04_services_vs_churn.png")

print(f"\nChurn Rate by Service Count:")
print(churn_by_services)

# SAVE
print("\n[3] Saving processed data...")
df_processed.to_csv(DATA_PROCESSED / 'features_engineered.csv', index=False)
print(f"Saved: data/processed/features_engineered.csv")
print(f"Shape: {df_processed.shape}")

print("\n" + "=" * 80)
print("NOTEBOOK 01 COMPLETE!")
print("=" * 80)
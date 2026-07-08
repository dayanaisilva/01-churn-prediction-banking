import pandas as pd

df = pd.read_csv('data/processed/features_engineered.csv')
print(f"Shape: {df.shape}")
print(f"\nNaN values:")
print(df.isnull().sum())
print(f"\nChurn distribution:")
print(df['Churn'].value_counts())
print(f"\nFirst few rows:")
print(df.head())
print(f"\nColumns:")
print(df.columns.tolist())

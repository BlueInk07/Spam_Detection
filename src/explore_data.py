import pandas as pd

# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Remove useless columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nClass Distribution:")
print(df['label'].value_counts())
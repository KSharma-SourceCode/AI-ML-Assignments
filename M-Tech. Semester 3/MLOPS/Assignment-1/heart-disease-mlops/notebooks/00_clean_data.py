import pandas as pd
import os
import urllib.request

os.makedirs("../data/raw", exist_ok=True)
os.makedirs("../data/processed", exist_ok=True)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
raw_path = "../data/raw/heart.csv"

urllib.request.urlretrieve(url, raw_path)
print("Downloaded dataset to", raw_path)

column_names = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

df = pd.read_csv(raw_path, names=column_names)
df.head()

# Replace '?' with NaN
df = df.replace("?", pd.NA)

# Convert columns to numeric
df = df.apply(pd.to_numeric)

print(df.isna().sum())

df_clean = df.dropna()
print("Before:", df.shape)
print("After:", df_clean.shape)

# Convert target to binary (0 = no disease, 1 = disease)
df_clean["target"] = df_clean["target"].apply(lambda x: 0 if x == 0 else 1)

df_clean["target"].value_counts()

clean_path = "../data/processed/heart_clean.csv"
df_clean.to_csv(clean_path, index=False)

print("Cleaned dataset saved to", clean_path)


df.to_csv("data/processed/heart_clean.csv", index=False)
print("Saved cleaned dataset to data/processed/heart_clean.csv")

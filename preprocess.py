import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

print("Loading raw dataset...")

df = pd.read_csv("data/raw/housing.csv")

print(f"Original Shape: {df.shape}")

df = df.dropna()

print(f"Processed Shape: {df.shape}")

df.to_csv("data/processed/clean_housing.csv", index=False)

print("Processed dataset saved.")
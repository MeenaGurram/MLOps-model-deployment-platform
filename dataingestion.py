from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

print("Downloading dataset...")

housing = fetch_california_housing(as_frame=True)

df = housing.frame

df.to_csv("data/raw/housing.csv", index=False)

print("Dataset saved to data/raw/housing.csv")
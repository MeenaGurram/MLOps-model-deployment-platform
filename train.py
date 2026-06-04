import os
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

os.makedirs("models", exist_ok=True)

print("Loading processed data...")

df = pd.read_csv(
    "data/processed/clean_housing.csv"
)

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

mlflow.set_experiment(
    "Housing_Price_Prediction"
)

with mlflow.start_run():

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "max_depth",
        10
    )

    mlflow.log_metric(
        "rmse",
        rmse
    )

    mlflow.log_metric(
        "r2_score",
        r2
    )

    mlflow.sklearn.log_model(
        model,
        "housing_model"
    )

    joblib.dump(
        model,
        "models/model.pkl"
    )

    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

print("Model saved successfully.")

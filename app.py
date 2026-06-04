from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="MLOps Model API")

# Load model
MODEL_PATH = "models/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise Exception("Model not found. Run src/train.py first.")

model = joblib.load(MODEL_PATH)


# Request schema
class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def home():
    return {"message": "MLOps API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: InputData):

    input_data = np.array([[
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]])

    prediction = model.predict(input_data)[0]

    return {
        "prediction": float(prediction)
    }
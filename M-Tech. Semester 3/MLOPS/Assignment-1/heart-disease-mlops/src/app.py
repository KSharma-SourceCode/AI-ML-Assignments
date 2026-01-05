from fastapi import FastAPI
from pydantic import BaseModel
import joblib,os
import pandas as pd

app = FastAPI(title="Heart Disease Prediction API")

# -------------------------
# Model loading
# -------------------------

def get_latest_model_path(base_dir="model"):
    versions = [
        int(d.replace("v", ""))
        for d in os.listdir(base_dir)
        if d.startswith("v")
    ]
    latest_version = max(versions)
    return f"{base_dir}/v{latest_version}/heart_model.pkl"


MODEL_PATH = get_latest_model_path()
model = joblib.load(MODEL_PATH)

print(f"✅ Loaded model from {MODEL_PATH}")

# -------------------------
# Input schema
# -------------------------

class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# -------------------------
# Routes
# -------------------------

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: PatientData):

    input_df = pd.DataFrame([{
        "age": data.age,
        "sex": data.sex,
        "cp": data.cp,
        "trestbps": data.trestbps,
        "chol": data.chol,
        "fbs": data.fbs,
        "restecg": data.restecg,
        "thalach": data.thalach,
        "exang": data.exang,
        "oldpeak": data.oldpeak,
        "slope": data.slope,
        "ca": data.ca,
        "thal": data.thal
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "heart_disease": int(prediction),
        "confidence": round(float(probability), 4)
    }

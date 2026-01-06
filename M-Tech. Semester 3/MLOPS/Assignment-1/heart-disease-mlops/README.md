# ❤️ Heart Disease Prediction – End-to-End MLOps Pipeline

## 📌 Overview

This project implements an **end-to-end MLOps pipeline** to predict the presence of heart disease using the **UCI Heart Disease dataset**.  
The solution demonstrates the complete lifecycle of a production-grade ML system including preprocessing, training, experiment tracking, model versioning, and API-based inference.

---

## 🧠 Problem Statement

Predict whether a patient has heart disease based on clinical and diagnostic attributes using a binary classification model.

---

## 📊 Dataset

- **Name:** UCI Heart Disease Dataset (Cleveland subset)  
- **Records:** 303 (after cleaning: 297)  
- **Features:** 14  
- **Target:** Heart disease (0 = No, 1 = Yes)

---

## 🏗️ Project Structure

```
heart-disease-mlops/
├── data/
│   └── raw/
│       └── heart.csv
│   └── processed/
│       └── heart_clean.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── app.py
├── notebooks/
│   ├── 00_clean_data.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   └── 03_inference.ipynb
├── tests/
│   ├── test_preprocess.py
│   └── test_train.py
├── model/
│   └── development/
│               └── v1/  
│                   └── heart_model.pkl
│   └── production/
│               └── v1/  
│                   └── heart_model.pkl
├── mlruns/
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## ⚙️ Setup Instructions

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🧪 Testing

Run all unit tests:

```bash
pytest
```

---

## 🤖 Model Training

Train the model and log experiments using MLflow:

```bash
python -m src.train
```

---

## 📊 MLflow Experiment Tracking

Start MLflow UI:

```bash
mlflow ui
```

Access at: http://localhost:5000

---

## 🧱 Model Versioning

Each training run creates a new model version:

```
model/
├── v1/
├── v2/
```

Only one version is used for production inference.

---

## 🌐 FastAPI Inference Service

Start the API:

```bash
uvicorn src.app:app --reload
```

Swagger UI: http://127.0.0.1:8000/docs

---

## 🔌 API Contract

### Endpoint
POST `/predict`

### Request
```json
{
  "age": 63,
  "sex": 1,
  "cp": 1,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 2,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 3,
  "ca": 0,
  "thal": 6
}
```

### Response
```json
{
  "heart_disease": 0,
  "confidence": 0.3271
}
```

---

## 📈 Key MLOps Concepts

- Reproducible pipelines
- Unit testing
- Experiment tracking
- Model versioning
- Training–serving consistency
- API-based inference

---

## 🏁 Conclusion

This project demonstrates a complete MLOps workflow from raw data to a deployable ML service, emphasizing reproducibility, reliability, and production readiness.

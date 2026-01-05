import joblib
import pandas as pd

model = joblib.load("../model/development/v1/heart_model.pkl")

sample = pd.DataFrame([{
    "age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,
    "restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,
    "slope":3,"ca":0,"thal":6
}])

model.predict(sample), model.predict_proba(sample)

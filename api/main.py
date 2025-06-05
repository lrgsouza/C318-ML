# /api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import numpy as np

# 1. Carregar o modelo e SHAP explainer
model = joblib.load("modelo/model_pipeline.pkl")
explainer = joblib.load("modelo/shap_explainer.pkl")
feature_order = joblib.load("modelo/feature_order.pkl")

# 2. Inicializar FastAPI
app = FastAPI(
    title="API de Previsão de Doença Cardíaca",
    description="Modelo treinado com SHAP e Random Forest",
    version="1.0.0"
)

# 3. Definir o schema de entrada com Pydantic
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# 4. Função auxiliar para converter entrada para DataFrame ordenado
def preprocess_input(data: PatientData) -> pd.DataFrame:
    df = pd.DataFrame([data.dict()])
    df = df[feature_order]
    return df

# 5. Rota de predição
@app.post("/predict")
def predict_disease(data: PatientData):
    df = preprocess_input(data)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return {
        "risco_doenca": int(prediction),
        "probabilidade": round(float(probability), 3)
    }

# 6. Rota de explicabilidade com SHAP
@app.post("/explain")
def explain_prediction(data: PatientData):
    df = preprocess_input(data)
    transformed = model.named_steps["preprocessor"].transform(df)
    shap_vals = explainer(transformed)
    shap_dict = {
        "base_value": float(shap_vals.base_values[0]),
        "shap_values": dict(zip(feature_order, map(float, shap_vals.values[0])))
    }
    return shap_dict

# 7. Rota de saúde da API
@app.get("/health")
def health_check():
    return {"status": "API is running"}
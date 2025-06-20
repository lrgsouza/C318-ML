# /api/main.py
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://heart-disease-prediction-psi-ecru.vercel.app", "http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    # Gerar as features derivadas
    df['age_chol_interaction'] = df['age'] * df['chol']
    max_oldpeak = max(2.1, df['oldpeak'].max())
    df['oldpeak_risk_group'] = pd.cut(
        df['oldpeak'],
        bins=[-0.1, 1.0, 2.0, max_oldpeak],
        labels=['Low', 'Medium', 'High'],
        right=True,
        include_lowest=True
    )

    # Garantir que todas as colunas do modelo estejam presentes
    missing = set(feature_order) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no input: {missing}")

    # Reordenar as colunas conforme esperado pelo modelo
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
        "base_value": float(np.ravel(shap_vals.base_values)[0]),
        "shap_values": {
            feature: float(np.ravel(val)[0])
            for feature, val in zip(feature_order, shap_vals.values[0])
        }
    }
    return shap_dict


# 7. Rota de saúde da API
@app.get("/health")
def health_check():
    return {"status": "API is running"}
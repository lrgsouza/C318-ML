# /modelo/train_model.py

import pandas as pd
import joblib
import kagglehub
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap


import os
import json

# Criar kaggle.json a partir da variável de ambiente
kaggle_token = os.getenv("KAGGLE_TOKEN_JSON")

if kaggle_token:
    os.makedirs("/root/.kaggle", exist_ok=True)
    with open("/root/.kaggle/kaggle.json", "w") as f:
        f.write(kaggle_token)

# 1. Baixar dataset do Kaggle usando kagglehub
dataset_path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
csv_path = os.path.join(dataset_path, "heart.csv")
df = pd.read_csv(csv_path)

# 2. Separar features e target
X = df.drop("target", axis=1)
y = df["target"]

# 3. Separar numéricas e categóricas
numerical_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_cols = list(set(X.columns) - set(numerical_cols))

# 4. Pipeline de pré-processamento
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

# 5. Pipeline final
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 6. Treino e avaliação
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))

# 7. SHAP
X_train_transformed_for_shap = pipeline.steps[0][1].transform(X_train) # Acessa o pré-processador e o transforma os dados de treino
explainer = shap.Explainer(pipeline.named_steps["classifier"], X_train_transformed_for_shap)
shap_values = explainer(X_train_transformed_for_shap)

# 8. Salvar artefatos
os.makedirs("modelo", exist_ok=True)
joblib.dump(pipeline, "modelo/model_pipeline.pkl")
joblib.dump(explainer, "modelo/shap_explainer.pkl")
joblib.dump(X.columns.tolist(), "modelo/feature_order.pkl")
print("Modelo e explainer salvos!")

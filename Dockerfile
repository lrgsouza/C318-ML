FROM python:3.10-slim

WORKDIR /app

COPY modelo/train_model.py modelo/train_model.py
COPY api/main.py api/main.py

RUN pip install --no-cache-dir pandas numpy scikit-learn joblib fastapi uvicorn shap kagglehub

# Executa o treinamento (kagglehub vai usar a variável de ambiente)
RUN python modelo/train_model.py

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]

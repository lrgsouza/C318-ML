# 🧠 Modelo de Machine Learning

## 🔄 Pipeline de Treinamento

1. Download do dataset (Kaggle ou local)
2. Engenharia de features:
   - `age_chol_interaction`: idade × colesterol
   - `oldpeak_risk_group`: Low / Medium / High
3. Pré-processamento:
   - Imputação de dados
   - One-Hot Encoding
   - Escalonamento numérico
4. Modelagem:
   - Gaussian Naive Bayes
   - Regressão Logística (com GridSearchCV)
   - Random Forest
5. Validação cruzada (5 folds)
6. Salvamento do melhor modelo

## 📊 Métricas

| Modelo               | Acurácia | Precisão | Recall |
|----------------------|----------|----------|--------|
| ⭐ Random Forest      | **0.92** | **0.91** | **0.93** |
| Regressão Logística  | 0.88     | 0.87     | 0.89   |
| Naive Bayes          | 0.83     | 0.82     | 0.84   |

# 🌐 Documentação da API

- [API de Previsão de Doença Cardíaca](https://c318-ml-1010639301046.europe-west1.run.app/docs)

## 🔗 Endpoints

### `POST /predict`

- Faz predição de risco.

**Request:**
```json
{
  "age": 45,
  "sex": 1,
  "cp": 2,
  "trestbps": 130,
  "chol": 240,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.2,
  "slope": 1,
  "ca": 0,
  "thal": 2
}
```

**Response:**
```json
{
  "risco_doenca": 1,
  "probabilidade": 0.872
}
```

---

### `POST /explain`

- Retorna explicação SHAP da predição.

### `GET /health`

- Verifica o status da API.

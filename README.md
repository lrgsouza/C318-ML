# 🫀 Previsão de Doença Cardíaca com ML + SHAP + FastAPI

Este projeto é um exemplo completo de aplicação de Machine Learning com foco em saúde. Ele prevê o risco de doença cardíaca com base em dados clínicos e explica as predições usando SHAP. A aplicação é disponibilizada por uma API FastAPI e pode ser consumida via formulário web.

---

## 📁 Estrutura do Projeto

```
.
├── modelo/
│   ├── train_model.py           # Script de treinamento do modelo
│   └── (gera arquivos .pkl com modelo e SHAP)
├── api/
│   └── main.py                  # API com FastAPI que serve o modelo
├── frontend/
│   └── index.html               # Formulário que consome a API
│   └── index_local.html         # Formulário que consome a API localmente
├── Dockerfile                   # Dockerfile unificado: treina + sobe API
├── requirements.txt             # Bibliotecas necessárias
```

---

## 🧪 Como rodar localmente

### 1. Clone o projeto

```bash
git clone https://github.com/lrgsouza/C318-ML.git
cd C318-ML
```

### 2. Rodando localmente com docker compose

```bash
docker-compose up --build
```

### 3. Acesse a API
Acesse a documentação da API em: [http://localhost:8080/docs](http://localhost:8080/docs)

### 4. Acesse o frontend
Acesse o formulário em: [http://localhost:80](http://localhost:80)

---

## 📦 Dependências (caso queira rodar localmente sem Docker)

```bash
pip install -r requirements.txt
```

## 🐍 Como rodar com python local

```bash
python modelo/train_model.py  # Treina o modelo e gera os arquivos .pkl
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

---

## ☁️ Deploy na nuvem (resumo)

- Google Cloud Run para a API
- Vercel para o Frontend (consumindo API pública)

---

## 📌 Créditos

- Dataset: [Heart Disease - Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- ML Explainability: [SHAP](https://github.com/shap/shap)
- API: [FastAPI](https://fastapi.tiangolo.com)

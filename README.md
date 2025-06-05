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

### 2. Defina a variável de ambiente com seu token do Kaggle

Obtenha seu token em https://www.kaggle.com/settings

```bash
export KAGGLE_TOKEN_JSON='{"username":"seunome","key":"seutoken"}'
```

### 3. Construa a imagem Docker

```bash
docker build -t modelo-api-unificada .
```

### 4. Rode a API localmente

```bash
docker run -p 8080:8080 -e KAGGLE_TOKEN_JSON="$KAGGLE_TOKEN_JSON" modelo-api-unificada
```

Acesse a documentação da API em: [http://localhost:8080/docs](http://localhost:8080/docs)

---

## 🌐 Como usar o frontend

1. Vá até a pasta `frontend/`  
2. Abra o arquivo `index_local.html` no navegador  
3. Preencha os dados clínicos do paciente  
4. Veja o resultado da predição e explicabilidade (SHAP)

---

## 📦 Dependências (caso queira rodar localmente sem Docker)

```bash
pip install -r requirements.txt
```

---

## ☁️ Deploy na nuvem (resumo)

- Google Cloud Run para a API
- Vercel para o Frontend (consumindo API pública)

---

## 🔒 Segurança

- O token do Kaggle é **injetado como variável de ambiente** (sem persistir em disco)
- O modelo e explicador SHAP são treinados no container e servidos imediatamente

---

## 📌 Créditos

- Dataset: [Heart Disease - Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- ML Explainability: [SHAP](https://github.com/shap/shap)
- API: [FastAPI](https://fastapi.tiangolo.com)

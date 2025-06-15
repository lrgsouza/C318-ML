# 🚀 Guia de Instalação

## ✅ Pré-requisitos

- Docker e Docker Compose  
**ou**  
- Python 3.8+

## 🔥 Com Docker (Recomendado)

```bash
git clone https://github.com/lrgsouza/C318-ML.git
cd C318-ML
docker-compose up --build
```

Acesse:  
- 🖥 Frontend: http://localhost  
- 🔗 API Docs: http://localhost:8080/docs  

## ⚙️ Sem Docker

```bash
pip install -r requirements.txt
python modelo/train_model.py
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

Abra o arquivo `frontend/index.html` no navegador.
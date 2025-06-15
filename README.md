# 🫀 Previsão de Doença Cardíaca com ML + SHAP + FastAPI

Este projeto é um exemplo completo de aplicação de Machine Learning com foco em saúde. Ele prevê o risco de doença cardíaca com base em dados clínicos e explica as predições usando SHAP. A aplicação é disponibilizada por meio de uma API com FastAPI e uma interface web intuitiva.

## 📁 Estrutura do Projeto

```
C318-ML/
├── modelo/
│   ├── train_model.py        # Script de treinamento do modelo
│   └── *.pkl                 # Arquivos do modelo e SHAP salvos
├── api/
│   └── main.py               # API com FastAPI que serve o modelo
├── frontend/
│   ├── index.html            # Formulário que consome a API
│   ├── index_local.html      # Formulário para uso local
│   └── result.html           # Página de exibição dos resultados
├── docs/                     # Documentação detalhada
│   ├── api.md                # Endpoints e uso da API
│   ├── docker.md             # Configuração e uso do Docker
│   ├── frontend.md           # Especificações do frontend
│   ├── installation.md       # Guias de instalação e execução
│   ├── model.md              # Detalhes do modelo de ML
│   └── tools.md              # Tecnologias e ferramentas utilizadas
├── Dockerfile                # Dockerfile unificado: treina + sobe API
├── docker-compose.yaml       # Orquestração dos serviços
├── requirements.txt          # Dependências Python
└── README.md                 # Documentação principal
```

## 📚 Documentação Detalhada

- **Documentação da API** (`docs/api.md`): Endpoints, exemplos de requisições e respostas.
- **Configuração com Docker** (`docs/docker.md`): Serviços, portas e comandos essenciais.
- **Especificações do Frontend** (`docs/frontend.md`): Funcionalidades, tecnologias e estrutura.
- **Guia de Instalação** (`docs/installation.md`): Execução com e sem Docker.
- **Detalhes do Modelo de ML** (`docs/model.md`): Pipeline, engenharia de features e métricas.
- **Ferramentas e Tecnologias** (`docs/tools.md`): Frameworks, bibliotecas e dependências.

## 🚀 Como Executar

### 1. Clonar o repositório

```bash
git clone https://github.com/lrgsouza/C318-ML.git
cd C318-ML
```

### 2. Executar com Docker (Recomendado)

```bash
docker-compose up --build
```

- **Swagger UI (API)**: http://localhost:8080/docs  
- **Frontend**: http://localhost

### 3. Executar Localmente (Sem Docker)

1. Instalar dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Treinar o modelo:
   ```bash
   python modelo/train_model.py
   ```
3. Iniciar a API:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8080
   ```
4. Abrir `frontend/index_local.html` no navegador para usar o formulário local.

### ☁️ Deploy na Nuvem

- **Google Cloud Run**: Hospedagem da API  
- **Vercel**: Deploy do frontend

## 🔧 Tecnologias Utilizadas

- **Machine Learning**: Scikit-learn, Pandas, SHAP  
- **API**: FastAPI  
- **Frontend**: HTML, Bootstrap, Plotly.js, JavaScript  
- **Containerização**: Docker, Docker Compose  
- **Cloud Deploy**: Google Cloud Run, Vercel  

## 📌 Créditos

- **Dataset**: Heart Disease – Kaggle  
- **Explainability**: SHAP  
- **Framework de API**: FastAPI  

> **⚠️ Aviso:** Esta aplicação é uma ferramenta de apoio à decisão clínica e não substitui o julgamento médico profissional.

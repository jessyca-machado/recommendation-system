# 🛍️ Recommendation System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-0194E2)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6)
![Pytest](https://img.shields.io/badge/Pytest-Tested-0A9EDC)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![License](https://img.shields.io/badge/License-MIT-green)

Sistema de recomendação de produtos construído utilizando o **RetailRocket Dataset**, aplicando algoritmos de filtragem colaborativa, versionamento de modelos com **MLflow**, versionamento de dados com **DVC** e disponibilização das recomendações através de uma **API REST com FastAPI**.

---

## ✨ Principais funcionalidades

* 📦 Pipeline reproduzível de Machine Learning
* 🤖 Treinamento de múltiplos algoritmos de recomendação
* 📊 Avaliação utilizando métricas de ranking
* 🗂️ Registro de modelos no MLflow
* 🚀 API REST para inferência em tempo real
* 🧪 Testes automatizados com Pytest
* 🐳 Ambiente reproduzível com Docker
* 📁 Versionamento de dados com DVC

---

# 📚 Dataset

O projeto utiliza o **RetailRocket Recommender System Dataset**, composto por milhões de eventos reais de navegação em um e-commerce.

Cada interação recebe um peso diferente durante a construção da matriz usuário-item.

| Evento         | Peso |
| -------------- | ---- |
| 👀 View        | 1    |
| 🛒 Add to Cart | 3    |
| 💳 Transaction | 5    |

---

# 🏗️ Arquitetura

```text
                    RetailRocket Dataset
                             │
                             ▼
                   Data Preparation
                             │
                             ▼
                  Feature Engineering
                             │
                             ▼
                    Model Training
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
    Popularity             ALS/BPR            KNN / MLP
        │                    │                    │
        └────────────────────┴────────────────────┘
                             │
                             ▼
                     Model Evaluation
                             │
                             ▼
                    MLflow Model Registry
                             │
                     Promote: production
                             │
                             ▼
                     FastAPI Inference API
                             │
                             ▼
                    Recommendation Endpoint
```

---

# 📁 Estrutura do projeto

```text
recommendation-system/
│
├── config/
├── data/
├── scripts/
├── src/
│   ├── api/            # a API FastAPI que serve o modelo
│   ├── config/         # Settings via Pydantic (.env) + leitura de params.yaml
│   ├── data/           # carregamento e limpeza dos eventos
│   ├── experiments/    # Orquestra o experimento, treino, mlflow e registry
│   ├── inference//     # Componentes responsáveis pela etapa de inferência do sistema de recomendação
│   ├── models/         # contrato BaseRecommender, baseline (Popularity, ALS, BPR, KNN), rede (MLP) e o Factory
│   └── promote.py      # melhor modelo promovido a production
│
├── tests/
├── docker-compose.yml
├── dvc.yaml
└── pyproject.toml
```

---

# 🧠 Modelos implementados

* Popularity
* ALS (Alternating Least Squares)
* BPR (Bayesian Personalized Ranking)
* KNN
* Neural Collaborative Filtering (MLP)

Todos os modelos são avaliados automaticamente e podem ser registrados no **MLflow Model Registry**.

| Ranking | Modelo | Hit Rate@10 | Precision@10 | NDCG@10 | MRR@10 | Tempo de treino (s) |
|---|---|---|---|---|---|---|
| 1 | Popularity | 0.5400 | 0.0540 | 0.3225 | 0.2559 | 0.08 |
| 2 | MLP | 0.4817 | 0.0483 | 0.2848 | 0.2241 | 5056.32 |
| 3 | ALS | 0.1645 | 0.0164 | 0.0921 | 0.0706 | 115.77 |
| 4 | BPR | 0.1590 | 0.0159 | 0.0833 | 0.0602 | 9.86 |
| 5 | KNN | 0.1106 | 0.0111 | 0.0650 | 0.0516 | 282.66 |

O modelo Popularity ganha em todas as métricas.

---

# ⚙️ Stack

* Python 3.12
* FastAPI
* MLflow
* DVC
* Docker
* PyTorch
* Implicit
* Pandas
* NumPy
* Scikit-learn
* Pytest
* Ruff
* UV
* CPU-only

---

## Docker

A imagem do treino é **multi-stage** (um estágio resolve as dependências com uv, outro é o
runtime enxuto) e roda como usuário não-root. O `docker-compose.yml` sobe dois serviços:

```bash
docker compose up mlflow             # servidor MLflow (UI em :5000)
docker compose up --build train      # constrói e roda o treino
docker compose down
```

---

---

# 🚀 Como executar

```bash
uv sync --group deep    # cria o ambiente e instala tudo a partir do lock
cp .env.example .env    # ajuste se precisar
uv run validate-env     # confere se a configuração está ok
uv run pytest           # roda os testes
uv run train            # roda os experimentos, treino e model registry
uv run promote          # promove o melhor modelo do experimento para produção
dvc repro               # roda o pipeline inteiro (preprocess -> promote)
```

---

# Pipeline de dados e modelo (DVC)

O pipeline é orquestrado pelo **DVC**, garantindo reprodutibilidade e execução incremental. Ao executar `dvc repro`, apenas os estágios afetados por alterações são reexecutados.

| Estágio | Entrada → saída | O que faz |
|---------|-----------------|-----------|
| `prepare` | `events.csv` → `train.parquet`, `test.parquet`, `item_catalog.parquet` | Realiza a divisão temporal entre treino e teste, cria os identificadores internos (`user_idx` e `item_idx`) e gera o catálogo de itens utilizado pela API. |
| `experiments` | dados processados → `experiment_results.json`, `best_model.json` | Treina e avalia automaticamente os algoritmos (Popularity, ALS, BPR, KNN e MLP), registra os experimentos no MLflow e salva as informações do melhor modelo. |
| `promote` | `best_model.json` → MLflow Model Registry | Promove a versão vencedora registrada no MLflow do alias `staging` para `production`, disponibilizando-a para consumo pela API. |

Durante a etapa de experimentação, os modelos são avaliados utilizando o protocolo **Neural Collaborative Filtering (NCF)**, no qual cada usuário possui uma interação positiva e 99 itens negativos amostrados para cálculo das métricas de ranking (Hit Rate, Precision, NDCG e MRR).

---

# Versionamento de dados

O dataset bruto e os artefatos produzidos pelo pipeline são versionados utilizando o **DVC**, permitindo reproduzir qualquer experimento a partir da mesma versão dos dados.

```bash
dvc repro      # executa o pipeline completo
dvc push       # envia os artefatos para o remoto
dvc pull       # recupera os artefatos da versão atual
```

O projeto foi desenvolvido utilizando um **Remote Adapter**, que desacopla a configuração do armazenamento remoto do restante da aplicação. Essa abordagem permite utilizar diferentes backends sem alterar o pipeline.

Atualmente são suportados:

| Backend | Exemplo |
|----------|---------|
| Local | `./artifacts/dvc-cache` |
| Amazon S3 | `s3://bucket-name/dvc-cache` |
| Google Cloud Storage | `gs://bucket-name/dvc-cache` |

O adaptador identifica automaticamente o tipo de armazenamento a partir da URI configurada e monta o plano de configuração correspondente. Para ambientes sem acesso ao armazenamento em nuvem, existe um mecanismo de **fallback offline**, que utiliza um cache local para permitir a execução do pipeline.

As credenciais dos provedores em nuvem permanecem fora do repositório e nunca são versionadas pelo Git.

---

# 🌐 Executando a API

Inicie o MLflow:

```bash
docker compose up -d mlflow
```

Depois execute a API:

```bash
uv run uvicorn src.api.app:app --reload
```

Documentação automática:

```text
http://localhost:8000/docs
```

---

# 📡 Endpoints

## Health Check

```http
GET /health
```

Resposta

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## Recomendações

```http
GET /recommend/{user_idx}?k=10
```

Resposta

```json
{
  "user_idx": 0,
  "recommendations": [
    {
      "item_idx": 2881,
      "rank": 1,
      "score": 1.0,
      "itemid": 5411
    }
  ]
}
```

---

# 🧪 Testes

Executar todos os testes

```bash
uv run pytest
```

Executar com cobertura

```bash
uv run pytest --cov=src --cov-report=term-missing
```

Os testes cobrem:

* inicialização da API
* carregamento do modelo via lifespan
* carregamento do catálogo
* contrato dos endpoints
* estrutura da resposta
* ordenação das recomendações
* quantidade de itens retornados
* parametrização do `top-k`

---

# Testes e qualidade

`ruff` cuida do lint e da formatação (com `pre-commit`), e o `pytest` cobre as peças
principais — encoding, split, baseline, rede neural, avaliador e a API.

---

# 📄 Licença

Este projeto foi desenvolvido com fins educacionais para demonstrar boas práticas em **Machine Learning Engineering**, **MLOps** e **Sistemas de Recomendação**.

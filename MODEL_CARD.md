# Model Card — Recommendation System (RetailRocket)

Este documento descreve o modelo atualmente disponibilizado pela API de recomendação. O objetivo é documentar o propósito do modelo, os dados utilizados, o processo de treinamento, as métricas obtidas e suas principais limitações.

---

## Detalhes do modelo

- **Projeto:** Recommendation System
- **Registry:** MLflow Model Registry
- **Versão:** controlada pelo MLflow
- **Alias de inferência:** `production`
- **Tipo:** Sistema de recomendação baseado em feedback implícito
- **Algoritmos avaliados:**
  - Popularity
  - Alternating Least Squares (ALS)
  - Bayesian Personalized Ranking (BPR)
  - Item-based KNN
  - Multi-Layer Perceptron (MLP)
- **Modelo atualmente promovido:** Popularity
- **Frameworks utilizados:**
  - Python
  - Pandas
  - SciPy
  - PyTorch
  - Implicit
  - MLflow

---

## Objetivo

O sistema recebe o identificador de um usuário (`user_idx`) e retorna uma lista ordenada de itens recomendados com base no histórico de interações observado durante o treinamento.

O projeto foi desenvolvido para demonstrar um pipeline completo de Machine Learning, incluindo:

- Preparação dos dados;
- Treinamento de múltiplos algoritmos;
- Comparação automática de desempenho;
- Registro dos experimentos no MLflow;
- Promoção automática do melhor modelo;
- disponibilização via API REST.

---

## Dados

**Dataset**

- **Fonte:** [RetailRocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
  (eventos `view` / `addtocart` / `transaction`, ~4,5 meses).

O conjunto contém aproximadamente quatro meses e meio de eventos de navegação em um e-commerce.

Eventos considerados:

- `view`
- `addtocart`
- `transaction`

Os eventos recebem pesos diferentes durante a construção da matriz de interação:

| Evento | Peso |
|---------|-----:|
| view | 1 |
| addtocart | 3 |
| transaction | 5 |

---

## Preparação dos dados

O pipeline realiza automaticamente:

- Leitura do dataset bruto;
- Divisão temporal entre treino e teste;
- Geração dos identificadores internos (`user_idx` e `item_idx`);
- Remoção de usuários e itens não presentes no conjunto de treino;
- Construção do catálogo de itens utilizado pela API.

Os dados processados são versionados pelo DVC.

---

## Avaliação

Todos os modelos são avaliados utilizando o protocolo proposto pelo artigo **Neural Collaborative Filtering (NCF)**.

Para cada usuário:

- É selecionada uma interação positiva;
- São amostrados 99 itens negativos;
- O modelo gera um ranking Top-K;
- São calculadas as métricas de recomendação.

Métricas utilizadas:

- Hit Rate@10
- Precision@10
- NDCG@10
- MRR@10

### Ranking dos modelos


| Ranking | Modelo | Hit Rate@10 | Precision@10 | NDCG@10 | MRR@10 | Tempo de treino (s) |
|---|---|---|---|---|---|---|
| **1** | **Popularity** | **0.5400** | **0.0540** | **0.3225** | **0.2559** | **0.08** |
| 2 | MLP | 0.4817 | 0.0483 | 0.2848 | 0.2241 | 5056.32 |
| 3 | ALS | 0.1645 | 0.0164 | 0.0921 | 0.0706 | 115.77 |
| 4 | BPR | 0.1590 | 0.0159 | 0.0833 | 0.0602 | 9.86 |
| 5 | KNN | 0.1106 | 0.0111 | 0.0650 | 0.0516 | 282.66 |

O modelo promovido é aquele que apresenta o maior valor de **NDCG@10**.

---

## Pipeline de treinamento

O pipeline automatizado executa as seguintes etapas:

```text
Prepare
    ↓
Experiments
    ↓
Registro do melhor modelo no MLflow
    ↓
Promoção para o alias production
    ↓
API de inferência
```

Durante os experimentos são registrados:

- Parâmetros;
- Métricas;
- Tempo de treinamento;
- Artefatos;
- Versão do modelo.

---

## Limitações

Este projeto possui algumas limitações conhecidas.

### Cold Start

Usuários e itens inexistentes durante o treinamento não podem ser recomendados.

### Feedback implícito

O modelo utiliza apenas interações observadas, sem informações explícitas de preferência.

### Sem informações de conteúdo

O sistema não considera atributos dos produtos, como:

- Categoria;
- Descrição;
- Preço;
- Marca.

Todas as recomendações são produzidas exclusivamente a partir do comportamento histórico dos usuários.

### Mudança de comportamento

Como qualquer sistema de recomendação, o desempenho pode degradar ao longo do tempo caso o comportamento dos usuários se altere significativamente, exigindo novo treinamento.

---

## Reprodução

```
uv sync
cp .env.example .env
uv run validate-env
uv run pytest
uv run train
uv run promote
dvc repro
```

O pipeline executa automaticamente:

- Preparação dos dados;
- Treinamento e comparação dos modelos;
- Registro do melhor modelo no MLflow;
- Promoção para o alias `production`.

---

## API

Após a promoção do modelo, a API carrega automaticamente o modelo associado ao alias configurado no MLflow.

Endpoints disponíveis:

```
GET /
GET /health
GET /recommend/{user_idx}
```

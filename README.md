# Credit Segmentation — Carteira de Crédito

Segmentação de clientes de cartão de crédito usando clustering avançado,
com pipeline automatizado, testes unitários e interpretação de negócio
orientada ao time de Crédito.

## Contexto de Negócio
Instituições financeiras tratam clientes de crédito de forma homogênea,
aplicando as mesmas políticas independente do comportamento individual.
Este projeto identifica 4 segmentos distintos na carteira, permitindo
estratégias diferenciadas de limite, cobrança e oferta de produtos.

## Resultados
| Segmento | Clientes | % Carteira | Risco | Ação Recomendada |
|----------|----------|------------|-------|------------------|
| Inativos | 1.480 | 16.5% | Baixo | Campanha de ativação |
| Sacadores Alto Risco | 2.996 | 33.5% | Alto | Reduzir limite, monitorar |
| Compradores Endividados | 2.334 | 26.1% | Médio-Alto | Renegociação proativa |
| Saudáveis | 2.140 | 23.9% | Baixo | Expansão de limite e produtos |

## Pipeline Técnico
```
Dados Brutos
    → Tratamento de Nulos
    → Transformação Logarítmica (assimetria)
    → Padronização (StandardScaler)
    → Redução de Dimensionalidade (PCA — 6 componentes, 80.4% variância)
    → Clustering (K-Means, K=4, Silhouette=0.262)
```

## Estrutura do Projeto
```
credit-segmentation/
├── data/
│   ├── raw/                    # dados originais
│   └── processed/              # dados tratados e clusterizados
├── notebooks/
│   ├── 01_eda.ipynb            # análise exploratória
│   ├── 02_feature_engineering.ipynb  # transformações e PCA
│   └── 03_modelagem.ipynb      # clustering e interpretação
├── src/
│   ├── pipeline.py             # pipeline sklearn completo
│   ├── data.py
│   ├── features.py
│   └── model.py
├── tests/
│   └── test_pipeline.py        # 11 testes unitários
├── models/                     # pipeline, scaler, pca salvos
└── requirements.txt
```

## Decisões Técnicas

**Por que transformação logarítmica?**
Variáveis financeiras têm distribuição assimétrica à direita com outliers
extremos. O log1p comprime a cauda sem deletar clientes VIP, que são
segmentos valiosos.

**Por que PCA antes do clustering?**
17 variáveis com alta multicolinearidade (ex: PURCHASES x ONEOFF_PURCHASES
= 0.92) distorcem as distâncias euclidianas do K-Means. PCA elimina
redundância e melhora a qualidade dos clusters.

**Por que K=4?**
Elbow Method indicou cotovelo em K=6, Silhouette indicou K=2. K=4 equilibra
qualidade estatística (Silhouette=0.262) com interpretabilidade de negócio —
segmentos acionáveis pelo time de Crédito.

**Por que sklearn Pipeline?**
Garante a mesma sequência de transformações em treino e produção,
prevenindo data leakage e erros de encadeamento.

## Componentes PCA Identificados
| Componente | Variância | Interpretação |
|------------|-----------|---------------|
| PC1 | 29.9% | Comportamento de Compra |
| PC2 | 21.9% | Nível de Endividamento |
| PC3 | 9.3% | Modalidade (à vista vs parcelado) |
| PC4 | 7.7% | Frequência de Saques |
| PC5 | 6.5% | Relacionamento e Limite |
| PC6 | 5.0% | Disciplina Financeira |

## Setup
```bash
git clone https://github.com/0Ryab/credit-segmentation.git
cd credit-segmentation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Rodando o Pipeline
```bash
python src/pipeline.py
```

## Rodando os Testes
```bash
python -m pytest tests/ -v
```

## Tecnologias
- Python 3.13
- Scikit-learn (Pipeline, KMeans, PCA, StandardScaler)
- Pandas, NumPy
- Matplotlib, Seaborn
- Pytest

## Etapas
- [x] Fase 1 — Setup e estrutura do projeto
- [x] Fase 2 — EDA
- [x] Fase 3 — Feature Engineering
- [x] Fase 4 — Modelagem
- [x] Fase 5 — Pipeline e Testes
- [x] Fase 6 — Documentação
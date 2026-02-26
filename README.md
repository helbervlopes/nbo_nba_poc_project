# Implementação Reprodutível (PoC) – NBO/NBA
# PUCRS – Pontifícia Universidade Católica do Rio Grande do Sul
# MBA em Tecnologia para Negócios: AI, Data Science e Big Data
# Uma Proposta de Modelo de NBO e NBA para Personalização e Aumento de Conversão
# MODELAGEM DE NEGÓCIO
# HELBER VIEIRA LOPES

Este projeto implementa uma PoC alinhada ao TCC (sem dados reais). Componentes:
1) Modelos supervisionados:
   - churn_90d (classificação)
   - convert_30d (classificação)
   - ltv_12m (regressão)
2) Segmentação comportamental (MiniBatchKMeans)
3) Camada decisória NBA (regras + scores)

## Execução
pip install -r requirements.txt
python src/train_and_score.py --data data_sintetica_mobilidade.csv --outdir outputs

## Métricas (indicativas)
- Churn: AUC=0.616, Precision@0.35=0.288, Recall@0.35=0.077
- Conversão: AUC=0.562, Precision@0.30=0.318, Recall@0.30=0.023
- LTV: RMSE=271.52

Arquivos principais:
- data_sintetica_mobilidade.csv
- outputs/scored_with_nba.csv
- exemplos_scores_decisoes.csv


## Nota sobre instalação do scikit-learn
Importe com:
    from sklearn import ...
Instale com:
    pip install scikit-learn
(Não use `pip install sklearn`)

## V3 – Reprodutibilidade e diagnóstico (Windows-friendly)

### Regra de ouro (erro comum)
- **Instale**: `pip install scikit-learn`
- **Importe**: `from sklearn ...`
- **Não use**: `pip install sklearn`

### Diagnóstico rápido do ambiente
Depois de ativar o venv, rode:
```
python src/verify_env.py
```

### Execução (pipeline completo)
```
python src/train_and_score.py --data data_sintetica_mobilidade.csv --outdir outputs
```

### Smoke test (execução rápida)
```
python src/smoke_test.py
```



## V4 – Explicabilidade, calibração e ranking NBO (Top-N)

### Explicabilidade (Permutation Importance)
```
python src/explainability.py --data data_sintetica_mobilidade.csv --target churn_90d --outdir outputs
python src/explainability.py --data data_sintetica_mobilidade.csv --target convert_30d --outdir outputs
```

### Calibração de probabilidades (sigmoid/isotonic)
```
python src/calibration.py --data data_sintetica_mobilidade.csv --target churn_90d --method sigmoid --outdir outputs
python src/calibration.py --data data_sintetica_mobilidade.csv --target convert_30d --method sigmoid --outdir outputs
```

### Ranking NBO (Top-N ofertas por cliente)
Após rodar o pipeline principal (gerando outputs/scored_with_nba.csv):
```
python src/rank_offers.py --scored outputs/scored_with_nba.csv --offers offers.yaml --out outputs/top3_offers.csv --topn 3
```

Função de valor esperado (exemplo):
EV = p_convert * (margin + alpha_ltv * LTV_hat) - cost

### relatorio  hb
python src/generate_full_report_v4.py

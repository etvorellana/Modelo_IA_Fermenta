# Modelo_IA_Fermenta

## Inferencia temporal

O script [inferencia_time.py](inferencia_time.py) carrega o checkpoint salvo em `artifacts_time/modelo_temporal.pth` e gera previsoes ate um tempo final usando passo temporal fixo.

Uso:

```bash
python inferencia_time.py --final-time 125.0 --time-step 0.5
```

Opcionalmente, voce pode informar outros caminhos:

```bash
python inferencia_time.py \
	--checkpoint artifacts_time/modelo_temporal.pth \
	--dataset report-file-1.csv \
	--final-time 125.0 \
	--time-step 0.5 \
	--output artifacts_time/previsoes_futuras_usuario.csv
```

O CSV de saida contem `flow_time` acumulado, `delta_time` usado em cada passo e as previsoes de `mon_sacarose`, `mon_glicose` e `mon_etanol`.

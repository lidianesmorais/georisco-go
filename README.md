#####Autor: Lidiane S Morais
#####Data: março de 2026

# GeoRisco – Goiás | Layout preservado

Esta versão mantém o layout e a estrutura visual da versão anterior, mas adiciona:
- suporte a CSV, Excel (.xlsx) e GeoJSON
- leitura robusta para arquivos brasileiros
- diagnóstico automático da base enviada
- aceitação de colunas extras sem quebrar a análise

## Campos obrigatórios
- UTM_E_m
- UTM_N_m
- ZONA

## Arquivos aceitos
- CSV
- Excel (.xlsx)
- GeoJSON

## Execução
```bash
pip install -r requirements.txt
streamlit run app.py

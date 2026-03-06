# GeoRisco GO — Versão 5.0

Versão beta com foco em **aceitação de arquivos reais de usuários** e **diagnóstico automático da base**.

## Arquivos aceitos
- CSV
- Excel `.xlsx`
- GeoJSON

## O que esta versão faz
- lê arquivos brasileiros e internacionais;
- detecta automaticamente o formato da base;
- verifica campos obrigatórios;
- identifica campos recomendados;
- identifica variáveis hidrogeológicas;
- aceita colunas extras;
- ignora colunas não utilizadas;
- gera um diagnóstico automático da prontidão da base.

## Campos obrigatórios
- `UTM_E_m`
- `UTM_N_m`
- `ZONA`

## Campos recomendados
- `IDADE`
- `QUANTIDADE DE TANQUES`
- `QUANTIDADE DE BOMBAS`
- `JAQUETADO`
- `MÉDIA DE IDADE DO TANQUE`
- `NÚMERO DE SONDAGENS`
- `ÁGUA SUBTERRÂNEA`
- `NIVEL_AGUA_m`

## Campos hidrogeológicos
- `dist_poco_mais_prox_m`
- `pocos_500m`
- `pocos_1km`
- `pocos_5km`

## Histórico e indicadores auxiliares
- `JÁ APRESENTOU CONTAMINAÇÃO ANTES`
- `CONC. BTEX`
- `CONC. PAH`

## Regras da versão beta
- obrigatórios ausentes → não executa a etapa principal;
- recomendados ausentes → executa com menor qualidade analítica;
- hidrogeológicos ausentes → executa, mas menos aderente ao estudo;
- campos extras → aceitos e ignorados quando não forem necessários.

## Execução local
```bash
pip install -r requirements.txt
streamlit run app.py
```

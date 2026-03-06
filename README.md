# GeoRisco – Goiás | Versão 3

Aplicação web em Streamlit para:
- carregar CSV de postos;
- aplicar modelo de previsão de risco (`.joblib`) ou usar coluna de risco existente;
- explicar ao usuário quais colunas o CSV deve ter;
- disponibilizar template de CSV para download;
- gerar mapa interativo de Goiás com:
  - limite do estado,
  - grade hexagonal,
  - hotspots,
  - heatmap,
  - coordenadas do mouse,
  - escala,
  - data de geração,
  - norte geográfico,
  - datum;
- gerar mapas estáticos em PNG com estrutura cartográfica;
- exportar CSV, GeoJSON, HTML e PNG.

## Como executar localmente

```bash
conda activate ambiental_sig
pip install -r requirements.txt
streamlit run app.py
```

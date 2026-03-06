# GeoRisco – Goiás | Versão 2

Aplicação web em Streamlit para:
- carregar CSV de postos;
- aplicar modelo de previsão de risco (`.joblib`) ou usar coluna de risco existente;
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
pip install -r requirements_v2.txt
streamlit run app_v2.py
```

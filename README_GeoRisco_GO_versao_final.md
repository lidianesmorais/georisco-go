# GeoRisco – Goiás | Versão Final

Aplicação web em Streamlit para:
- carregar CSV de postos;
- aplicar modelo de previsão de risco (`.joblib`) ou usar coluna de risco existente;
- explicar ao usuário quais colunas o CSV deve ter;
- disponibilizar um modelo robusto de CSV para download;
- apresentar painéis explicativos sobre o modelo, a importância das variáveis e a interpretação do risco;
- gerar mapa interativo de Goiás;
- gerar mapas estáticos em PNG;
- oferecer explicação individual do risco com explicabilidade do modelo;
- exportar CSV, GeoJSON, HTML e PNG.

## Como executar localmente

```bash
conda activate ambiental_sig
pip install -r requirements.txt
streamlit run app.py
```


import io, math, tempfile
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from folium import plugins
import branca.colormap as cm
import shap

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from pyproj import Transformer
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

st.set_page_config(page_title="GeoRisco – Goiás", page_icon="🗺️", layout="wide")
st.title("🗺️ GeoRisco – Goiás")
st.caption("Aplicação interativa para cálculo de risco ambiental em postos de combustíveis no estado de Goiás.")

st.info("""Esta aplicação utiliza **aprendizado de máquina** e **análise espacial** para estimar o
**risco ambiental de contaminação** em postos de combustíveis no estado de Goiás.

A ferramenta considera fatores como:
- presença de água subterrânea;
- histórico de contaminação;
- proximidade de poços de água;
- características estruturais dos tanques;
- contexto espacial do entorno.""")

ARQUIVO_MODELO_PADRAO = Path("rf_model_com_hidro.joblib")
ARQUIVO_LIMITE_GOIAS = Path("LimiteEstadoGO/1_Estado-Goias_SIRGAS_Poly.shp")
ARQUIVO_IMPORTANCIA = Path("rf_feature_importance_com_hidro.csv")

for chave, valor in {
    "analise_concluida": False,
    "dados_resultado": None,
    "coluna_risco": None,
    "grade_hexagonal": None,
    "mapa_html": None,
    "mapa_risco_png": None,
    "mapa_hotspots_png": None,
    "estatisticas": None,
    "modelo_carregado": None,
    "explainer": None,
    "nomes_variaveis_transformadas": None,
    "base_transformada": None,
}.items():
    if chave not in st.session_state:
        st.session_state[chave] = valor

def ler_csv_seguro(arquivo_enviado):
    bruto = arquivo_enviado.getvalue()
    for codificacao in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(bruto), encoding=codificacao)
        except Exception:
            pass
    raise ValueError("Não foi possível ler o arquivo CSV. Salve o arquivo em UTF-8 e tente novamente.")

def detectar_coluna_risco(df):
    for coluna in ["RISK_PROBA_HIDRO", "RISK_PROBA", "risco", "risco_medio"]:
        if coluna in df.columns:
            return coluna
    return None

def converter_utm_para_wgs84(df, coluna_e="UTM_E_m", coluna_n="UTM_N_m", coluna_zona="ZONA"):
    base = df.copy()
    base[coluna_e] = pd.to_numeric(base[coluna_e], errors="coerce")
    base[coluna_n] = pd.to_numeric(base[coluna_n], errors="coerce")
    base[coluna_zona] = base[coluna_zona].astype(str).str.strip().str.upper()
    base = base.dropna(subset=[coluna_e, coluna_n, coluna_zona]).copy()

    longitudes, latitudes = [], []
    for _, linha in base.iterrows():
        zona_numero = "".join(ch for ch in str(linha[coluna_zona]) if ch.isdigit())
        zona_numero = int(zona_numero) if zona_numero else 22
        transformador = Transformer.from_crs(f"EPSG:{32700 + zona_numero}", "EPSG:4326", always_xy=True)
        lon, lat = transformador.transform(linha[coluna_e], linha[coluna_n])
        longitudes.append(lon)
        latitudes.append(lat)

    base["lon"] = longitudes
    base["lat"] = latitudes
    return base

def carregar_limite_goias():
    if not ARQUIVO_LIMITE_GOIAS.exists():
        raise FileNotFoundError(f"Não encontrei o shapefile do limite de Goiás em: {ARQUIVO_LIMITE_GOIAS}")
    goias = gpd.read_file(ARQUIVO_LIMITE_GOIAS)
    crs_origem = str(goias.crs)
    return goias.to_crs(epsg=4326), crs_origem

def obter_colunas_esperadas_modelo(modelo):
    try:
        if hasattr(modelo, "feature_names_in_"):
            return list(modelo.feature_names_in_)
    except Exception:
        pass
    try:
        preprocessamento = modelo.named_steps["preprocess"]
        esperadas = []
        for _, _, colunas in preprocessamento.transformers:
            if colunas == "drop":
                continue
            if isinstance(colunas, list):
                esperadas.extend(colunas)
        return list(dict.fromkeys(esperadas))
    except Exception:
        return None

def aplicar_modelo(df, modelo):
    base = df.copy()
    colunas_esperadas = obter_colunas_esperadas_modelo(modelo)
    if colunas_esperadas is None:
        X = base.copy()
    else:
        for coluna in colunas_esperadas:
            if coluna not in base.columns:
                base[coluna] = np.nan
        X = base[colunas_esperadas].copy()
    base["RISK_PROBA_HIDRO"] = modelo.predict_proba(X)[:, 1]
    return base, "RISK_PROBA_HIDRO"

def criar_hexagono(x_centro, y_centro, tamanho):
    return Polygon([
        (
            x_centro + tamanho * math.cos(math.radians(angulo)),
            y_centro + tamanho * math.sin(math.radians(angulo))
        )
        for angulo in range(0, 360, 60)
    ])

def construir_grade_hexagonal(pontos_wgs, limite_wgs, coluna_risco, tamanho_hexagono=5000.0):
    pontos_m = pontos_wgs.to_crs(epsg=3857)
    limite_m = limite_wgs.to_crs(epsg=3857)
    minx, miny, maxx, maxy = pontos_m.total_bounds
    minx -= tamanho_hexagono; miny -= tamanho_hexagono; maxx += tamanho_hexagono; maxy += tamanho_hexagono
    dx = 1.5 * tamanho_hexagono
    dy = math.sqrt(3) * tamanho_hexagono

    geometrias, ids = [], []
    coluna = 0
    x = minx
    while x < maxx:
        y = miny + (0 if coluna % 2 == 0 else dy / 2)
        linha = 0
        while y < maxy:
            geometrias.append(criar_hexagono(x, y, tamanho_hexagono))
            ids.append(f"hex_{coluna}_{linha}")
            y += dy; linha += 1
        x += dx; coluna += 1

    grade = gpd.GeoDataFrame({"hex_id": ids}, geometry=geometrias, crs=pontos_m.crs)
    grade = gpd.overlay(grade, limite_m, how="intersection")
    juncao = gpd.sjoin(pontos_m, grade, how="left", predicate="within")
    estatisticas = juncao.groupby("hex_id").agg(
        postos=(coluna_risco, "count"),
        risco_medio=(coluna_risco, "mean"),
        risco_maximo=(coluna_risco, "max")
    ).reset_index()
    grade = grade.merge(estatisticas, on="hex_id", how="left")
    for c in ["postos", "risco_medio", "risco_maximo"]:
        grade[c] = grade[c].fillna(0)
    grade = grade[grade["postos"] > 0].copy()
    grade["hotspot"] = np.where(grade["risco_medio"] >= grade["risco_medio"].quantile(0.75), 1, 0) if len(grade) else 0
    return grade.to_crs(epsg=4326)

def adicionar_seta_norte(ax, x=0.95, y=0.12):
    ax.annotate("N", xy=(x, y), xytext=(x, y - 0.10),
                arrowprops=dict(facecolor="black", width=3, headwidth=10),
                ha="center", va="center", fontsize=12, xycoords=ax.transAxes)

def construir_mapa_interativo(df_pontos, limite_wgs, grade_wgs, coluna_risco, crs_origem_limite):
    mapa = folium.Map(
        location=[float(df_pontos["lat"].mean()), float(df_pontos["lon"].mean())],
        zoom_start=6, tiles="CartoDB positron", control_scale=True
    )
    folium.GeoJson(limite_wgs, name="Limite de Goiás",
                   style_function=lambda x: {"color": "black", "weight": 2, "fillOpacity": 0},
                   tooltip="Estado de Goiás").add_to(mapa)

    minimo = float(grade_wgs["risco_medio"].min()) if len(grade_wgs) else 0.0
    maximo = float(grade_wgs["risco_medio"].max()) if len(grade_wgs) else 1.0
    if maximo == minimo:
        maximo = minimo + 0.0001

    escala = cm.LinearColormap(["#440154", "#31688e", "#35b779", "#fde725"], vmin=minimo, vmax=maximo, caption="Risco médio por hexágono")

    folium.GeoJson(
        grade_wgs, name="Grade hexagonal de risco",
        style_function=lambda f: {
            "fillColor": escala(f["properties"].get("risco_medio", 0)),
            "color": "black", "weight": 0.5, "fillOpacity": 0.55
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[c for c in ["hex_id", "postos", "risco_medio", "risco_maximo", "hotspot"] if c in grade_wgs.columns],
            aliases=["Hexágono", "Nº de postos", "Risco médio", "Risco máximo", "Hotspot"]
        )
    ).add_to(mapa)
    escala.add_to(mapa)

    hotspots = grade_wgs[grade_wgs["hotspot"] == 1].copy() if "hotspot" in grade_wgs.columns else grade_wgs.iloc[0:0]
    if len(hotspots):
        folium.GeoJson(
            hotspots, name="Hotspots",
            style_function=lambda x: {"fillColor": "#ff0000", "color": "#8b0000", "weight": 1.2, "fillOpacity": 0.30},
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ["hex_id", "risco_medio", "hotspot"] if c in hotspots.columns],
                aliases=["Hexágono", "Risco médio", "Hotspot"]
            )
        ).add_to(mapa)

    agrupador = plugins.MarkerCluster(name="Postos").add_to(mapa)
    for _, linha in df_pontos.iterrows():
        risco = float(linha[coluna_risco])
        popup = [f"<b>Posto:</b> {linha.get('NÚMERO DO POSTO', 'N/D')}",
                 f"<b>Cidade:</b> {linha.get('CIDADE', 'N/D')}",
                 f"<b>Risco:</b> {risco:.3f}"]
        folium.CircleMarker(
            location=[linha["lat"], linha["lon"]],
            radius=4 + 10 * risco,
            color="#d62728", fill=True, fill_color="#d62728", fill_opacity=0.75, weight=1,
            popup=folium.Popup("<br>".join(popup), max_width=350),
            tooltip=f"{linha.get('CIDADE', 'Posto')} | risco {risco:.3f}"
        ).add_to(agrupador)

    plugins.HeatMap(df_pontos[["lat", "lon", coluna_risco]].values.tolist(), name="Mapa de calor dos postos",
                    radius=25, blur=18, min_opacity=0.30).add_to(mapa)
    plugins.MousePosition(position="bottomleft", separator=" | ", prefix="Coordenadas", num_digits=6).add_to(mapa)
    plugins.MiniMap(toggle_display=True).add_to(mapa)
    folium.LayerControl(collapsed=False).add_to(mapa)

    data_geracao = datetime.now().strftime("%d/%m/%Y %H:%M")
    mapa.get_root().html.add_child(folium.Element(f'<div style="position: fixed; bottom: 85px; left: 10px; z-index: 9999; background-color: white; padding: 8px 10px; border: 1px solid #777; border-radius: 4px; font-size: 12px;"><b>Data de geração:</b><br>{data_geracao}</div>'))
    mapa.get_root().html.add_child(folium.Element(f'<div style="position: fixed; bottom: 10px; left: 10px; z-index: 9999; background-color: white; padding: 8px 10px; border: 1px solid #777; border-radius: 4px; font-size: 12px;"><b>Datum / sistema de referência:</b><br>Mapa web: WGS84 (EPSG:4326)<br>Limite GO (origem): {crs_origem_limite}</div>'))
    mapa.get_root().html.add_child(folium.Element('<div style="position: fixed; top: 10px; right: 10px; z-index: 9999; background-color: white; width: 62px; text-align: center; padding: 6px 4px; border: 1px solid #777; border-radius: 4px; font-size: 12px;"><div style="font-weight: bold;">N</div><div style="font-size: 24px; line-height: 22px;">↑</div><div style="font-size: 11px;">Norte</div></div>'))
    return mapa

def gerar_mapas_estaticos(pontos_wgs, limite_wgs, grade_wgs, crs_origem_limite):
    texto_data = datetime.now().strftime("%d/%m/%Y %H:%M")

    def desenhar(coluna, titulo, cor_pontos):
        fig, ax = plt.subplots(figsize=(10, 10))
        limite_wgs.boundary.plot(ax=ax, linewidth=1.8, color="black")
        if len(grade_wgs):
            grade_wgs.plot(column=coluna, legend=True, ax=ax, cmap="viridis" if coluna=="risco_medio" else "Reds",
                           edgecolor="black", linewidth=0.3, alpha=0.7)
        pontos_wgs.plot(ax=ax, markersize=10, color=cor_pontos)
        ax.set_title(titulo, fontsize=14)
        ax.set_xlabel("Longitude (graus)"); ax.set_ylabel("Latitude (graus)")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
        adicionar_seta_norte(ax)
        ax.add_artist(AnchoredSizeBar(ax.transData, 1.0, "Escala indicativa", "lower right",
                                      pad=0.4, color="black", frameon=True, size_vertical=0.01,
                                      fontproperties=fm.FontProperties(size=9)))
        ax.text(0.01, 0.01, f'''Gerado em: {texto_data}
Datum do mapa: WGS84 (EPSG:4326)
Limite GO (origem): {crs_origem_limite}''',
                transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
        fig.tight_layout()
        arquivo = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(arquivo.name, dpi=220)
        plt.close(fig)
        return arquivo.name

    return (
        desenhar("risco_medio", "Mapa hexagonal de risco médio — Goiás", "red"),
        desenhar("hotspot", "Hotspots de risco ambiental — Goiás", "blue")
    )

def construir_template_robusto():
    return pd.DataFrame({
        "NÚMERO DO POSTO": [1, 2],
        "CIDADE": ["GOIÂNIA", "ANÁPOLIS"],
        "ZONA": ["22K", "22K"],
        "UTM_E_m": [686304.36, 745210.55],
        "UTM_N_m": [8216649.19, 8198500.12],
        "IDADE": [20, 12],
        "QUANTIDADE DE TANQUES": [4, 3],
        "QUANTIDADE DE BOMBAS": [8, 6],
        "JAQUETADO": ["SIM", "NÃO"],
        "MÉDIA DE IDADE DO TANQUE": [15, 10],
        "NÚMERO DE SONDAGENS ": [4, 2],
        "ÁGUA SUBTERRÂNEA": [1, 0],
        "NIVEL_AGUA_m": [10.5, 14.2],
        "dist_poco_mais_prox_m": [240.0, 680.0],
        "pocos_500m": [2, 0],
        "pocos_1km": [7, 3],
        "pocos_5km": [120, 85],
        "JÁ APRESENTOU CONTAMINAÇÃO ANTES": [1, 0],
        "CONC. BTEX": [0, 0],
        "CONC. PAH": [0, 0],
    })

def montar_explicador(modelo, df_para_explicar):
    preprocessamento = modelo.named_steps["preprocess"]
    floresta = modelo.named_steps["rf"]
    X_transformado = preprocessamento.transform(df_para_explicar)
    if hasattr(X_transformado, "toarray"):
        X_transformado = X_transformado.toarray()
    nomes = []
    try:
        nomes.extend(list(preprocessamento.named_transformers_["num"].get_feature_names_out()))
    except Exception:
        pass
    try:
        nomes.extend(list(preprocessamento.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out()))
    except Exception:
        pass
    return shap.TreeExplainer(floresta), X_transformado, nomes

def grafico_importancia_variaveis():
    if not ARQUIVO_IMPORTANCIA.exists():
        return None
    imp = pd.read_csv(ARQUIVO_IMPORTANCIA)
    if "feature" not in imp.columns or "importance" not in imp.columns:
        return None
    topo = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(topo["feature"], topo["importance"])
    ax.set_title("Importância das variáveis do modelo")
    ax.set_xlabel("Importância")
    fig.tight_layout()
    return fig

def grafico_explicacao_local(explicador, base_transformada, nomes_variaveis, posicao_linha):
    """
    Gera um gráfico com os principais fatores que aumentaram ou reduziram
    o risco estimado para um posto específico.

    Esta função trata diferentes formatos de saída do SHAP,
    evitando erro de dimensão (1D x 2D).
    """
    try:
        valores = explicador.shap_values(base_transformada)

        # ----------------------------------------------------
        # Caso 1: SHAP retorna lista (classificação binária)
        # ----------------------------------------------------
        if isinstance(valores, list):
            if len(valores) > 1:
                contribuicoes = valores[1][posicao_linha]
            else:
                contribuicoes = valores[0][posicao_linha]

        # ----------------------------------------------------
        # Caso 2: SHAP retorna array 3D
        # Ex.: (n_amostras, n_variaveis, n_classes)
        # ----------------------------------------------------
        elif isinstance(valores, np.ndarray) and valores.ndim == 3:
            contribuicoes = valores[posicao_linha, :, 1]

        # ----------------------------------------------------
        # Caso 3: SHAP retorna array 2D
        # Ex.: (n_amostras, n_variaveis)
        # ----------------------------------------------------
        elif isinstance(valores, np.ndarray) and valores.ndim == 2:
            contribuicoes = valores[posicao_linha]

        else:
            raise ValueError(f"Formato inesperado do SHAP: {type(valores)} / ndim={getattr(valores, 'ndim', 'N/A')}")

        # Garantir que fique 1D
        contribuicoes = np.ravel(contribuicoes)

        # Ajustar tamanho caso haja pequena diferença
        n = min(len(contribuicoes), len(nomes_variaveis))
        contribuicoes = contribuicoes[:n]
        nomes_variaveis = nomes_variaveis[:n]

        serie = pd.Series(contribuicoes, index=nomes_variaveis)
        serie = serie.sort_values(key=np.abs, ascending=False).head(12).iloc[::-1]

        fig, ax = plt.subplots(figsize=(8, 6))
        cores = ["#d62728" if v > 0 else "#1f77b4" for v in serie.values]

        ax.barh(serie.index, serie.values, color=cores)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title("Principais fatores que influenciaram o risco do posto")
        ax.set_xlabel("Contribuição para o risco")
        plt.tight_layout()

        return fig

    except Exception as e:
        st.warning(f"Não foi possível gerar a explicação individual do risco. Detalhe técnico: {e}")
        return None

with st.expander("Formato esperado do arquivo CSV", expanded=True):
    st.markdown("""Para que a ferramenta funcione corretamente, o arquivo CSV deve conter **no mínimo**:

### Campos obrigatórios
- **UTM_E_m**
- **UTM_N_m**
- **ZONA**

### Campos recomendados
- **IDADE**
- **QUANTIDADE DE TANQUES**
- **QUANTIDADE DE BOMBAS**
- **JAQUETADO**
- **MÉDIA DE IDADE DO TANQUE**
- **NÚMERO DE SONDAGENS**
- **ÁGUA SUBTERRÂNEA**
- **NIVEL_AGUA_m**

### Campos hidrogeológicos
- **dist_poco_mais_prox_m**
- **pocos_500m**
- **pocos_1km**
- **pocos_5km**

### Histórico e indicadores auxiliares
- **JÁ APRESENTOU CONTAMINAÇÃO ANTES**
- **CONC. BTEX**
- **CONC. PAH**""")
    st.download_button("Baixar modelo de CSV", data=construir_template_robusto().to_csv(index=False).encode("utf-8"),
                       file_name="template_georisco_goias.csv", mime="text/csv")

with st.expander("Explicação do modelo", expanded=False):
    st.markdown("""### Modelo utilizado
**Floresta aleatória**

### Objetivo
Estimar a probabilidade de contaminação ambiental em postos de combustíveis.

### Base de treinamento
O modelo foi treinado com dados de postos do estado de Goiás, combinando:
- dados operacionais;
- dados ambientais;
- variáveis hidrogeológicas;
- variáveis espaciais derivadas de poços.

### Como o resultado é interpretado
- **0 a 0,30** → baixo risco
- **0,30 a 0,60** → risco moderado
- **0,60 a 0,80** → alto risco
- **0,80 a 1,00** → risco muito alto""")

with st.expander("Importância das variáveis do modelo", expanded=False):
    st.markdown("""As variáveis com maior importância são aquelas que mais influenciam a separação entre casos de menor e maior risco.""")
    fig_imp = grafico_importancia_variaveis()
    if fig_imp is not None:
        st.pyplot(fig_imp, use_container_width=True)
    else:
        st.warning("Não foi possível carregar o arquivo de importância das variáveis.")

with st.expander("Interpretação do risco ambiental", expanded=False):
    st.markdown("""O risco calculado pela ferramenta **não substitui uma investigação ambiental detalhada**.

Ele funciona como uma **triagem técnica**, ajudando a priorizar áreas para vistoria, investigação, monitoramento e tomada de decisão.""")

st.sidebar.header("⚙️ Configurações")
arquivo_csv = st.sidebar.file_uploader("1) Arraste e solte o arquivo CSV dos postos aqui", type=["csv"])
arquivo_modelo = st.sidebar.file_uploader("2) (Opcional) Arraste aqui o modelo de previsão de risco", type=["joblib"])
st.sidebar.caption("O modelo de previsão de risco é um arquivo treinado com inteligência artificial. Ele calcula a probabilidade de contaminação ambiental com base nos dados do posto.")
tamanho_hexagono = st.sidebar.slider("3) Tamanho do hexágono (metros)", 2000, 10000, 5000, 500)
filtro_municipio = st.sidebar.text_input("4) Filtrar por município (opcional)")
risco_minimo = st.sidebar.slider("5) Risco mínimo para exibir", 0.0, 1.0, 0.0, 0.01)
executar = st.sidebar.button("Gerar análise", type="primary")

if arquivo_csv is None:
    st.info("Envie um CSV para começar.")
    st.stop()

try:
    df = ler_csv_seguro(arquivo_csv)
    st.success("CSV carregado com sucesso.")
except Exception as e:
    st.error(f"Erro ao ler o CSV: {e}")
    st.stop()

st.subheader("Pré-visualização da base")
st.dataframe(df.head(10), use_container_width=True)

faltando = [c for c in ["UTM_E_m", "UTM_N_m", "ZONA"] if c not in df.columns]
if faltando:
    st.error(f"O CSV precisa conter: UTM_E_m, UTM_N_m e ZONA. Faltando: {faltando}")
    st.stop()

try:
    df_geo = converter_utm_para_wgs84(df)
    goias_wgs, crs_origem_limite = carregar_limite_goias()
except Exception as e:
    st.error(str(e))
    st.stop()

if executar:
    modelo = None
    try:
        if arquivo_modelo is not None:
            modelo = joblib.load(arquivo_modelo)
            st.success("Modelo de previsão enviado carregado com sucesso.")
        elif ARQUIVO_MODELO_PADRAO.exists():
            modelo = joblib.load(ARQUIVO_MODELO_PADRAO)
            st.success("Modelo padrão carregado com sucesso.")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.stop()

    if modelo is not None:
        try:
            df_resultado, coluna_risco = aplicar_modelo(df_geo, modelo)
            st.success("Risco calculado pelo modelo com sucesso.")
        except Exception as e:
            st.error(f"Erro ao aplicar o modelo: {e}")
            st.stop()
    else:
        coluna_detectada = detectar_coluna_risco(df_geo)
        if coluna_detectada is None:
            st.error("Nenhum modelo foi carregado e o CSV não possui coluna de risco reconhecida.")
            st.stop()
        df_resultado = df_geo.copy()
        coluna_risco = coluna_detectada
        st.warning(f"Sem modelo. O sistema utilizará a coluna de risco existente: {coluna_risco}")

    if filtro_municipio and "CIDADE" in df_resultado.columns:
        df_resultado = df_resultado[df_resultado["CIDADE"].astype(str).str.contains(filtro_municipio, case=False, na=False)].copy()

    df_resultado = df_resultado[pd.to_numeric(df_resultado[coluna_risco], errors="coerce") >= risco_minimo].copy()
    gdf_pontos = gpd.GeoDataFrame(df_resultado.copy(), geometry=gpd.points_from_xy(df_resultado["lon"], df_resultado["lat"]), crs="EPSG:4326")

    try:
        grade_hexagonal = construir_grade_hexagonal(gdf_pontos, goias_wgs, coluna_risco, float(tamanho_hexagono))
        mapa = construir_mapa_interativo(df_resultado, goias_wgs, grade_hexagonal, coluna_risco, crs_origem_limite)
        mapa_html = mapa.get_root().render()
        mapa_risco_png, mapa_hotspots_png = gerar_mapas_estaticos(gdf_pontos, goias_wgs, grade_hexagonal, crs_origem_limite)
    except Exception as e:
        st.error(f"Erro ao gerar os produtos espaciais: {e}")
        st.stop()

    explicador = base_transformada = nomes_transformados = None
    if modelo is not None:
        try:
            explicador, base_transformada, nomes_transformados = montar_explicador(modelo, df_resultado)
        except Exception:
            pass

    st.session_state.analise_concluida = True
    st.session_state.dados_resultado = df_resultado
    st.session_state.coluna_risco = coluna_risco
    st.session_state.grade_hexagonal = grade_hexagonal
    st.session_state.mapa_html = mapa_html
    st.session_state.mapa_risco_png = mapa_risco_png
    st.session_state.mapa_hotspots_png = mapa_hotspots_png
    st.session_state.modelo_carregado = modelo
    st.session_state.explainer = explicador
    st.session_state.nomes_variaveis_transformadas = nomes_transformados
    st.session_state.base_transformada = base_transformada
    st.session_state.estatisticas = {
        "postos_validos": len(df_resultado),
        "hexagonos_com_postos": int(len(grade_hexagonal)),
        "hotspots": int(grade_hexagonal["hotspot"].sum()) if "hotspot" in grade_hexagonal.columns else 0,
        "risco_medio_geral": float(df_resultado[coluna_risco].mean()) if len(df_resultado) else 0.0
    }
    st.success("Análise concluída. Role a página para visualizar os mapas, explicações e downloads.")

if st.session_state.analise_concluida:
    df_resultado = st.session_state.dados_resultado
    coluna_risco = st.session_state.coluna_risco
    grade_hexagonal = st.session_state.grade_hexagonal
    mapa_html = st.session_state.mapa_html
    mapa_risco_png = st.session_state.mapa_risco_png
    mapa_hotspots_png = st.session_state.mapa_hotspots_png
    estatisticas = st.session_state.estatisticas

    st.subheader("Estatísticas principais")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Postos válidos", estatisticas["postos_validos"])
    c2.metric("Hexágonos com postos", estatisticas["hexagonos_com_postos"])
    c3.metric("Hotspots", estatisticas["hotspots"])
    c4.metric("Risco médio geral", f'{estatisticas["risco_medio_geral"]:.3f}')

    st.subheader("Mapa interativo de Goiás")
    st.components.v1.html(mapa_html, height=700, scrolling=True)

    st.subheader("Mapas estáticos com estrutura cartográfica")
    col1, col2 = st.columns(2)
    with col1:
        st.image(mapa_risco_png, caption="Mapa hexagonal de risco médio — Goiás", use_container_width=True)
    with col2:
        st.image(mapa_hotspots_png, caption="Hotspots de risco ambiental — Goiás", use_container_width=True)

    st.subheader("Explicação individual do risco")
    if st.session_state.explainer is not None and st.session_state.base_transformada is not None:
        opcoes = df_resultado.index.tolist()
        indice_escolhido = st.selectbox("Escolha o índice do posto para explicar", options=opcoes)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.dataframe(df_resultado.loc[[indice_escolhido]].drop(columns=["geometry"], errors="ignore"), use_container_width=True)
        with col_b:
            figura_local = grafico_explicacao_local(
                st.session_state.explainer,
                st.session_state.base_transformada,
                st.session_state.nomes_variaveis_transformadas,
                list(df_resultado.index).index(indice_escolhido)
            )
            if figura_local is not None:
                st.pyplot(figura_local, use_container_width=True)
            else:
                st.warning("Não foi possível gerar a explicação individual para este posto.")
    else:
        st.info("A explicação individual do risco está disponível quando o modelo é carregado corretamente.")

    st.subheader("Downloads")
    st.download_button("Baixar CSV com risco", data=df_resultado.to_csv(index=False).encode("utf-8"),
                       file_name="postos_scored_risk_app.csv", mime="text/csv")
    st.download_button("Baixar grade hexagonal (GeoJSON)", data=grade_hexagonal.to_json().encode("utf-8"),
                       file_name="hex_grid_risco_goias_app.geojson", mime="application/geo+json")
    st.download_button("Baixar mapa interativo (HTML)", data=mapa_html.encode("utf-8"),
                       file_name="mapa_interativo_georisco_goias.html", mime="text/html")
    with open(mapa_risco_png, "rb") as f:
        st.download_button("Baixar PNG (risco médio)", data=f.read(), file_name="mapa_hex_risco_goias.png", mime="image/png")
    with open(mapa_hotspots_png, "rb") as f:
        st.download_button("Baixar PNG (hotspots)", data=f.read(), file_name="mapa_hex_hotspots_goias.png", mime="image/png")

    st.subheader("Pré-visualização dos resultados")
    st.dataframe(df_resultado.head(20), use_container_width=True)
else:
    st.info("Clique em **Gerar análise** na barra lateral para rodar o processo.")

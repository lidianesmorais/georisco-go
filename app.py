#####Autor: Lidiane S Morais
#####Data: março de 2026

# ============================================================
# GeoRisco – Goiás
# Versão final com layout preservado
# ============================================================

import io
import math
import tempfile
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

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from pyproj import Transformer
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

st.set_page_config(page_title="GeoRisco – Goiás", page_icon="🗺️", layout="wide")

st.title("🗺️ GeoRisco – Goiás")
st.caption("Aplicação interativa para cálculo de risco ambiental em postos de combustíveis.")

st.info(
    """
Esta aplicação utiliza **Machine Learning e análise espacial** para estimar o
**risco ambiental de contaminação em postos de combustíveis no estado de Goiás**.

O sistema considera fatores como:
- presença de água subterrânea
- histórico de contaminação
- proximidade de poços de água
- características estruturais dos tanques
"""
)

DEFAULT_MODEL_PATH = Path("rf_model_com_hidro.joblib")
DEFAULT_BOUNDARY_PATH = Path("LimiteEstadoGO/1_Estado-Goias_SIRGAS_Poly.shp")

defaults = {
    "analysis_done": False,
    "df_scored": None,
    "risk_col": None,
    "hex_grid_wgs": None,
    "fmap_html": None,
    "png_risco": None,
    "png_hotspots": None,
    "stats": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def ler_planilha_segura(uploaded_file):
    nome = uploaded_file.name.lower()

    if nome.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        return df

    if nome.endswith(".geojson") or nome.endswith(".json"):
        gdf = gpd.read_file(uploaded_file)
        df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        return df

    raw = uploaded_file.getvalue()
    tentativas = [
        {"encoding": "utf-8", "sep": ";", "decimal": ","},
        {"encoding": "utf-8-sig", "sep": ";", "decimal": ","},
        {"encoding": "latin1", "sep": ";", "decimal": ","},
        {"encoding": "cp1252", "sep": ";", "decimal": ","},
        {"encoding": "utf-8", "sep": ",", "decimal": "."},
        {"encoding": "utf-8-sig", "sep": ",", "decimal": "."},
        {"encoding": "latin1", "sep": ",", "decimal": "."},
        {"encoding": "cp1252", "sep": ",", "decimal": "."},
    ]

    ultimo_erro = None
    for tentativa in tentativas:
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                encoding=tentativa["encoding"],
                sep=tentativa["sep"],
                decimal=tentativa["decimal"]
            )
            if df.shape[1] > 1:
                df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
                return df
        except Exception as e:
            ultimo_erro = e

    raise ValueError(
        "Não foi possível ler o arquivo enviado. "
        "Use preferencialmente CSV com separador ';' ou arquivo Excel .xlsx. "
        f"Detalhe técnico: {ultimo_erro}"
    )


def normalizar_nome_coluna(nome):
    return (
        str(nome)
        .strip()
        .upper()
        .replace("Á", "A").replace("À", "A").replace("Ã", "A").replace("Â", "A")
        .replace("É", "E").replace("Ê", "E")
        .replace("Í", "I")
        .replace("Ó", "O").replace("Ô", "O").replace("Õ", "O")
        .replace("Ú", "U")
        .replace("Ç", "C")
    )


def diagnosticar_colunas(df):
    mapa = {normalizar_nome_coluna(c): c for c in df.columns}
    grupos = {
        "obrigatorios": ["UTM_E_M", "UTM_N_M", "ZONA"],
        "recomendados": [
            "IDADE", "QUANTIDADE DE TANQUES", "QUANTIDADE DE BOMBAS", "JAQUETADO",
            "MEDIA DE IDADE DO TANQUE", "NUMERO DE SONDAGENS", "AGUA SUBTERRANEA", "NIVEL_AGUA_M",
        ],
        "hidrogeologicos": ["DIST_POCO_MAIS_PROX_M", "POCOS_500M", "POCOS_1KM", "POCOS_5KM"],
        "auxiliares": ["JA APRESENTOU CONTAMINACAO ANTES", "CONC. BTEX", "CONC. PAH"],
    }
    resultado = {}
    for grupo, campos in grupos.items():
        encontrados, faltando = [], []
        for campo in campos:
            if campo in mapa:
                encontrados.append(mapa[campo])
            else:
                faltando.append(campo)
        resultado[grupo] = {"encontrados": encontrados, "faltando": faltando}
    return resultado


def detect_risk_column(df):
    for c in ["RISK_PROBA_HIDRO", "RISK_PROBA", "risco", "risco_medio"]:
        if c in df.columns:
            return c
    return None


def convert_utm_to_wgs84(df, e_col="UTM_E_m", n_col="UTM_N_m", zone_col="ZONA"):
    work = df.copy()
    work[e_col] = pd.to_numeric(work[e_col], errors="coerce")
    work[n_col] = pd.to_numeric(work[n_col], errors="coerce")
    work[zone_col] = work[zone_col].astype(str).str.strip().str.upper()
    work = work.dropna(subset=[e_col, n_col, zone_col]).copy()

    lons, lats = [], []
    for _, row in work.iterrows():
        zona_num = "".join(ch for ch in str(row[zone_col]) if ch.isdigit())
        zona_num = int(zona_num) if zona_num else 22
        transformer = Transformer.from_crs(f"EPSG:{32700 + zona_num}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(row[e_col], row[n_col])
        lons.append(lon)
        lats.append(lat)

    work["lon"] = lons
    work["lat"] = lats
    return work


def load_goias_boundary():
    if not DEFAULT_BOUNDARY_PATH.exists():
        raise FileNotFoundError(f"Não encontrei o shapefile em: {DEFAULT_BOUNDARY_PATH}")
    goias = gpd.read_file(DEFAULT_BOUNDARY_PATH)
    original_crs = str(goias.crs)
    goias = goias.to_crs(epsg=4326)
    return goias, original_crs


def get_model_expected_columns(model):
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass
    try:
        pre = model.named_steps["preprocess"]
        expected = []
        for _, _, cols in pre.transformers:
            if cols == "drop":
                continue
            if isinstance(cols, list):
                expected.extend(cols)
        return list(dict.fromkeys(expected))
    except Exception:
        pass
    return None


def score_with_model(df, model):
    work = df.copy()
    expected = get_model_expected_columns(model)
    if expected is None:
        X = work.copy()
    else:
        for col in expected:
            if col not in work.columns:
                work[col] = np.nan
        X = work[expected].copy()
    proba = model.predict_proba(X)[:, 1]
    work["RISK_PROBA_HIDRO"] = proba
    return work, "RISK_PROBA_HIDRO"


def create_hexagon(center_x, center_y, size):
    return Polygon([
        (
            center_x + size * math.cos(math.radians(angle)),
            center_y + size * math.sin(math.radians(angle))
        )
        for angle in range(0, 360, 60)
    ])


def build_hex_grid(gdf_points_wgs, boundary_wgs, risk_col, hex_size=5000.0):
    gdf_m = gdf_points_wgs.to_crs(epsg=3857)
    boundary_m = boundary_wgs.to_crs(epsg=3857)

    minx, miny, maxx, maxy = gdf_m.total_bounds
    minx -= hex_size
    miny -= hex_size
    maxx += hex_size
    maxy += hex_size

    dx = 1.5 * hex_size
    dy = math.sqrt(3) * hex_size

    hexagons, ids = [], []
    col = 0
    x = minx
    while x < maxx:
        y_offset = 0 if col % 2 == 0 else dy / 2
        y = miny + y_offset
        row = 0
        while y < maxy:
            hexagons.append(create_hexagon(x, y, hex_size))
            ids.append(f"hex_{col}_{row}")
            y += dy
            row += 1
        x += dx
        col += 1

    hex_grid = gpd.GeoDataFrame({"hex_id": ids}, geometry=hexagons, crs=gdf_m.crs)
    hex_grid = gpd.overlay(hex_grid, boundary_m, how="intersection")

    joined = gpd.sjoin(gdf_m, hex_grid, how="left", predicate="within")
    hex_stats = joined.groupby("hex_id").agg(
        postos=(risk_col, "count"),
        risco_medio=(risk_col, "mean"),
        risco_max=(risk_col, "max")
    ).reset_index()

    hex_grid = hex_grid.merge(hex_stats, on="hex_id", how="left")
    hex_grid["postos"] = hex_grid["postos"].fillna(0)
    hex_grid["risco_medio"] = hex_grid["risco_medio"].fillna(0)
    hex_grid["risco_max"] = hex_grid["risco_max"].fillna(0)

    hex_used = hex_grid[hex_grid["postos"] > 0].copy()
    if len(hex_used) > 0:
        threshold = hex_used["risco_medio"].quantile(0.75)
        hex_used["hotspot"] = np.where(hex_used["risco_medio"] >= threshold, 1, 0)
    else:
        hex_used["hotspot"] = 0

    return hex_used.to_crs(epsg=4326)


def add_north_arrow(ax, x=0.95, y=0.12):
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - 0.10),
        arrowprops=dict(facecolor="black", width=3, headwidth=10),
        ha="center",
        va="center",
        fontsize=12,
        xycoords=ax.transAxes
    )


def build_interactive_map(df_points, boundary_wgs, hex_grid_wgs, risk_col, original_boundary_crs):
    center_lat = float(df_points["lat"].mean())
    center_lon = float(df_points["lon"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron", control_scale=True)

    folium.GeoJson(
        boundary_wgs,
        name="Limite de Goiás",
        style_function=lambda x: {"color": "black", "weight": 2, "fillOpacity": 0},
        tooltip="Estado de Goiás"
    ).add_to(m)

    vmin = float(hex_grid_wgs["risco_medio"].min()) if len(hex_grid_wgs) else 0.0
    vmax = float(hex_grid_wgs["risco_medio"].max()) if len(hex_grid_wgs) else 1.0
    if vmax == vmin:
        vmax = vmin + 0.0001

    colormap = cm.LinearColormap(
        colors=["#440154", "#31688e", "#35b779", "#fde725"],
        vmin=vmin,
        vmax=vmax,
        caption="Risco médio por hexágono"
    )

    folium.GeoJson(
        hex_grid_wgs,
        name="Grade hexagonal de risco",
        style_function=lambda feature: {
            "fillColor": colormap(feature["properties"].get("risco_medio", 0)),
            "color": "black", "weight": 0.5, "fillOpacity": 0.55
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[c for c in ["hex_id", "postos", "risco_medio", "risco_max", "hotspot"] if c in hex_grid_wgs.columns],
            aliases=["Hexágono", "Nº de postos", "Risco médio", "Risco máximo", "Hotspot"]
        )
    ).add_to(m)
    colormap.add_to(m)

    if "hotspot" in hex_grid_wgs.columns:
        hotspots = hex_grid_wgs[hex_grid_wgs["hotspot"] == 1].copy()
        if len(hotspots):
            folium.GeoJson(
                hotspots,
                name="Hotspots",
                style_function=lambda x: {"fillColor": "#ff0000", "color": "#8b0000", "weight": 1.2, "fillOpacity": 0.30},
                tooltip=folium.GeoJsonTooltip(
                    fields=[c for c in ["hex_id", "risco_medio", "hotspot"] if c in hotspots.columns],
                    aliases=["Hexágono", "Risco médio", "Hotspot"]
                )
            ).add_to(m)

    marker_cluster = plugins.MarkerCluster(name="Postos").add_to(m)
    for _, row in df_points.iterrows():
        risco = float(row[risk_col])
        popup_lines = [
            f"<b>Posto:</b> {row.get('NÚMERO DO POSTO', 'N/D')}",
            f"<b>Cidade:</b> {row.get('CIDADE', 'N/D')}",
            f"<b>Risco:</b> {risco:.3f}",
        ]
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4 + 10 * risco,
            color="#d62728",
            fill=True,
            fill_color="#d62728",
            fill_opacity=0.75,
            weight=1,
            popup=folium.Popup("<br>".join(popup_lines), max_width=350),
            tooltip=f"{row.get('CIDADE', 'Posto')} | risco {risco:.3f}"
        ).add_to(marker_cluster)

    heat_data = df_points[["lat", "lon", risk_col]].values.tolist()
    plugins.HeatMap(heat_data, name="Heatmap dos postos", radius=25, blur=18, min_opacity=0.30).add_to(m)
    plugins.MousePosition(position="bottomleft", separator=" | ", prefix="Coordenadas", num_digits=6).add_to(m)
    plugins.MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    date_text = datetime.now().strftime("%d/%m/%Y %H:%M")
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position: fixed; bottom: 85px; left: 10px; z-index: 9999;
                background-color: white; padding: 8px 10px; border: 1px solid #777;
                border-radius: 4px; font-size: 12px;">
        <b>Data de geração:</b><br>{date_text}
    </div>"""))

    m.get_root().html.add_child(folium.Element(f"""
    <div style="position: fixed; bottom: 10px; left: 10px; z-index: 9999;
                background-color: white; padding: 8px 10px; border: 1px solid #777;
                border-radius: 4px; font-size: 12px;">
        <b>Datum / CRS:</b><br>
        Web map: WGS84 (EPSG:4326)<br>
        Limite GO (origem): {original_boundary_crs}
    </div>"""))

    m.get_root().html.add_child(folium.Element("""
    <div style="position: fixed; top: 10px; right: 10px; z-index: 9999;
                background-color: white; width: 62px; text-align: center;
                padding: 6px 4px; border: 1px solid #777; border-radius: 4px; font-size: 12px;">
        <div style="font-weight: bold;">N</div>
        <div style="font-size: 24px; line-height: 22px;">↑</div>
        <div style="font-size: 11px;">Norte</div>
    </div>"""))

    return m


def make_static_maps(df_points_wgs, boundary_wgs, hex_grid_wgs, original_boundary_crs):
    now_text = datetime.now().strftime("%d/%m/%Y %H:%M")
    postos = df_points_wgs.copy()
    goias = boundary_wgs.copy()
    hexes = hex_grid_wgs.copy()

    def make_one(column, title, point_color, cmap):
        fig, ax = plt.subplots(figsize=(10, 10))
        goias.boundary.plot(ax=ax, linewidth=1.8, color="black")
        if len(hexes):
            hexes.plot(column=column, legend=True, ax=ax, cmap=cmap, edgecolor="black", linewidth=0.3, alpha=0.7)
        postos.plot(ax=ax, markersize=10, color=point_color)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Longitude (graus)")
        ax.set_ylabel("Latitude (graus)")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
        add_north_arrow(ax)
        ax.add_artist(AnchoredSizeBar(
            ax.transData, 1.0, "Escala indicativa", "lower right", pad=0.4,
            color="black", frameon=True, size_vertical=0.01,
            fontproperties=fm.FontProperties(size=9)
        ))
        ax.text(
            0.01, 0.01,
            f"""Gerado em: {now_text}
Datum do mapa: WGS84 (EPSG:4326)
Limite GO (origem): {original_boundary_crs}""",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
        )
        fig.tight_layout()
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(out.name, dpi=220)
        plt.close(fig)
        return out.name

    return (
        make_one("risco_medio", "Mapa hexagonal de risco médio — Goiás", "red", "viridis"),
        make_one("hotspot", "Hotspots de risco ambiental — Goiás", "blue", "Reds")
    )


def gerar_template():
    return pd.DataFrame({
        "NÚMERO DO POSTO": [1, 2],
        "UTM_E_m": [686304.36, 687577.99],
        "UTM_N_m": [8216649.19, 8158767.99],
        "ZONA": ["22K", "22K"],
        "CIDADE": ["GOIANIA", "GOIANIA"],
        "IDADE": [20, 15],
        "QUANTIDADE DE TANQUES": [4, 3],
        "QUANTIDADE DE BOMBAS": [8, 6],
        "NIVEL_AGUA_m": [10, 12]
    })


with st.expander("📂 Formato esperado do arquivo CSV", expanded=True):
    st.markdown("""
Para que a ferramenta funcione corretamente, o arquivo enviado deve conter **no mínimo**:

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
- **CONC. PAH**

### Regras da versão beta
- Se os **campos obrigatórios** existirem, o sistema pode continuar.
- Se os **campos recomendados** existirem, a previsão tende a ser melhor.
- Se os **campos hidrogeológicos** existirem, a análise fica mais alinhada ao estudo original.
- Se houver **colunas extras**, elas são aceitas e ignoradas quando não forem necessárias.

### Arquivos aceitos
- **CSV**
- **Excel (.xlsx)**
- **GeoJSON**
""")
    st.download_button(
        "⬇️ Baixar modelo de CSV",
        data=gerar_template().to_csv(index=False).encode("utf-8"),
        file_name="template_postos.csv",
        mime="text/csv"
    )

with st.expander("🧠 Explicação do modelo", expanded=False):
    st.markdown("""
O sistema utiliza um modelo de **Machine Learning** treinado com dados de postos de combustíveis.

Ele considera informações como:
- idade do posto e dos tanques
- quantidade de tanques e bombas
- presença de água subterrânea
- histórico de contaminação
- proximidade de poços de água
- concentração espacial de poços em diferentes distâncias
""")

with st.expander("📊 Importância das variáveis do modelo", expanded=False):
    st.markdown("""
As variáveis com maior importância são aquelas que mais influenciam a separação entre casos de menor e maior risco.
""")

with st.expander("⚠️ Interpretação do risco ambiental", expanded=False):
    st.markdown("""
O risco calculado pela ferramenta **não substitui uma investigação ambiental detalhada**.
""")

st.sidebar.header("⚙️ Configurações")

uploaded_csv = st.sidebar.file_uploader(
    "1) Arraste e solte o arquivo CSV dos postos aqui",
    type=["csv", "xlsx", "geojson", "json"]
)

uploaded_model = st.sidebar.file_uploader(
    "2) (Opcional) Arraste aqui o modelo de previsão de risco",
    type=["joblib"]
)

st.sidebar.caption("""
O modelo de previsão de risco é um arquivo treinado com inteligência artificial. Ele
calcula a probabilidade de contaminação ambiental com base nos dados do posto.
""")

hex_size = st.sidebar.slider("3) Tamanho do hexágono (metros)", 2000, 10000, 5000, 500)
municipio_filter = st.sidebar.text_input("4) Filtrar por município (opcional)")
risk_min = st.sidebar.slider("5) Risco mínimo para exibir", 0.0, 1.0, 0.0, 0.01)
run_button = st.sidebar.button("▶️ Gerar análise", type="primary")

if uploaded_csv is None:
    st.info("Envie um CSV, Excel ou GeoJSON para começar.")
    st.stop()

try:
    df = ler_planilha_segura(uploaded_csv)
    st.success("Arquivo carregado com sucesso.")
except Exception as e:
    st.error(f"Erro ao ler o arquivo: {e}")
    st.stop()

st.subheader("📋 Pré-visualização da base")
st.dataframe(df.head(10), use_container_width=True)

diag = diagnosticar_colunas(df)
if len(diag["obrigatorios"]["faltando"]) > 0:
    st.error(f"Campos obrigatórios ausentes: {diag['obrigatorios']['faltando']}")
    st.stop()

with st.expander("🔎 Diagnóstico da base enviada", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Obrigatórios", f"{len(diag['obrigatorios']['encontrados'])}/3")
    c2.metric("Recomendados", f"{len(diag['recomendados']['encontrados'])}/8")
    c3.metric("Hidrogeológicos", f"{len(diag['hidrogeologicos']['encontrados'])}/4")
    c4.metric("Auxiliares", f"{len(diag['auxiliares']['encontrados'])}/3")

try:
    df_geo = convert_utm_to_wgs84(df)
except Exception as e:
    st.error(f"Erro na conversão UTM -> WGS84: {e}")
    st.stop()

try:
    goias_wgs, boundary_original_crs = load_goias_boundary()
except Exception as e:
    st.error(f"Erro ao carregar o limite de Goiás: {e}")
    st.stop()

if run_button:
    model = None
    try:
        if uploaded_model is not None:
            model = joblib.load(uploaded_model)
            st.success("Modelo de previsão enviado carregado com sucesso.")
        elif DEFAULT_MODEL_PATH.exists():
            model = joblib.load(DEFAULT_MODEL_PATH)
            st.success(f"Modelo local carregado: {DEFAULT_MODEL_PATH}")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.stop()

    if model is not None:
        try:
            df_scored, risk_col = score_with_model(df_geo, model)
            st.success("Risco calculado pelo modelo com sucesso.")
        except Exception as e:
            st.error(f"Erro ao aplicar o modelo: {e}")
            st.stop()
    else:
        detected = detect_risk_column(df_geo)
        if detected is None:
            st.error("Nenhum modelo foi carregado e a base não possui coluna de risco reconhecida.")
            st.stop()
        df_scored = df_geo.copy()
        risk_col = detected
        st.warning(f"Sem modelo. Usando a coluna de risco existente: {risk_col}")

    if municipio_filter and "CIDADE" in df_scored.columns:
        df_scored = df_scored[df_scored["CIDADE"].astype(str).str.contains(municipio_filter, case=False, na=False)].copy()

    df_scored = df_scored[pd.to_numeric(df_scored[risk_col], errors="coerce") >= risk_min].copy()
    gdf_points_wgs = gpd.GeoDataFrame(
        df_scored.copy(),
        geometry=gpd.points_from_xy(df_scored["lon"], df_scored["lat"]),
        crs="EPSG:4326"
    )

    try:
        hex_grid_wgs = build_hex_grid(gdf_points_wgs=gdf_points_wgs, boundary_wgs=goias_wgs, risk_col=risk_col, hex_size=float(hex_size))
        fmap = build_interactive_map(df_scored, goias_wgs, hex_grid_wgs, risk_col, boundary_original_crs)
        fmap_html = fmap.get_root().render()
        png_risco, png_hotspots = make_static_maps(gdf_points_wgs, goias_wgs, hex_grid_wgs, boundary_original_crs)
    except Exception as e:
        st.error(f"Erro ao gerar produtos espaciais: {e}")
        st.stop()

    st.session_state.analysis_done = True
    st.session_state.df_scored = df_scored
    st.session_state.risk_col = risk_col
    st.session_state.hex_grid_wgs = hex_grid_wgs
    st.session_state.fmap_html = fmap_html
    st.session_state.png_risco = png_risco
    st.session_state.png_hotspots = png_hotspots
    st.session_state.stats = {
        "postos_validos": len(df_scored),
        "hexagonos_com_postos": int(len(hex_grid_wgs)),
        "hotspots": int(hex_grid_wgs["hotspot"].sum()) if "hotspot" in hex_grid_wgs.columns else 0,
        "risco_medio_geral": float(df_scored[risk_col].mean()) if len(df_scored) else 0.0
    }

    st.success("Análise concluída. Role a página para visualizar os mapas e downloads.")

if st.session_state.analysis_done:
    df_scored = st.session_state.df_scored
    risk_col = st.session_state.risk_col
    hex_grid_wgs = st.session_state.hex_grid_wgs
    fmap_html = st.session_state.fmap_html
    png_risco = st.session_state.png_risco
    png_hotspots = st.session_state.png_hotspots
    stats = st.session_state.stats

    st.subheader("📈 Estatísticas principais")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Postos válidos", stats["postos_validos"])
    c2.metric("Hexágonos com postos", stats["hexagonos_com_postos"])
    c3.metric("Hotspots", stats["hotspots"])
    c4.metric("Risco médio geral", f"{stats['risco_medio_geral']:.3f}")

    st.subheader("🗺️ Mapa interativo de Goiás")
    st.components.v1.html(fmap_html, height=700, scrolling=True)

    st.subheader("🖼️ Mapas estáticos com estrutura cartográfica")
    col1, col2 = st.columns(2)
    with col1:
        st.image(png_risco, caption="Mapa hexagonal de risco médio — Goiás", use_container_width=True)
    with col2:
        st.image(png_hotspots, caption="Hotspots de risco ambiental — Goiás", use_container_width=True)

    st.subheader("📦 Downloads")
    st.download_button("⬇️ Baixar CSV com risco", data=df_scored.to_csv(index=False).encode("utf-8"),
                       file_name="postos_scored_risk_app.csv", mime="text/csv")
    st.download_button("⬇️ Baixar grade hexagonal (GeoJSON)", data=hex_grid_wgs.to_json().encode("utf-8"),
                       file_name="hex_grid_risco_goias_app.geojson", mime="application/geo+json")
    st.download_button("⬇️ Baixar mapa interativo (HTML)", data=fmap_html.encode("utf-8"),
                       file_name="mapa_interativo_goias_app.html", mime="text/html")
    with open(png_risco, "rb") as f:
        st.download_button("⬇️ Baixar PNG (risco médio)", data=f.read(), file_name="mapa_hex_risco_goias.png", mime="image/png")
    with open(png_hotspots, "rb") as f:
        st.download_button("⬇️ Baixar PNG (hotspots)", data=f.read(), file_name="mapa_hex_hotspots_goias.png", mime="image/png")

    st.subheader("📋 Pré-visualização dos resultados")
    st.dataframe(df_scored.head(20), use_container_width=True)
else:
    st.info("Clique em **Gerar análise** na barra lateral para rodar o processo.")

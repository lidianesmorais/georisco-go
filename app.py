# GeoRisco – Goiás | Versão 3
import io, math, tempfile
from pathlib import Path
from datetime import datetime
import joblib, numpy as np, pandas as pd, geopandas as gpd, streamlit as st, folium, branca.colormap as cm
from folium import plugins
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from pyproj import Transformer
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

st.set_page_config(page_title="GeoRisco – Goiás", page_icon="🗺️", layout="wide")
st.title("🗺️ GeoRisco – Goiás")
st.caption("Aplicação interativa para cálculo de risco ambiental em postos de combustíveis.")
st.info("""Esta aplicação utiliza **Machine Learning e análise espacial** para estimar o **risco ambiental de contaminação em postos de combustíveis no estado de Goiás**.

O sistema considera fatores como:
- presença de água subterrânea
- histórico de contaminação
- proximidade de poços de água
- características estruturais dos tanques""")

DEFAULT_MODEL_PATH = Path("rf_model_com_hidro.joblib")
DEFAULT_BOUNDARY_PATH = Path("LimiteEstadoGO/1_Estado-Goias_SIRGAS_Poly.shp")
for k, v in {"analysis_done":False,"df_scored":None,"risk_col":None,"hex_grid_wgs":None,"fmap_html":None,"png_risco":None,"png_hotspots":None,"stats":None}.items():
    if k not in st.session_state: st.session_state[k]=v

def safe_read_csv(uploaded_file):
    raw = uploaded_file.getvalue()
    for enc in ("utf-8","latin1","cp1252"):
        try: return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception: pass
    raise ValueError("Não foi possível ler o CSV. Tente salvar o arquivo em UTF-8.")

def detect_risk_column(df):
    for c in ["RISK_PROBA_HIDRO","RISK_PROBA","risco","risco_medio"]:
        if c in df.columns: return c
    return None

def convert_utm_to_wgs84(df,e_col="UTM_E_m",n_col="UTM_N_m",zone_col="ZONA"):
    work=df.copy()
    work[e_col]=pd.to_numeric(work[e_col],errors="coerce")
    work[n_col]=pd.to_numeric(work[n_col],errors="coerce")
    work[zone_col]=work[zone_col].astype(str).str.strip().str.upper()
    work=work.dropna(subset=[e_col,n_col,zone_col]).copy()
    lons,lats=[],[]
    for _,row in work.iterrows():
        zona_num="".join(ch for ch in str(row[zone_col]) if ch.isdigit())
        zona_num=int(zona_num) if zona_num else 22
        transformer=Transformer.from_crs(f"EPSG:{32700+zona_num}","EPSG:4326",always_xy=True)
        lon,lat=transformer.transform(row[e_col],row[n_col]); lons.append(lon); lats.append(lat)
    work["lon"]=lons; work["lat"]=lats
    return work

def load_goias_boundary():
    if not DEFAULT_BOUNDARY_PATH.exists(): raise FileNotFoundError(f"Não encontrei o shapefile em: {DEFAULT_BOUNDARY_PATH}")
    goias=gpd.read_file(DEFAULT_BOUNDARY_PATH); original_crs=str(goias.crs); goias=goias.to_crs(epsg=4326)
    return goias, original_crs

def get_model_expected_columns(model):
    try:
        if hasattr(model,"feature_names_in_"): return list(model.feature_names_in_)
    except Exception: pass
    try:
        pre=model.named_steps["preprocess"]; expected=[]
        for _,_,cols in pre.transformers:
            if cols=="drop": continue
            if isinstance(cols,list): expected.extend(cols)
        return list(dict.fromkeys(expected))
    except Exception: return None

def score_with_model(df,model):
    work=df.copy(); expected=get_model_expected_columns(model)
    if expected is None: X=work.copy()
    else:
        for col in expected:
            if col not in work.columns: work[col]=np.nan
        X=work[expected].copy()
    work["RISK_PROBA_HIDRO"]=model.predict_proba(X)[:,1]
    return work,"RISK_PROBA_HIDRO"

def create_hexagon(center_x,center_y,size):
    return Polygon([(center_x+size*math.cos(math.radians(a)), center_y+size*math.sin(math.radians(a))) for a in range(0,360,60)])

def build_hex_grid(gdf_points_wgs,boundary_wgs,risk_col,hex_size=5000.0):
    gdf_m=gdf_points_wgs.to_crs(epsg=3857); boundary_m=boundary_wgs.to_crs(epsg=3857)
    minx,miny,maxx,maxy=gdf_m.total_bounds; minx-=hex_size; miny-=hex_size; maxx+=hex_size; maxy+=hex_size
    dx=1.5*hex_size; dy=math.sqrt(3)*hex_size
    hexagons,ids=[],[]; col=0; x=minx
    while x<maxx:
        y=miny+(0 if col%2==0 else dy/2); row=0
        while y<maxy:
            hexagons.append(create_hexagon(x,y,hex_size)); ids.append(f"hex_{col}_{row}"); y+=dy; row+=1
        x+=dx; col+=1
    hex_grid=gpd.GeoDataFrame({"hex_id":ids},geometry=hexagons,crs=gdf_m.crs)
    hex_grid=gpd.overlay(hex_grid,boundary_m,how="intersection")
    joined=gpd.sjoin(gdf_m,hex_grid,how="left",predicate="within")
    stats=joined.groupby("hex_id").agg(postos=(risk_col,"count"),risco_medio=(risk_col,"mean"),risco_max=(risk_col,"max")).reset_index()
    hex_grid=hex_grid.merge(stats,on="hex_id",how="left")
    for c in ["postos","risco_medio","risco_max"]: hex_grid[c]=hex_grid[c].fillna(0)
    hex_used=hex_grid[hex_grid["postos"]>0].copy()
    hex_used["hotspot"]=np.where(hex_used["risco_medio"]>=hex_used["risco_medio"].quantile(0.75),1,0) if len(hex_used) else 0
    return hex_used.to_crs(epsg=4326)

def add_north_arrow(ax,x=0.95,y=0.12):
    ax.annotate("N",xy=(x,y),xytext=(x,y-0.10),arrowprops=dict(facecolor="black",width=3,headwidth=10),ha="center",va="center",fontsize=12,xycoords=ax.transAxes)

def build_interactive_map(df_points,boundary_wgs,hex_grid_wgs,risk_col,original_boundary_crs):
    m=folium.Map(location=[float(df_points["lat"].mean()),float(df_points["lon"].mean())],zoom_start=6,tiles="CartoDB positron",control_scale=True)
    folium.GeoJson(boundary_wgs,name="Limite de Goiás",style_function=lambda x: {"color":"black","weight":2,"fillOpacity":0},tooltip="Estado de Goiás").add_to(m)
    vmin=float(hex_grid_wgs["risco_medio"].min()) if len(hex_grid_wgs) else 0.0; vmax=float(hex_grid_wgs["risco_medio"].max()) if len(hex_grid_wgs) else 1.0
    if vmax==vmin: vmax=vmin+0.0001
    colormap=cm.LinearColormap(colors=["#440154","#31688e","#35b779","#fde725"],vmin=vmin,vmax=vmax,caption="Risco médio por hexágono")
    folium.GeoJson(hex_grid_wgs,name="Grade hexagonal de risco",
        style_function=lambda f: {"fillColor":colormap(f["properties"].get("risco_medio",0)),"color":"black","weight":0.5,"fillOpacity":0.55},
        tooltip=folium.GeoJsonTooltip(fields=[c for c in ["hex_id","postos","risco_medio","risco_max","hotspot"] if c in hex_grid_wgs.columns],
                                      aliases=["Hexágono","Nº de postos","Risco médio","Risco máximo","Hotspot"])).add_to(m)
    colormap.add_to(m)
    if "hotspot" in hex_grid_wgs.columns:
        hotspots=hex_grid_wgs[hex_grid_wgs["hotspot"]==1].copy()
        if len(hotspots):
            folium.GeoJson(hotspots,name="Hotspots",
                style_function=lambda x: {"fillColor":"#ff0000","color":"#8b0000","weight":1.2,"fillOpacity":0.30},
                tooltip=folium.GeoJsonTooltip(fields=[c for c in ["hex_id","risco_medio","hotspot"] if c in hotspots.columns],
                                              aliases=["Hexágono","Risco médio","Hotspot"])).add_to(m)
    marker_cluster=plugins.MarkerCluster(name="Postos").add_to(m)
    for _,row in df_points.iterrows():
        risco=float(row[risk_col]); popup=[f"<b>Posto:</b> {row.get('NÚMERO DO POSTO','N/D')}",f"<b>Cidade:</b> {row.get('CIDADE','N/D')}",f"<b>Risco:</b> {risco:.3f}"]
        for c,label in [("dist_poco_mais_prox_m","Dist. poço mais próximo (m)"),("pocos_500m","Poços em 500 m"),("pocos_1km","Poços em 1 km"),("pocos_5km","Poços em 5 km")]:
            if c in row.index: popup.append(f"<b>{label}:</b> {row[c]}")
        folium.CircleMarker(location=[row["lat"],row["lon"]],radius=4+10*risco,color="#d62728",fill=True,fill_color="#d62728",fill_opacity=0.75,weight=1,
            popup=folium.Popup("<br>".join(popup),max_width=350),tooltip=f"{row.get('CIDADE','Posto')} | risco {risco:.3f}").add_to(marker_cluster)
    plugins.HeatMap(df_points[["lat","lon",risk_col]].values.tolist(),name="Heatmap dos postos",radius=25,blur=18,min_opacity=0.30).add_to(m)
    plugins.MousePosition(position="bottomleft",separator=" | ",prefix="Coordenadas",num_digits=6).add_to(m)
    plugins.MiniMap(toggle_display=True).add_to(m); folium.LayerControl(collapsed=False).add_to(m)
    date_text=datetime.now().strftime("%d/%m/%Y %H:%M")
    m.get_root().html.add_child(folium.Element(f'<div style="position: fixed; bottom: 85px; left: 10px; z-index: 9999; background-color: white; padding: 8px 10px; border: 1px solid #777; border-radius: 4px; font-size: 12px;"><b>Data de geração:</b><br>{date_text}</div>'))
    m.get_root().html.add_child(folium.Element(f'<div style="position: fixed; bottom: 10px; left: 10px; z-index: 9999; background-color: white; padding: 8px 10px; border: 1px solid #777; border-radius: 4px; font-size: 12px;"><b>Datum / CRS:</b><br>Web map: WGS84 (EPSG:4326)<br>Limite GO (origem): {original_boundary_crs}</div>'))
    m.get_root().html.add_child(folium.Element('<div style="position: fixed; top: 10px; right: 10px; z-index: 9999; background-color: white; width: 62px; text-align: center; padding: 6px 4px; border: 1px solid #777; border-radius: 4px; font-size: 12px;"><div style="font-weight: bold;">N</div><div style="font-size: 24px; line-height: 22px;">↑</div><div style="font-size: 11px;">Norte</div></div>'))
    return m

def make_static_maps(df_points_wgs,boundary_wgs,hex_grid_wgs,original_boundary_crs):
    now_text=datetime.now().strftime("%d/%m/%Y %H:%M")
    postos,goias,hexes=df_points_wgs.copy(),boundary_wgs.copy(),hex_grid_wgs.copy()
    fig,ax=plt.subplots(figsize=(10,10)); goias.boundary.plot(ax=ax,linewidth=1.8,color="black")
    if len(hexes): hexes.plot(column="risco_medio",legend=True,ax=ax,cmap="viridis",edgecolor="black",linewidth=0.3,alpha=0.7)
    postos.plot(ax=ax,markersize=10,color="red"); ax.set_title("Mapa hexagonal de risco médio — Goiás",fontsize=14); ax.set_xlabel("Longitude (graus)"); ax.set_ylabel("Latitude (graus)"); ax.grid(True,linestyle="--",linewidth=0.4,alpha=0.4); add_north_arrow(ax)
    ax.add_artist(AnchoredSizeBar(ax.transData,1.0,"Escala indicativa","lower right",pad=0.4,color="black",frameon=True,size_vertical=0.01,fontproperties=fm.FontProperties(size=9)))
    ax.text(0.01,0.01,f"""Gerado em: {now_text}
Datum do mapa: WGS84 (EPSG:4326)
Limite GO (origem): {original_boundary_crs}""",transform=ax.transAxes,fontsize=9,verticalalignment="bottom",bbox=dict(facecolor="white",alpha=0.8,edgecolor="gray"))
    fig.tight_layout(); out_risco=tempfile.NamedTemporaryFile(delete=False,suffix=".png"); fig.savefig(out_risco.name,dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,10)); goias.boundary.plot(ax=ax,linewidth=1.8,color="black")
    if len(hexes): hexes.plot(column="hotspot",legend=True,ax=ax,cmap="Reds",edgecolor="black",linewidth=0.3,alpha=0.6)
    postos.plot(ax=ax,markersize=10,color="blue"); ax.set_title("Hotspots de risco ambiental — Goiás",fontsize=14); ax.set_xlabel("Longitude (graus)"); ax.set_ylabel("Latitude (graus)"); ax.grid(True,linestyle="--",linewidth=0.4,alpha=0.4); add_north_arrow(ax)
    ax.add_artist(AnchoredSizeBar(ax.transData,1.0,"Escala indicativa","lower right",pad=0.4,color="black",frameon=True,size_vertical=0.01,fontproperties=fm.FontProperties(size=9)))
    ax.text(0.01,0.01,f"""Gerado em: {now_text}
Datum do mapa: WGS84 (EPSG:4326)
Limite GO (origem): {original_boundary_crs}""",transform=ax.transAxes,fontsize=9,verticalalignment="bottom",bbox=dict(facecolor="white",alpha=0.8,edgecolor="gray"))
    fig.tight_layout(); out_hotspots=tempfile.NamedTemporaryFile(delete=False,suffix=".png"); fig.savefig(out_hotspots.name,dpi=220); plt.close(fig)
    return out_risco.name,out_hotspots.name

with st.expander("📂 Formato esperado do arquivo CSV", expanded=True):
    st.markdown("""Para que a ferramenta funcione corretamente, o arquivo CSV deve conter **no mínimo**:

### Campos obrigatórios
- **UTM_E_m** → Coordenada UTM Leste
- **UTM_N_m** → Coordenada UTM Norte
- **ZONA** → Zona UTM (exemplo: 22K)

### Campos recomendados
- **IDADE**
- **QUANTIDADE DE TANQUES**
- **MÉDIA DE IDADE DO TANQUE**
- **QUANTIDADE DE BOMBAS**
- **NÚMERO DE SONDAGENS**
- **ÁGUA SUBTERRÂNEA**
- **NIVEL_AGUA_m**

### Campos hidrogeológicos (opcional)
- **dist_poco_mais_prox_m**
- **pocos_500m**
- **pocos_1km**
- **pocos_5km**

Se você **não tiver um modelo** para enviar, o sistema pode usar uma coluna de risco já existente no CSV:
- **RISK_PROBA_HIDRO**
- **RISK_PROBA**""")
    template = pd.DataFrame({"NÚMERO DO POSTO":[1,2],"UTM_E_m":[686304.36,687577.99],"UTM_N_m":[8216649.19,8158767.99],"ZONA":["22K","22K"],"CIDADE":["GOIANIA","GOIANIA"],"IDADE":[20,15],"QUANTIDADE DE TANQUES":[4,3],"QUANTIDADE DE BOMBAS":[8,6],"NIVEL_AGUA_m":[10,12]})
    st.download_button("⬇️ Baixar modelo de CSV", data=template.to_csv(index=False).encode("utf-8"), file_name="template_postos.csv", mime="text/csv")

with st.expander("🧠 Como o sistema calcula o risco?", expanded=False):
    st.markdown("""O sistema utiliza um modelo de **Machine Learning** treinado com dados de postos de combustíveis no estado de Goiás.

Ele considera informações como:
- idade do posto e dos tanques
- quantidade de tanques e bombas
- presença de água subterrânea
- histórico de contaminação
- proximidade de poços de água
- concentração espacial de poços em diferentes distâncias

### Como interpretar o resultado
- valores próximos de **0** → risco mais baixo
- valores próximos de **1** → risco mais alto""")

st.sidebar.header("⚙️ Configurações")
uploaded_csv = st.sidebar.file_uploader("1) Arraste e solte o arquivo CSV dos postos aqui", type=["csv"])
uploaded_model = st.sidebar.file_uploader("2) (Opcional) Arraste aqui o modelo de previsão de risco", type=["joblib"])
st.sidebar.caption("O modelo de previsão de risco é um arquivo treinado com inteligência artificial. Ele calcula a probabilidade de contaminação ambiental com base nos dados do posto.")
hex_size = st.sidebar.slider("3) Tamanho do hexágono (metros)", 2000, 10000, 5000, 500)
municipio_filter = st.sidebar.text_input("4) Filtrar por município (opcional)")
risk_min = st.sidebar.slider("5) Risco mínimo para exibir", 0.0, 1.0, 0.0, 0.01)
run_button = st.sidebar.button("▶️ Gerar análise", type="primary")

if uploaded_csv is None:
    st.info("Envie um CSV para começar."); st.stop()
try:
    df = safe_read_csv(uploaded_csv); st.success("CSV carregado com sucesso.")
except Exception as e:
    st.error(f"Erro ao ler o CSV: {e}"); st.stop()

st.subheader("📋 Pré-visualização da base")
st.dataframe(df.head(10), use_container_width=True)
missing_basic=[c for c in ["UTM_E_m","UTM_N_m","ZONA"] if c not in df.columns]
if missing_basic:
    st.error(f"O CSV precisa conter: ['UTM_E_m', 'UTM_N_m', 'ZONA']. Faltando: {missing_basic}"); st.stop()
try:
    df_geo = convert_utm_to_wgs84(df)
    goias_wgs, boundary_original_crs = load_goias_boundary()
except Exception as e:
    st.error(str(e)); st.stop()

if run_button:
    model=None
    try:
        if uploaded_model is not None:
            model=joblib.load(uploaded_model); st.success("Modelo de previsão enviado carregado com sucesso.")
        elif DEFAULT_MODEL_PATH.exists():
            model=joblib.load(DEFAULT_MODEL_PATH); st.success(f"Modelo local carregado: {DEFAULT_MODEL_PATH}")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}"); st.stop()
    if model is not None:
        try:
            df_scored,risk_col=score_with_model(df_geo,model); st.success("Risco calculado pelo modelo com sucesso.")
        except Exception as e:
            st.error(f"Erro ao aplicar o modelo: {e}"); st.stop()
    else:
        detected=detect_risk_column(df_geo)
        if detected is None:
            st.error("Nenhum modelo foi carregado e o CSV não possui coluna de risco reconhecida."); st.stop()
        df_scored=df_geo.copy(); risk_col=detected; st.warning(f"Sem modelo. Usando a coluna de risco existente: {risk_col}")
    if municipio_filter and "CIDADE" in df_scored.columns:
        df_scored=df_scored[df_scored["CIDADE"].astype(str).str.contains(municipio_filter, case=False, na=False)].copy()
    df_scored=df_scored[pd.to_numeric(df_scored[risk_col],errors="coerce")>=risk_min].copy()
    gdf_points_wgs=gpd.GeoDataFrame(df_scored.copy(),geometry=gpd.points_from_xy(df_scored["lon"],df_scored["lat"]),crs="EPSG:4326")
    try:
        hex_grid_wgs=build_hex_grid(gdf_points_wgs=gdf_points_wgs,boundary_wgs=goias_wgs,risk_col=risk_col,hex_size=float(hex_size))
        fmap=build_interactive_map(df_points=df_scored,boundary_wgs=goias_wgs,hex_grid_wgs=hex_grid_wgs,risk_col=risk_col,original_boundary_crs=boundary_original_crs)
        fmap_html=fmap.get_root().render()
        png_risco,png_hotspots=make_static_maps(gdf_points_wgs,goias_wgs,hex_grid_wgs,boundary_original_crs)
    except Exception as e:
        st.error(f"Erro ao gerar saídas espaciais: {e}"); st.stop()
    st.session_state.analysis_done=True; st.session_state.df_scored=df_scored; st.session_state.risk_col=risk_col
    st.session_state.hex_grid_wgs=hex_grid_wgs; st.session_state.fmap_html=fmap_html
    st.session_state.png_risco=png_risco; st.session_state.png_hotspots=png_hotspots
    st.session_state.stats={"postos_validos":len(df_scored),"hexagonos_com_postos":int(len(hex_grid_wgs)),"hotspots":int(hex_grid_wgs["hotspot"].sum()) if "hotspot" in hex_grid_wgs.columns else 0,"risco_medio_geral":float(df_scored[risk_col].mean()) if len(df_scored) else 0.0}
    st.success("Análise concluída. Role a página para visualizar os mapas e downloads.")

if st.session_state.analysis_done:
    df_scored=st.session_state.df_scored; risk_col=st.session_state.risk_col; hex_grid_wgs=st.session_state.hex_grid_wgs
    fmap_html=st.session_state.fmap_html; png_risco=st.session_state.png_risco; png_hotspots=st.session_state.png_hotspots; stats=st.session_state.stats
    st.subheader("📈 Estatísticas principais")
    c1,c2,c3,c4=st.columns(4); c1.metric("Postos válidos",stats["postos_validos"]); c2.metric("Hexágonos com postos",stats["hexagonos_com_postos"]); c3.metric("Hotspots",stats["hotspots"]); c4.metric("Risco médio geral",f'{stats["risco_medio_geral"]:.3f}')
    st.subheader("🗺️ Mapa interativo de Goiás")
    st.components.v1.html(fmap_html,height=700,scrolling=True)
    st.subheader("🖼️ Mapas estáticos com estrutura cartográfica")
    col1,col2=st.columns(2)
    with col1: st.image(png_risco, caption="Mapa hexagonal de risco médio — Goiás", use_container_width=True)
    with col2: st.image(png_hotspots, caption="Hotspots de risco ambiental — Goiás", use_container_width=True)
    st.subheader("📦 Downloads")
    st.download_button("⬇️ Baixar CSV com risco", data=df_scored.to_csv(index=False).encode("utf-8"), file_name="postos_scored_risk_app.csv", mime="text/csv")
    st.download_button("⬇️ Baixar grade hexagonal (GeoJSON)", data=hex_grid_wgs.to_json().encode("utf-8"), file_name="hex_grid_risco_goias_app.geojson", mime="application/geo+json")
    st.download_button("⬇️ Baixar mapa interativo (HTML)", data=fmap_html.encode("utf-8"), file_name="mapa_interativo_goias_app.html", mime="text/html")
    with open(png_risco, "rb") as f: st.download_button("⬇️ Baixar PNG (risco médio)", data=f.read(), file_name="mapa_hex_risco_goias.png", mime="image/png")
    with open(png_hotspots, "rb") as f: st.download_button("⬇️ Baixar PNG (hotspots)", data=f.read(), file_name="mapa_hex_hotspots_goias.png", mime="image/png")
    st.subheader("📋 Pré-visualização dos resultados"); st.dataframe(df_scored.head(20), use_container_width=True)
else:
    st.info("Clique em **Gerar análise** na barra lateral para rodar o processo.")

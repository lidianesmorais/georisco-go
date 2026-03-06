
# ============================================================
# GeoRisco GO — Versão 5.0
# Suporte a CSV, Excel e GeoJSON com validação automática
# ============================================================

import io
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GeoRisco – Goiás", page_icon="🌎", layout="wide")

st.title("🌎 GeoRisco – Goiás")
st.caption("Versão beta com suporte a CSV, Excel e GeoJSON, validação automática de campos e diagnóstico da planilha.")

# ------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------
def ler_arquivo(arquivo):
    """
    Lê automaticamente:
    - CSV brasileiro
    - CSV internacional
    - Excel (.xlsx)
    - GeoJSON
    """
    nome = arquivo.name.lower()

    if nome.endswith(".xlsx"):
        return pd.read_excel(arquivo), "xlsx"

    if nome.endswith(".geojson") or nome.endswith(".json"):
        gdf = gpd.read_file(arquivo)
        return pd.DataFrame(gdf.drop(columns="geometry", errors="ignore")), "geojson"

    bruto = arquivo.getvalue()

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
                io.BytesIO(bruto),
                encoding=tentativa["encoding"],
                sep=tentativa["sep"],
                decimal=tentativa["decimal"]
            )
            if df.shape[1] > 1:
                df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
                return df, "csv"
        except Exception as e:
            ultimo_erro = e

    raise ValueError(f"Não foi possível ler o arquivo. Detalhe técnico: {ultimo_erro}")


def normalizar_colunas(df):
    """
    Mantém os nomes originais e cria uma versão normalizada para comparação.
    """
    mapa = {}
    for c in df.columns:
        chave = (
            str(c)
            .strip()
            .upper()
            .replace("Á", "A")
            .replace("À", "A")
            .replace("Ã", "A")
            .replace("Â", "A")
            .replace("É", "E")
            .replace("Ê", "E")
            .replace("Í", "I")
            .replace("Ó", "O")
            .replace("Ô", "O")
            .replace("Õ", "O")
            .replace("Ú", "U")
            .replace("Ç", "C")
        )
        mapa[chave] = c
    return mapa


def diagnosticar_colunas(df):
    """
    Faz o diagnóstico de campos:
    - obrigatórios
    - recomendados
    - hidrogeológicos
    - auxiliares
    """
    mapa = normalizar_colunas(df)

    grupos = {
        "obrigatorios": ["UTM_E_M", "UTM_N_M", "ZONA"],
        "recomendados": [
            "IDADE",
            "QUANTIDADE DE TANQUES",
            "QUANTIDADE DE BOMBAS",
            "JAQUETADO",
            "MEDIA DE IDADE DO TANQUE",
            "NUMERO DE SONDAGENS",
            "AGUA SUBTERRANEA",
            "NIVEL_AGUA_M"
        ],
        "hidrogeologicos": [
            "DIST_POCO_MAIS_PROX_M",
            "POCOS_500M",
            "POCOS_1KM",
            "POCOS_5KM"
        ],
        "auxiliares": [
            "JA APRESENTOU CONTAMINACAO ANTES",
            "CONC. BTEX",
            "CONC. PAH"
        ]
    }

    resultado = {}
    for grupo, campos in grupos.items():
        encontrados = []
        faltando = []

        for campo in campos:
            if campo in mapa:
                encontrados.append(mapa[campo])
            else:
                faltando.append(campo)

        resultado[grupo] = {
            "encontrados": encontrados,
            "faltando": faltando
        }

    return resultado


def classificar_prontidao(diag):
    obrig_ok = len(diag["obrigatorios"]["faltando"]) == 0
    qtd_recomendados = len(diag["recomendados"]["encontrados"])
    qtd_hidro = len(diag["hidrogeologicos"]["encontrados"])

    if not obrig_ok:
        return "insuficiente", "A base não tem os campos mínimos para execução."
    if qtd_recomendados >= 5 and qtd_hidro >= 2:
        return "alta", "A base está muito bem preparada para análise."
    if qtd_recomendados >= 3:
        return "media", "A base pode ser usada, mas alguns campos importantes estão ausentes."
    return "basica", "A base atende ao mínimo, porém a qualidade da previsão pode ser reduzida."


def gerar_template_robusto():
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
        "NÚMERO DE SONDAGENS": [4, 2],
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


def texto_resumo_diagnostico(diag):
    return dedent(f"""
    **Obrigatórios encontrados:** {len(diag["obrigatorios"]["encontrados"])} de 3  
    **Recomendados encontrados:** {len(diag["recomendados"]["encontrados"])} de 8  
    **Hidrogeológicos encontrados:** {len(diag["hidrogeologicos"]["encontrados"])} de 4  
    **Auxiliares encontrados:** {len(diag["auxiliares"]["encontrados"])} de 3
    """).strip()


# ------------------------------------------------------------
# Painéis explicativos
# ------------------------------------------------------------
with st.expander("📄 Requisitos dos dados de entrada", expanded=True):
    st.markdown("""
### Campos obrigatórios
Esses campos são indispensáveis para execução do sistema e posicionamento espacial:

- **UTM_E_m**
- **UTM_N_m**
- **ZONA**

### Campos recomendados
Melhoram a consistência da previsão:

- **IDADE**
- **QUANTIDADE DE TANQUES**
- **QUANTIDADE DE BOMBAS**
- **JAQUETADO**
- **MÉDIA DE IDADE DO TANQUE**
- **NÚMERO DE SONDAGENS**
- **ÁGUA SUBTERRÂNEA**
- **NIVEL_AGUA_m**

### Campos hidrogeológicos
São as variáveis espaciais derivadas do estudo com base hidrogeológica:

- **dist_poco_mais_prox_m**
- **pocos_500m**
- **pocos_1km**
- **pocos_5km**

### Histórico e indicadores auxiliares
São complementares e ajudam a enriquecer a análise:

- **JÁ APRESENTOU CONTAMINAÇÃO ANTES**
- **CONC. BTEX**
- **CONC. PAH**

### Regras de funcionamento da versão beta
- Se a base tiver os **campos obrigatórios**, o sistema pode continuar.
- Se tiver os **campos recomendados**, a previsão tende a ser melhor.
- Se tiver os **campos hidrogeológicos**, a análise fica mais alinhada ao estudo original.
- Se houver **colunas extras**, elas são aceitas e ignoradas quando não forem necessárias.
""")

    st.download_button(
        "⬇️ Baixar modelo robusto de planilha",
        data=gerar_template_robusto().to_csv(index=False).encode("utf-8"),
        file_name="template_georisco_goias_v5.csv",
        mime="text/csv"
    )

st.warning("Para planilhas exportadas do Excel brasileiro, o formato mais compatível é CSV com separador ';' ou arquivo Excel .xlsx.")

# ------------------------------------------------------------
# Upload
# ------------------------------------------------------------
arquivo = st.file_uploader(
    "Envie a planilha de dados dos postos",
    type=["csv", "xlsx", "geojson", "json"]
)

if arquivo is not None:
    try:
        df, tipo = ler_arquivo(arquivo)

        st.success(f"Arquivo carregado com sucesso. Tipo identificado: {tipo.upper()}")

        st.subheader("📋 Pré-visualização da base")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("🔎 Diagnóstico automático da planilha")
        diag = diagnosticar_colunas(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Obrigatórios encontrados", f'{len(diag["obrigatorios"]["encontrados"])}/3')
        c2.metric("Recomendados encontrados", f'{len(diag["recomendados"]["encontrados"])}/8')
        c3.metric("Hidrogeológicos encontrados", f'{len(diag["hidrogeologicos"]["encontrados"])}/4')
        c4.metric("Auxiliares encontrados", f'{len(diag["auxiliares"]["encontrados"])}/3')

        nivel, mensagem = classificar_prontidao(diag)

        if nivel == "alta":
            st.success("Prontidão alta: " + mensagem)
        elif nivel == "media":
            st.info("Prontidão média: " + mensagem)
        elif nivel == "basica":
            st.warning("Prontidão básica: " + mensagem)
        else:
            st.error("Prontidão insuficiente: " + mensagem)

        st.markdown(texto_resumo_diagnostico(diag))

        with st.expander("📌 Detalhamento dos campos encontrados e ausentes", expanded=False):
            for grupo, titulo in [
                ("obrigatorios", "Campos obrigatórios"),
                ("recomendados", "Campos recomendados"),
                ("hidrogeologicos", "Campos hidrogeológicos"),
                ("auxiliares", "Histórico e indicadores auxiliares"),
            ]:
                st.markdown(f"### {titulo}")
                st.write("**Encontrados:**", diag[grupo]["encontrados"] if diag[grupo]["encontrados"] else "Nenhum")
                st.write("**Ausentes:**", diag[grupo]["faltando"] if diag[grupo]["faltando"] else "Nenhum")

        st.subheader("🧭 Interpretação operacional")
        st.markdown("""
- **Obrigatórios ausentes** → o sistema não deve prosseguir para o mapa.
- **Recomendados ausentes** → o sistema pode rodar, mas com menos qualidade analítica.
- **Hidrogeológicos ausentes** → o sistema continua, porém fica menos aderente ao estudo original.
- **Auxiliares ausentes** → não impedem a execução.
""")

        if len(diag["obrigatorios"]["faltando"]) == 0:
            st.success("A base possui os campos mínimos necessários para execução.")
        else:
            st.error("A base ainda não possui os campos mínimos necessários para execução.")

        st.subheader("📦 Exportação do diagnóstico")
        resumo = {
            "tipo_arquivo": [tipo],
            "total_registros": [len(df)],
            "obrigatorios_encontrados": [len(diag["obrigatorios"]["encontrados"])],
            "recomendados_encontrados": [len(diag["recomendados"]["encontrados"])],
            "hidrogeologicos_encontrados": [len(diag["hidrogeologicos"]["encontrados"])],
            "auxiliares_encontrados": [len(diag["auxiliares"]["encontrados"])],
            "classificacao": [nivel],
            "mensagem": [mensagem],
        }
        df_resumo = pd.DataFrame(resumo)

        st.download_button(
            "⬇️ Baixar resumo do diagnóstico",
            data=df_resumo.to_csv(index=False).encode("utf-8"),
            file_name="diagnostico_georisco_v5.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Data Healthcheck Pro", layout="wide")

def get_healthcheck_stats(df):
    stats = pd.DataFrame({
        'Nulidad (%)': (df.isnull().mean() * 100).round(2),
        'Tipo de Dato': df.dtypes.astype(str)
    })
    outlier_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_counts[col] = count
    stats['Outliers Detectados'] = pd.Series(outlier_counts)
    return stats, df.duplicated().sum()

def justify_imputation(col_data):
    if pd.api.types.is_numeric_dtype(col_data):
        skewness = col_data.skew()
        if abs(skewness) < 0.5:
            return "Media", f"Simétrica (skew: {skewness:.2f})"
        else:
            return "Mediana", f"Sesgada (skew: {skewness:.2f})"
    return "Moda", "Variable categórica"


def detect_date_candidates(df):
    keywords = ['fecha', 'date', 'dia', 'day', 'timestamp', 'time']
    candidates = []
    for c in df.columns:
        name = c.lower()
        if any(k in name for k in keywords):
            candidates.append(c)
            continue
        sample = df[c].dropna().head(20)
        if len(sample) > 0:
            parsed = pd.to_datetime(sample, errors='coerce')
            if parsed.notna().sum() / len(sample) > 0.6:
                candidates.append(c)
    return candidates


def detect_category_candidates(df):
    keywords = ['categor', 'cat', 'tipo', 'segmento', 'category']
    candidates = []
    for c in df.columns:
        name = c.lower()
        dtype = str(df[c].dtype)
        nunique = df[c].nunique(dropna=True)
        if any(k in name for k in keywords):
            candidates.append(c)
            continue
        if dtype.startswith('object') or dtype.startswith('category'):
            # prefer columns with not-too-many distinct values
            if nunique > 0 and nunique < max(100, len(df) * 0.5):
                candidates.append(c)
    return candidates


def detect_bodega_candidates(df):
    keywords = ['bodega', 'tienda', 'store', 'warehouse', 'sucursal', 'branch']
    candidates = []
    for c in df.columns:
        name = c.lower()
        nunique = df[c].nunique(dropna=True)
        if any(k in name for k in keywords):
            candidates.append(c)
            continue
        # heuristic: small-cardinality location/store-like columns
        if (str(df[c].dtype).startswith('object') or str(df[c].dtype).startswith('category')) and nunique > 0 and nunique < min(500, max(10, len(df) * 0.2)):
            candidates.append(c)
    return candidates

# --- SIDEBAR ESTRUCTURADA ---
with st.sidebar:
    st.title("📂 Panel de Control")
    uploaded_files = st.file_uploader("Sube tus datasets", type=['csv', 'xlsx'], accept_multiple_files=True)
    
    filtros_activos = {} # Diccionario para guardar filtros por archivo

    if uploaded_files:
        st.divider()
        for i, file in enumerate(uploaded_files[:3]):
            st.subheader(f"🛠️ Filtros: {file.name}")
            df_temp = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)

            # Detectar candidatos para cada tipo de filtro
            date_candidates = detect_date_candidates(df_temp)
            cat_candidates = detect_category_candidates(df_temp)
            bod_candidates = detect_bodega_candidates(df_temp)

            # Selección por parte del usuario (puede elegir 'ninguna')
            sel_date = st.selectbox(f"Columna de fecha para {file.name}", ['(ninguna)'] + date_candidates, key=f"sel_date_{i}")
            dates = (None, None)
            if sel_date and sel_date != '(ninguna)':
                try:
                    df_temp[sel_date] = pd.to_datetime(df_temp[sel_date], errors='coerce')
                    min_d, max_d = df_temp[sel_date].min(), df_temp[sel_date].max()
                    # si no hay fechas válidas, evitamos el date_input
                    if pd.isna(min_d) or pd.isna(max_d):
                        st.warning(f"La columna {sel_date} no contiene fechas reconocibles.")
                    else:
                        dates = st.date_input(f"Rango de {sel_date}", [min_d, max_d], key=f"date_{i}")
                except Exception:
                    st.warning(f"No se pudo convertir {sel_date} a datetime.")

            sel_cat = st.selectbox(f"Columna de categoría para {file.name}", ['(ninguna)'] + cat_candidates, key=f"sel_cat_{i}")
            cat_sel = []
            if sel_cat and sel_cat != '(ninguna)':
                values = df_temp[sel_cat].dropna().unique().tolist()
                cat_sel = st.multiselect(f"Valores en {sel_cat}", values, key=f"cat_{i}")

            sel_bod = st.selectbox(f"Columna de bodega/tienda para {file.name}", ['(ninguna)'] + bod_candidates, key=f"sel_bod_{i}")
            bod_sel = []
            if sel_bod and sel_bod != '(ninguna)':
                values = df_temp[sel_bod].dropna().unique().tolist()
                bod_sel = st.multiselect(f"Valores en {sel_bod}", values, key=f"bod_{i}")
            
            filtros_activos[file.name] = {
                'date_col': None if sel_date == '(ninguna)' else sel_date,
                'dates': dates,
                'cat_col': None if sel_cat == '(ninguna)' else sel_cat,
                'cat_sel': cat_sel,
                'bod_col': None if sel_bod == '(ninguna)' else sel_bod,
                'bod_sel': bod_sel
            }
            st.divider()

        if st.button("🔄 Refrescar Análisis", use_container_width=True):
            st.rerun()

# --- CUERPO PRINCIPAL ---
st.title("🛡️ Data Cleaning Healthcheck")

if not uploaded_files:
    st.info("👈 Sube los archivos para aplicar los filtros de Categoría, Bodega y Fecha.")
else:
    for file in uploaded_files[:3]:
        st.header(f"Dataset: {file.name}")
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        
        # --- APLICACIÓN DE FILTROS ---
        f = filtros_activos[file.name]
        if f['date_col'] and len(f['dates']) == 2:
            df[f['date_col']] = pd.to_datetime(df[f['date_col']], errors='coerce')
            df = df[(df[f['date_col']].dt.date >= f['dates'][0]) & (df[f['date_col']].dt.date <= f['dates'][1])]
        if f['cat_sel']:
            df = df[df[f['cat_col']].isin(f['cat_sel'])]
        if f['bod_sel']:
            df = df[df[f['bod_col']].isin(f['bod_sel'])]

        # --- HEALTHCHECK ---
        col1, col2 = st.columns(2)
        df_orig = df.copy() # El "Antes" ya viene filtrado por el usuario
        
        with col1:
            st.subheader("🔍 Healthcheck: Antes")
            stats_before, dups_before = get_healthcheck_stats(df_orig)
            st.write(f"**Registros filtrados:** {len(df_orig)} | **Duplicados:** {dups_before}")
            st.dataframe(stats_before)

        # --- LIMPIEZA ---
        df_clean = df_orig.drop_duplicates()
        decisiones = []
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                metodo, razon = justify_imputation(df_clean[col])
                decisiones.append({"Columna": col, "Decisión": f"Imputar con {metodo}", "Justificación": razon})
                fill_val = df_clean[col].mean() if metodo == "Media" else (df_clean[col].median() if metodo == "Mediana" else df_clean[col].mode()[0])
                df_clean[col] = df_clean[col].fillna(fill_val)

        with col2:
            st.subheader("✅ Healthcheck: Después")
            stats_after, _ = get_healthcheck_stats(df_clean)
            st.write(f"**Registros finales:** {len(df_clean)} | **Limpieza completa**")
            st.dataframe(stats_after)

        st.info("### ⚖️ Decisión Ética")
        if decisiones: st.table(pd.DataFrame(decisiones))
        else: st.write("Sin nulos detectados en la selección.")
        st.divider()
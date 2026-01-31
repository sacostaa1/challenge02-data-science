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
            
            # 1. Filtro de Fecha (Detección automática)
            date_col = next((c for c in df_temp.columns if 'fecha' in c.lower()), None)
            dates = (None, None)
            if date_col:
                df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                min_d, max_d = df_temp[date_col].min(), df_temp[date_col].max()
                dates = st.date_input(f"Rango de {date_col}", [min_d, max_d], key=f"date_{i}")
            
            # 2. Filtro de Categoría
            cat_col = next((c for c in df_temp.columns if 'categor' in c.lower()), None)
            cat_sel = st.multiselect(f"Categorías en {file.name}", df_temp[cat_col].unique(), key=f"cat_{i}") if cat_col else []
            
            # 3. Filtro de Bodega / Tienda
            bod_col = next((c for c in df_temp.columns if 'bodega' in c.lower() or 'tienda' in c.lower()), None)
            bod_sel = st.multiselect(f"Bodega/Tienda en {file.name}", df_temp[bod_col].unique(), key=f"bod_{i}") if bod_col else []
            
            filtros_activos[file.name] = {
                'date_col': date_col, 'dates': dates,
                'cat_col': cat_col, 'cat_sel': cat_sel,
                'bod_col': bod_col, 'bod_sel': bod_sel
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
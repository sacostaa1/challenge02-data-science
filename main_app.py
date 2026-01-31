import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_healthcheck_stats(df):
    """Calcula las métricas de calidad solicitadas."""
    stats = pd.DataFrame({
        'Nulidad (%)': (df.isnull().mean() * 100).round(2),
        'Tipo de Dato': df.dtypes.astype(str)
    })
    
    # Detección de outliers (IQR)
    outlier_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_counts[col] = count
    
    stats['Outliers Detectados'] = pd.Series(outlier_counts)
    return stats, df.duplicated().sum()

def justify_imputation(col_data):
    """Lógica para la decisión ética basada en distribución."""
    if pd.api.types.is_numeric_dtype(col_data):
        skewness = col_data.skew()
        if abs(skewness) < 0.5:
            return "Media", f"Distribución simétrica (skew: {skewness:.2f})"
        else:
            return "Mediana", f"Distribución sesgada (skew: {skewness:.2f})"
    else:
        return "Moda", "Variable categórica"

# --- INTERFAZ STREAMLIT ---
st.set_page_config(page_title="Data Healthcheck", layout="wide")
st.title("🛡️ Data Cleaning Healthcheck")

uploaded_files = st.file_uploader("Sube tus datasets (Máximo 3)", type=['csv', 'xlsx'], accept_multiple_files=True)

if uploaded_files:
    for i, file in enumerate(uploaded_files[:3]):
        st.header(f"Dataset {i+1}: {file.name}")
        df_orig = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        
        col1, col2 = st.columns(2)
        
        # --- ESTADO INICIAL ---
        with col1:
            st.subheader("🔍 Healthcheck: Antes")
            stats_before, dups_before = get_healthcheck_stats(df_orig)
            st.write(f"**Registros duplicados:** {dups_before}")
            st.dataframe(stats_before)
            
        # --- PROCESO DE LIMPIEZA (SIMULADO / AUTOMATIZADO) ---
        df_clean = df_orig.copy()
        df_clean = df_clean.drop_duplicates()
        
        decisiones = []
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                metodo, razon = justify_imputation(df_clean[col])
                decisiones.append({"Columna": col, "Decisión": f"Imputar con {metodo}", "Justificación": razon})
                # Aplicar imputación
                if metodo == "Media": df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif metodo == "Mediana": df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else: df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

        # --- ESTADO FINAL ---
        with col2:
            st.subheader("✅ Healthcheck: Después")
            stats_after, dups_after = get_healthcheck_stats(df_clean)
            st.write(f"**Duplicados eliminados:** {dups_before - dups_after}")
            st.dataframe(stats_after)

        # --- SECCIÓN ÉTICA ---
        st.info("### ⚖️ Decisión Ética y Justificación")
        if decisiones:
            st.table(pd.DataFrame(decisiones))
        else:
            st.write("No se requirió imputación de datos.")

        # --- VISUALIZACIÓN DE MAGNITUD ---
        if not df_orig.select_dtypes(include=[np.number]).empty:
            st.subheader("📊 Magnitud de Outliers (Boxplot)")
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            num_col = df_orig.select_dtypes(include=[np.number]).columns[0] # Tomamos la primera numérica
            sns.boxplot(y=df_orig[num_col], ax=ax[0], color="salmon").set_title(f"Antes: {num_col}")
            sns.boxplot(y=df_clean[num_col], ax=ax[1], color="lightblue").set_title(f"Después: {num_col}")
            st.pyplot(fig)
        
        st.divider()

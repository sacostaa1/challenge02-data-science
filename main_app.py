import streamlit as st
import pandas as pd
import numpy as np
from clean_inventario import clean_inventario_central
from filters import apply_filters_ui

st.set_page_config(page_title="Data Healthcheck Pro", layout="wide")


# =========================
#      HEALTHCHECK CORE
# =========================
def outlier_iqr_stats(series: pd.Series):
    """Devuelve conteo y % de outliers por regla IQR (1.5*IQR)."""
    s = series.dropna()
    if s.empty:
        return 0, 0.0, np.nan, np.nan

    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1

    if IQR == 0 or pd.isna(IQR):
        return 0, 0.0, Q1, Q3

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((s < lower) | (s > upper)).sum()
    pct = (outliers / len(s)) * 100
    return int(outliers), round(pct, 2), lower, upper


def get_healthcheck_report(df: pd.DataFrame):
    duplicated_rows = int(df.duplicated().sum())

    report = pd.DataFrame(index=df.columns)
    report["Tipo de Dato"] = df.dtypes.astype(str)
    report["Nulos (#)"] = df.isnull().sum()
    report["Nulidad (%)"] = (df.isnull().mean() * 100).round(2)
    report["Únicos (#)"] = df.nunique(dropna=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report["Mín"] = np.nan
    report["Máx"] = np.nan
    if num_cols:
        report.loc[num_cols, "Mín"] = df[num_cols].min(numeric_only=True)
        report.loc[num_cols, "Máx"] = df[num_cols].max(numeric_only=True)

    report["Outliers (#)"] = np.nan
    report["Outliers (%)"] = np.nan
    report["IQR Lower"] = np.nan
    report["IQR Upper"] = np.nan

    for col in num_cols:
        out_n, out_pct, lower, upper = outlier_iqr_stats(df[col])
        report.loc[col, "Outliers (#)"] = out_n
        report.loc[col, "Outliers (%)"] = out_pct
        report.loc[col, "IQR Lower"] = lower
        report.loc[col, "IQR Upper"] = upper

    report = report.sort_values(
        by=["Nulidad (%)", "Outliers (%)", "Únicos (#)"],
        ascending=[False, False, True]
    )

    resumen = {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "duplicados": duplicated_rows,
        "pct_duplicados": round((duplicated_rows / len(df)) * 100, 2) if len(df) else 0.0,
        "total_nulos": int(df.isnull().sum().sum()),
        "pct_nulos_total": round((df.isnull().sum().sum() / (df.size)) * 100, 2) if df.size else 0.0,
        "cols_con_nulos": int((df.isnull().sum() > 0).sum()),
        "cols_con_outliers": int(((report["Outliers (#)"].fillna(0)) > 0).sum())
    }

    return report, resumen


# =========================
#   BUSINESS FINDINGS SAFE
# =========================
def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def dataset_business_checks(name: str, df: pd.DataFrame, inventory_df=None):
    findings = []

    def add(issue, metric, detail=""):
        findings.append({"Hallazgo": issue, "Métrica": int(metric), "Detalle": detail})

    lname = name.lower()

    if "inventario" in lname:
        if "Stock_Actual" in df.columns:
            stock = safe_numeric(df["Stock_Actual"])
            neg = int((stock < 0).sum())
            add("Stock negativo", neg, "Existencias < 0 (interpretación: lista de espera)")

        if "Costo_Unitario_USD" in df.columns:
            costo = safe_numeric(df["Costo_Unitario_USD"])
            zero_or_neg = int((costo <= 0).sum())
            add("Costo <= 0", zero_or_neg, "Costos en 0 o negativos")

        if "Ultima_Revision" in df.columns:
            parsed = pd.to_datetime(df["Ultima_Revision"], errors="coerce")
            invalid = int(parsed.isna().sum())
            add("Fechas inválidas (Ultima_Revision)", invalid, "No parseables a datetime")

        # inventory cleaning rules moved to clean_inventario.py (imported above)

    return pd.DataFrame(findings)


# =========================
#   GENERIC CLEAN (OTHERS)
# =========================
def justify_imputation(col_data: pd.Series):
    if pd.api.types.is_numeric_dtype(col_data):
        skewness = col_data.dropna().skew()
        if pd.isna(skewness):
            return "Mediana", "No se pudo calcular skew (pocos datos). Se usa mediana por robustez."
        if abs(skewness) < 0.5:
            return "Media", f"Distribución aproximadamente simétrica (skew: {skewness:.2f})"
        else:
            return "Mediana", f"Distribución sesgada (skew: {skewness:.2f}), mediana es más robusta"
    return "Moda", "Variable categórica: se usa el valor más frecuente (moda)"


def clean_dataset_generic(df: pd.DataFrame):
    decisiones = []

    dup_count = int(df.duplicated().sum())
    df_clean = df.drop_duplicates().copy()

    decisiones.append({
        "Acción": "Eliminar duplicados exactos",
        "Columna": "(todas)",
        "Registros afectados": dup_count,
        "Justificación": "Se eliminaron filas duplicadas completas para evitar doble conteo."
    })

    for col in df_clean.columns:
        nulls = int(df_clean[col].isnull().sum())
        if nulls == 0:
            continue

        metodo, razon = justify_imputation(df_clean[col])

        if metodo == "Media":
            fill_val = pd.to_numeric(df_clean[col], errors="coerce").mean()
        elif metodo == "Mediana":
            fill_val = pd.to_numeric(df_clean[col], errors="coerce").median()
        else:
            mode_series = df_clean[col].mode(dropna=True)
            fill_val = mode_series.iloc[0] if not mode_series.empty else None

        if fill_val is not None and not (isinstance(fill_val, float) and np.isnan(fill_val)):
            df_clean[col] = df_clean[col].fillna(fill_val)

        decisiones.append({
            "Acción": "Imputar nulos",
            "Columna": col,
            "Registros afectados": nulls,
            "Justificación": f"{metodo}. {razon}"
        })

    return df_clean, pd.DataFrame(decisiones)


# =========================
#       UI SIDEBAR
# =========================
st.title("🛡️ Data Healthcheck Pro (Diagnóstico + Limpieza Dirigida)")

with st.sidebar:
    st.title("📂 Panel de Control")
    uploaded_files = st.file_uploader(
        "Sube tus datasets",
        type=['csv', 'xlsx'],
        accept_multiple_files=True
    )

    if st.button("🔄 Refrescar App", use_container_width=True):
        st.rerun()


# =========================
#        MAIN BODY
# =========================
if not uploaded_files:
    st.info("👈 Sube los archivos para generar métricas, limpieza y descarga.")
    st.stop()

inventory_ref = None
for f in uploaded_files:
    if "inventario" in f.name.lower():
        try:
            f.seek(0)
            inventory_ref = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
        except Exception:
            inventory_ref = None
        break

st.subheader("🧾 Diagnóstico y Limpieza (por dataset)")

for file in uploaded_files[:3]:
    st.header(f"Dataset: {file.name}")

    try:
        file.seek(0)
    except Exception:
        pass

    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    except Exception as e:
        st.warning(f"Error leyendo {file.name}: {e}")
        continue

    # ==============
    # HEALTHCHECK BEFORE
    # ==============
    st.subheader("🔍 Healthcheck (Antes)")
    report_before, resumen_before = get_healthcheck_report(df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filas", resumen_before["filas"])
    m2.metric("Columnas", resumen_before["columnas"])
    m3.metric("Duplicados", resumen_before["duplicados"])
    m4.metric("% Nulos total", f'{resumen_before["pct_nulos_total"]}%')

    st.dataframe(report_before, use_container_width=True)

    # ==============
    # CLEANING (specific inventory rules)
    # ==============
    st.subheader("🧹 Limpieza aplicada")

    if "inventario_central" in file.name.lower() or "inventario" in file.name.lower():
        df_clean, decisiones_df = clean_inventario_central(df)
        st.success("Se aplicó limpieza específica para inventario_central.")
    else:
        df_clean, decisiones_df = clean_dataset_generic(df)
        st.info("Se aplicó limpieza genérica (duplicados + imputación).")

    # ==============
    # HEALTHCHECK AFTER
    # ==============
    st.subheader("✅ Healthcheck (Después)")
    report_after, resumen_after = get_healthcheck_report(df_clean)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filas finales", resumen_after["filas"])
    m2.metric("Cols con nulos", resumen_after["cols_con_nulos"])
    m3.metric("Duplicados (después)", resumen_after["duplicados"])
    m4.metric("Cols con outliers", resumen_after["cols_con_outliers"])

    st.dataframe(report_after, use_container_width=True)

        # ==============
        # FILTRADO SOBRE DATA LIMPIA
        # ==============
        with st.expander("🔎 Aplicar filtros al dataset limpio (fecha/categoría/bodega)", expanded=False):
            df_filtered = apply_filters_ui(df_clean, file.name, key_prefix=file.name)

            st.markdown("**Healthcheck tras filtros aplicados**")
            report_filt, resumen_filt = get_healthcheck_report(df_filtered)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Filas (filtradas)", resumen_filt["filas"])
            m2.metric("Columnas", resumen_filt["columnas"])
            m3.metric("Duplicados", resumen_filt["duplicados"])
            m4.metric("% Nulos total", f'{resumen_filt["pct_nulos_total"]}%')

            st.dataframe(report_filt, use_container_width=True)

            # descarga del subset filtrado
            sub_csv = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📥 Descargar subset filtrado: {file.name}",
                data=sub_csv,
                file_name=f"{file.name.replace('.csv','').replace('.xlsx','')}_filtered.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ==============
    # BUSINESS FINDINGS
    # ==============
    st.subheader("🧠 Hallazgos de negocio (Data Quality Rules)")
    findings = dataset_business_checks(file.name, df, inventory_df=inventory_ref)
    if findings.empty:
        st.success("No se detectaron hallazgos con las reglas actuales.")
    else:
        st.dataframe(findings, use_container_width=True)

    # ==============
    # ETHICAL DECISION MODULE
    # ==============
    st.subheader("⚖️ Decisión Ética (Qué eliminé y qué imputé)")
    st.caption("Incluye normalizaciones, imputaciones y eliminaciones de filas con justificación.")

    if decisiones_df.empty:
        st.info("No se registraron decisiones (dataset sin cambios).")
    else:
        st.dataframe(decisiones_df, use_container_width=True)

    # ==============
    # DOWNLOAD CLEAN CSV
    # ==============
    st.subheader("⬇️ Descargar dataset limpio")

    clean_csv = df_clean.to_csv(index=False).encode("utf-8")
    clean_name = file.name.replace(".csv", "").replace(".xlsx", "")
    st.download_button(
        label=f"📥 Descargar {clean_name}_clean.csv",
        data=clean_csv,
        file_name=f"{clean_name}_clean.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()


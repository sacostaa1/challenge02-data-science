import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    """
    Reporte completo:
    - nulidad
    - tipos
    - duplicados
    - outliers numéricos
    - min/max para inspección rápida
    """
    duplicated_rows = int(df.duplicated().sum())

    report = pd.DataFrame(index=df.columns)
    report["Tipo de Dato"] = df.dtypes.astype(str)
    report["Nulos (#)"] = df.isnull().sum()
    report["Nulidad (%)"] = (df.isnull().mean() * 100).round(2)
    report["Únicos (#)"] = df.nunique(dropna=True)

    # min/max numéricas
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report["Mín"] = np.nan
    report["Máx"] = np.nan
    if num_cols:
        report.loc[num_cols, "Mín"] = df[num_cols].min(numeric_only=True)
        report.loc[num_cols, "Máx"] = df[num_cols].max(numeric_only=True)

    # outliers
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


def justify_imputation(col_data: pd.Series):
    if pd.api.types.is_numeric_dtype(col_data):
        skewness = col_data.skew()
        if abs(skewness) < 0.5:
            return "Media", f"Distribución simétrica (skew: {skewness:.2f})"
        else:
            return "Mediana", f"Distribución sesgada (skew: {skewness:.2f})"
    return "Moda", "Variable categórica"


def dataset_business_checks(name: str, df: pd.DataFrame, inventory_df=None):
    """
    Retorna una tabla de hallazgos de negocio por dataset.
    inventory_df se usa para validar integridad referencial en transacciones.
    """
    findings = []

    def add(issue, metric, detail=""):
        findings.append({
            "Hallazgo": issue,
            "Métrica": metric,
            "Detalle": detail
        })

    lname = name.lower()

    # --- INVENTARIO ---
    if "inventario" in lname:
        if "Stock_Actual" in df.columns:
            neg = int((df["Stock_Actual"] < 0).sum())
            add("Stock negativo", neg, "Existencias < 0 (inconsistencia contable)")

        if "Costo_Unitario_USD" in df.columns:
            zero_or_neg = int((df["Costo_Unitario_USD"] <= 0).sum())
            add("Costo <= 0", zero_or_neg, "Costos en 0 o negativos")

        if "Ultima_Revision" in df.columns:
            parsed = pd.to_datetime(df["Ultima_Revision"], errors="coerce")
            invalid = int(parsed.isna().sum())
            add("Fechas inválidas (Ultima_Revision)", invalid, "No parseables a datetime")

        if "Lead_Time_Dias" in df.columns:
            extreme = int((df["Lead_Time_Dias"] > 365).sum())
            add("Lead Time extremo (>365 días)", extreme, "Sospechoso para logística")

    # --- TRANSACCIONES ---
    if "transacciones" in lname or "logistica" in lname:
        if "Fecha_Venta" in df.columns:
            parsed = pd.to_datetime(df["Fecha_Venta"], errors="coerce")
            invalid = int(parsed.isna().sum())
            add("Fechas inválidas (Fecha_Venta)", invalid, "Formato inconsistente")

        if "Tiempo_Entrega_Real" in df.columns:
            extreme = int((df["Tiempo_Entrega_Real"] > 365).sum())
            add("Tiempo entrega extremo (>365 días)", extreme, "Ej: 999 días")

        if inventory_df is not None and "SKU_ID" in df.columns and "SKU_ID" in inventory_df.columns:
            missing = df[~df["SKU_ID"].isin(inventory_df["SKU_ID"])]
            add("SKUs sin maestro (inventario)", int(len(missing)), "Ventas con SKU inexistente")

    # --- FEEDBACK ---
    if "feedback" in lname or "clientes" in lname:
        if "Edad_Cliente" in df.columns:
            impossible = int((df["Edad_Cliente"] > 110).sum())
            add("Edad imposible (>110)", impossible, "Ej: 195 años")

        if "Satisfaccion_NPS" in df.columns:
            below = int((df["Satisfaccion_NPS"] < -100).sum())
            above = int((df["Satisfaccion_NPS"] > 100).sum())
            add("NPS fuera de rango [-100,100]", below + above, "Requiere normalización")

        dup = int(df.duplicated().sum())
        add("Duplicados detectados (posibles intencionales)", dup, "Validar si deben eliminarse")

    return pd.DataFrame(findings)


# =========================
#       FILTROS SMART
# =========================
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
        if (str(df[c].dtype).startswith('object') or str(df[c].dtype).startswith('category')) and nunique > 0 and nunique < min(500, max(10, len(df) * 0.2)):
            candidates.append(c)
    return candidates


# =========================
#         UI SIDEBAR
# =========================
st.title("🛡️ Data Healthcheck Pro (Separado del EDA)")

with st.sidebar:
    st.title("📂 Panel de Control")
    uploaded_files = st.file_uploader(
        "Sube tus datasets",
        type=['csv', 'xlsx'],
        accept_multiple_files=True
    )

    filtros_activos = {}

    if uploaded_files:
        st.divider()
        st.caption("Configura filtros por dataset (máx 3 archivos).")

        for i, file in enumerate(uploaded_files[:3]):
            st.subheader(f"🛠️ Filtros: {file.name}")

            # Leer dataset para detectar columnas candidatas
            try:
                df_temp = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
            except Exception as e:
                st.warning(f"No se pudo leer {file.name} en sidebar: {e}")
                continue

            # Reset pointer
            try:
                file.seek(0)
            except Exception:
                pass

            date_candidates = detect_date_candidates(df_temp)
            cat_candidates = detect_category_candidates(df_temp)
            bod_candidates = detect_bodega_candidates(df_temp)

            # Fecha
            sel_date = st.selectbox(
                f"Columna de fecha para {file.name}",
                ['(ninguna)'] + date_candidates,
                key=f"sel_date_{i}"
            )
            dates = (None, None)
            if sel_date and sel_date != '(ninguna)':
                try:
                    df_temp[sel_date] = pd.to_datetime(df_temp[sel_date], errors='coerce')
                    min_d, max_d = df_temp[sel_date].min(), df_temp[sel_date].max()

                    if pd.isna(min_d) or pd.isna(max_d):
                        st.warning(f"La columna {sel_date} no contiene fechas reconocibles.")
                    else:
                        dates = st.date_input(
                            f"Rango de {sel_date}",
                            [min_d, max_d],
                            key=f"date_{i}"
                        )
                except Exception:
                    st.warning(f"No se pudo convertir {sel_date} a datetime.")

            # Categoría
            sel_cat = st.selectbox(
                f"Columna de categoría para {file.name}",
                ['(ninguna)'] + cat_candidates,
                key=f"sel_cat_{i}"
            )
            cat_sel = []
            if sel_cat and sel_cat != '(ninguna)':
                values = df_temp[sel_cat].dropna().unique().tolist()
                cat_sel = st.multiselect(f"Valores en {sel_cat}", values, key=f"cat_{i}")

            # Bodega/Tienda
            sel_bod = st.selectbox(
                f"Columna de bodega/tienda para {file.name}",
                ['(ninguna)'] + bod_candidates,
                key=f"sel_bod_{i}"
            )
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


# =========================
#        MAIN BODY
# =========================
if not uploaded_files:
    st.info("👈 Sube los archivos para ejecutar Healthcheck y luego EDA.")
    st.stop()

# Referencia inventario para integridad referencial (si existe)
inventory_ref = None
for f in uploaded_files:
    if "inventario" in f.name.lower():
        try:
            f.seek(0)
            inventory_ref = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
        except Exception:
            inventory_ref = None
        break


# Tabs: Healthcheck separado del EDA
tab_health, tab_eda = st.tabs(["🛡️ Healthcheck", "📊 EDA (Gráficas)"])

# =========================
#         TAB HEALTH
# =========================
with tab_health:
    st.subheader("🛡️ Módulo de Healthcheck (Calidad del Dataset)")
    st.caption("Incluye nulidad por columna, duplicados y magnitud de outliers + reglas de negocio.")

    for file in uploaded_files[:3]:
        st.header(f"Dataset: {file.name}")

        try:
            file.seek(0)
        except Exception:
            pass

        try:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        except pd.errors.EmptyDataError:
            st.warning(f"El archivo {file.name} está vacío y será omitido.")
            continue
        except Exception as e:
            st.warning(f"Error leyendo {file.name}: {e}")
            continue

        # --- APLICACIÓN DE FILTROS ---
        f = filtros_activos.get(file.name, {})
        df_filtered = df.copy()

        if f.get('date_col') and f.get('dates') and len(f['dates']) == 2 and f['dates'][0] and f['dates'][1]:
            df_filtered[f['date_col']] = pd.to_datetime(df_filtered[f['date_col']], errors='coerce')
            df_filtered = df_filtered[
                (df_filtered[f['date_col']].dt.date >= f['dates'][0]) &
                (df_filtered[f['date_col']].dt.date <= f['dates'][1])
            ]

        if f.get('cat_sel'):
            df_filtered = df_filtered[df_filtered[f['cat_col']].isin(f['cat_sel'])]

        if f.get('bod_sel'):
            df_filtered = df_filtered[df_filtered[f['bod_col']].isin(f['bod_sel'])]

        st.write(f"**Registros filtrados:** {len(df_filtered)}")

        # --- HEALTHCHECK ANTES / DESPUÉS ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔍 Antes (estado actual)")
            report_before, resumen_before = get_healthcheck_report(df_filtered)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Filas", resumen_before["filas"])
            m2.metric("Columnas", resumen_before["columnas"])
            m3.metric("Duplicados", resumen_before["duplicados"])
            m4.metric("% Nulos total", f'{resumen_before["pct_nulos_total"]}%')

            st.dataframe(report_before, use_container_width=True)

        # Limpieza
        df_clean = df_filtered.drop_duplicates().copy()

        decisiones = []
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                metodo, razon = justify_imputation(df_clean[col])
                decisiones.append({"Columna": col, "Decisión": f"Imputar con {metodo}", "Justificación": razon})

                if metodo == "Media":
                    fill_val = df_clean[col].mean()
                elif metodo == "Mediana":
                    fill_val = df_clean[col].median()
                else:
                    fill_val = df_clean[col].mode(dropna=True)[0]

                df_clean[col] = df_clean[col].fillna(fill_val)

        with col2:
            st.subheader("✅ Después (limpieza automática)")
            report_after, resumen_after = get_healthcheck_report(df_clean)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Filas finales", resumen_after["filas"])
            m2.metric("Duplicados eliminados", resumen_before["duplicados"])
            m3.metric("Cols con nulos", resumen_after["cols_con_nulos"])
            m4.metric("Cols con outliers", resumen_after["cols_con_outliers"])

            st.dataframe(report_after, use_container_width=True)

        # Reglas de negocio (hallazgos)
        st.subheader("🧠 Hallazgos de negocio (Data Quality Rules)")
        findings = dataset_business_checks(file.name, df_filtered, inventory_df=inventory_ref)

        if findings.empty:
            st.success("No se detectaron hallazgos de negocio con las reglas actuales.")
        else:
            st.dataframe(findings, use_container_width=True)

        # Decisiones éticas
        st.info("### ⚖️ Decisión Ética (Imputación)")
        if decisiones:
            st.table(pd.DataFrame(decisiones))
        else:
            st.write("Sin nulos detectados en la selección.")

        st.divider()


# =========================
#          TAB EDA
# =========================
with tab_eda:
    st.subheader("📊 EDA (Gráficas)")
    st.caption("Este tab queda separado para que el Healthcheck sea la primera capa de diagnóstico.")

    for file in uploaded_files[:3]:
        st.header(f"EDA: {file.name}")

        try:
            file.seek(0)
        except Exception:
            pass

        try:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        except Exception as e:
            st.warning(f"No se pudo leer {file.name}: {e}")
            continue

        # Aplicar filtros también al EDA
        f = filtros_activos.get(file.name, {})
        df_filtered = df.copy()

        if f.get('date_col') and f.get('dates') and len(f['dates']) == 2 and f['dates'][0] and f['dates'][1]:
            df_filtered[f['date_col']] = pd.to_datetime(df_filtered[f['date_col']], errors='coerce')
            df_filtered = df_filtered[
                (df_filtered[f['date_col']].dt.date >= f['dates'][0]) &
                (df_filtered[f['date_col']].dt.date <= f['dates'][1])
            ]

        if f.get('cat_sel'):
            df_filtered = df_filtered[df_filtered[f['cat_col']].isin(f['cat_sel'])]

        if f.get('bod_sel'):
            df_filtered = df_filtered[df_filtered[f['bod_col']].isin(f['bod_sel'])]

        st.write(f"**Registros filtrados para EDA:** {len(df_filtered)}")

        # Gráficas simples (placeholder inicial)
        num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()

        if not num_cols:
            st.info("No hay columnas numéricas para graficar.")
            st.divider()
            continue

        col_to_plot = st.selectbox(
            f"Selecciona una variable numérica ({file.name})",
            num_cols,
            key=f"eda_col_{file.name}"
        )

        fig, ax = plt.subplots()
        ax.hist(df_filtered[col_to_plot].dropna(), bins=30)
        ax.set_title(f"Histograma: {col_to_plot}")
        ax.set_xlabel(col_to_plot)
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)

        st.divider()

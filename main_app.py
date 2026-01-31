import streamlit as st
import pandas as pd
import numpy as np
import re

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

        if "Lead_Time_Dias" in df.columns:
            lt = df["Lead_Time_Dias"].astype(str)
            weird = int(lt.str.contains("-", na=False).sum())
            add("Lead Time en rango (texto)", weird, "Ej: '25-30 días' requiere estandarización")

    return pd.DataFrame(findings)


# =========================
#   INVENTARIO CLEAN RULES
# =========================
def normalize_categoria_value(x):
    """
    Normaliza valores similares: smartphone/smartphones -> smartphone
    ??? -> otros
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower()

    # reemplazo directo para ruido
    if s in ["???", "??", "?", "nan", "none", "null", "sin dato", "desconocido"]:
        return "otros"

    # limpieza de caracteres raros
    s = re.sub(r"\s+", " ", s)

    # normalización simple plural -> singular (casos conocidos)
    # (puedes extender el diccionario si aparecen más)
    mapping = {
        "smartphones": "smartphone",
        "smartphone": "smartphone",
        "laptops": "laptop",
        "laptop": "laptop",
        "tablets": "tablet",
        "tablet": "tablet",
        "accesorios": "accesorios",
        "accesorio": "accesorios",
    }

    return mapping.get(s, s)


def clean_inventario_central(df: pd.DataFrame):
    """
    Limpieza específica según tus directrices para inventario_central.
    Devuelve df_clean + decisiones_df.
    """
    decisiones = []
    df_clean = df.copy()

    before_rows = len(df_clean)

    # 1) Normalizar columna Categoria
    categoria_cols = [c for c in df_clean.columns if c.lower() in ["categoria", "categoría"]]
    if categoria_cols:
        cat_col = categoria_cols[0]
        before_unique = df_clean[cat_col].nunique(dropna=True)

        df_clean[cat_col] = df_clean[cat_col].apply(normalize_categoria_value)

        after_unique = df_clean[cat_col].nunique(dropna=True)
        decisiones.append({
            "Acción": "Normalización categórica",
            "Columna": cat_col,
            "Registros afectados": int(len(df_clean)),
            "Justificación": (
                "Se normalizaron valores similares (ej: smartphone/smartphones) "
                "para evitar fragmentación en análisis por categoría. "
                f"Únicos antes: {before_unique}, únicos después: {after_unique}."
            )
        })

        # reemplazo ??? -> otros (ya está dentro de normalize)
        replaced_otros = int((df_clean[cat_col] == "otros").sum())
        decisiones.append({
            "Acción": "Reasignación de categoría no informativa",
            "Columna": cat_col,
            "Registros afectados": replaced_otros,
            "Justificación": "Valores como '???' no aportan significado, se renombraron a 'otros'."
        })

    # 2) Stock: nulos -> 0, negativos se quedan
    stock_cols = [c for c in df_clean.columns if c.lower() in ["stock_actual", "stock", "cantidad_stock"]]
    if stock_cols:
        stock_col = stock_cols[0]
        # convertir a numérico (sin forzar limpieza de negativos)
        stock_numeric = pd.to_numeric(df_clean[stock_col], errors="coerce")

        nulls_before = int(stock_numeric.isna().sum())
        df_clean[stock_col] = stock_numeric.fillna(0)

        decisiones.append({
            "Acción": "Imputación de stock nulo",
            "Columna": stock_col,
            "Registros afectados": nulls_before,
            "Justificación": (
                "Se imputaron nulos con 0, interpretando ausencia de stock físico. "
                "Los valores negativos se conservaron (interpretación: unidades en lista de espera/backorder)."
            )
        })

    # 3) Costo_unitario: eliminar outliers extremos (fila)
    costo_cols = [c for c in df_clean.columns if c.lower() in ["costo_unitario_usd", "costo_unitario", "costo"]]
    if costo_cols:
        costo_col = costo_cols[0]
        costo_num = pd.to_numeric(df_clean[costo_col], errors="coerce")

        # criterio: IQR + filtro de extremos muy agresivos
        # (esto captura casos como 5 USD y 850000 USD si se salen de distribución)
        s = costo_num.dropna()
        removed = 0

        if not s.empty:
            Q1 = s.quantile(0.25)
            Q3 = s.quantile(0.75)
            IQR = Q3 - Q1

            if IQR > 0 and not pd.isna(IQR):
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                # extra guardrail: costos <= 0 también son sospechosos
                mask_out = (costo_num < lower) | (costo_num > upper) | (costo_num <= 0)
                removed = int(mask_out.sum())

                df_clean = df_clean.loc[~mask_out].copy()

                decisiones.append({
                    "Acción": "Eliminación de filas por costo outlier",
                    "Columna": costo_col,
                    "Registros afectados": removed,
                    "Justificación": (
                        "Se eliminaron registros con costos extremos (outliers) "
                        "que distorsionan métricas financieras. "
                        f"Umbrales IQR: [{lower:.2f}, {upper:.2f}] y además costo <= 0."
                    )
                })
            else:
                decisiones.append({
                    "Acción": "Revisión de outliers (sin acción)",
                    "Columna": costo_col,
                    "Registros afectados": 0,
                    "Justificación": "No se aplicó IQR porque la dispersión (IQR) fue 0 o inválida."
                })

        df_clean[costo_col] = pd.to_numeric(df_clean[costo_col], errors="coerce")

    # 4) Lead_Time: "25-30 días" -> "25-30", NaN -> "indefinido"
    lead_cols = [c for c in df_clean.columns if c.lower() in ["lead_time_dias", "lead_time", "leadtime"]]
    if lead_cols:
        lead_col = lead_cols[0]

        # contar nulos antes
        lead_before = df_clean[lead_col].copy()
        nulls_before = int(lead_before.isna().sum())

        def normalize_lead_time(v):
            if pd.isna(v):
                return "indefinido"

            s = str(v).strip().lower()

            if s in ["nan", "none", "null", ""]:
                return "indefinido"

            # ejemplo: "25-30 días" -> "25-30"
            s = s.replace("días", "").replace("dias", "").strip()

            # si viene algo como "25 - 30" -> "25-30"
            s = re.sub(r"\s*-\s*", "-", s)

            # mantener solo números y guion
            s = re.sub(r"[^0-9\-]", "", s)

            return s if s != "" else "indefinido"

        df_clean[lead_col] = df_clean[lead_col].apply(normalize_lead_time)

        decisiones.append({
            "Acción": "Estandarización Lead Time",
            "Columna": lead_col,
            "Registros afectados": int(len(df_clean)),
            "Justificación": (
                "Se estandarizó Lead_Time eliminando texto ('días') y dejando solo números/rangos. "
                f"Los NaN se reemplazaron por 'indefinido' (nulos detectados: {nulls_before})."
            )
        })

    after_rows = len(df_clean)
    removed_rows_total = before_rows - after_rows

    if removed_rows_total > 0:
        decisiones.append({
            "Acción": "Resumen de eliminación de filas",
            "Columna": "(dataset)",
            "Registros afectados": removed_rows_total,
            "Justificación": "Se eliminaron filas únicamente por reglas explícitas (outliers de costo_unitario)."
        })

    return df_clean, pd.DataFrame(decisiones)


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

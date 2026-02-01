import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from clean_inventario import clean_inventario_central
from clean_transacciones import clean_transacciones_logistica
from clean_feedback import clean_feedback_clientes

from filters import apply_filters_ui, render_filters_panel
from ai_module import generate_ai_strategy

from features_profitability import add_profitability_features, profitability_summary
from features_logistics import add_logistics_features, corr_delivery_vs_nps_by_city_warehouse, kpis_logistics_by_city_warehouse


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
    st.subheader("🤖 Configuración IA")
    api_key = st.text_input("Groq API Key (Llama-3)", type="password", help="Obtenla en console.groq.com")
    st.divider()
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


# ======================================================
#  TABS: separación Healthcheck/Limpieza vs EDA
# ======================================================
tab_health, tab_eda = st.tabs(["🛡️ Healthcheck + Limpieza + Integración", "📊 EDA Dashboard"])


# ======================================================
#  TAB 1: TODO lo actual (SIN cambiar funcionalidad)
# ======================================================
with tab_health:
    st.subheader("🧾 Diagnóstico y Limpieza (por dataset)")

    # ======================================================
    #  contenedores para integrar al final (NO rompe nada)
    # ======================================================
    inventario_clean = None
    transacciones_clean = None
    feedback_clean = None

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
        # CLEANING (specific rules by dataset)
        # ==============
        st.subheader("🧹 Limpieza aplicada")

        fname = file.name.lower()

        if "inventario" in fname:
            df_clean, decisiones_df = clean_inventario_central(df)
            st.success("✅ Se aplicó limpieza específica para inventario_central.")

            # guardar para integración
            inventario_clean = df_clean.copy()

        elif "transacciones" in fname or "logistica" in fname:
            df_clean, decisiones_df = clean_transacciones_logistica(df)
            st.success("✅ Se aplicó limpieza específica para transacciones_logistica.")

            # guardar para integración
            transacciones_clean = df_clean.copy()

        elif "feedback" in fname or "clientes" in fname:
            df_clean, decisiones_df = clean_feedback_clientes(df)
            st.success("✅ Se aplicó limpieza específica para feedback_clientes.")

            # guardar para integración
            feedback_clean = df_clean.copy()

        else:
            df_clean, decisiones_df = clean_dataset_generic(df)
            st.info("ℹ️ Se aplicó limpieza genérica (duplicados + imputación).")

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
        df_filtered = render_filters_panel(
            df_clean,
            file.name,
            key_prefix=file.name,
            report_func=get_healthcheck_report
        )

        # ======================================================
        #  NUEVO: MÓDULO DE INTELIGENCIA ARTIFICIAL (LLAMA-3)
        # ======================================================
        st.subheader("🤖 Recomendación Estratégica (IA Llama-3)")
        
        if st.button(f"Generar Estrategia con IA - {file.name}", key=f"ai_btn_{file.name}"):
            if not api_key:
                st.error("Debes ingresar la API Key en el panel lateral.")
            else:
                with st.spinner("Llama-3 analizando métricas en tiempo real..."):
                    estrategia = generate_ai_strategy(df_filtered, api_key, file.name)
                    st.markdown(f'<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;">{estrategia}</div>', unsafe_allow_html=True)

        # ==============
        # BUSINESS FINDINGS (sobre data limpia/filtrada)
        # ==============
        st.subheader("🧠 Hallazgos de negocio (Data Quality Rules)")
        findings = dataset_business_checks(file.name, df_filtered, inventory_df=inventory_ref)
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

    # ======================================================
    #  INTEGRACIÓN (JOIN) + DESCARGA SINGLE SOURCE OF TRUTH
    # ======================================================
    st.subheader("🔗 Integración de Datos: Single Source of Truth")
    st.caption("Se construye una tabla maestra usando transacciones como tabla principal (LEFT JOIN).")

    df_master = None

    if transacciones_clean is None:
        st.warning("⚠️ Para crear la Single Source of Truth debes subir transacciones_logistica.")
    else:
        df_master = transacciones_clean.copy()

        # ---------- JOIN con inventario por SKU ----------
        if inventario_clean is not None:
            sku_col_trans = None
            sku_col_inv = None

            for c in ["SKU", "sku", "Sku", "SKU_ID"]:
                if c in df_master.columns:
                    sku_col_trans = c
                    break

            for c in ["SKU", "sku", "Sku", "SKU_ID"]:
                if c in inventario_clean.columns:
                    sku_col_inv = c
                    break

            if sku_col_trans is not None and sku_col_inv is not None:
                df_master = df_master.merge(
                    inventario_clean,
                    left_on=sku_col_trans,
                    right_on=sku_col_inv,
                    how="left",
                    suffixes=("", "_inv"),
                    indicator=True
                )

                df_master["sku_en_inventario"] = df_master["_merge"].eq("both")
                sku_fantasmas = int((df_master["_merge"] == "left_only").sum())
                df_master.drop(columns=["_merge"], inplace=True)

                if "categoria" in df_master.columns:
                    df_master["categoria"] = df_master["categoria"].fillna("no_catalogado")

                st.success("✅ Join aplicado: transacciones + inventario (LEFT JOIN por SKU).")
                st.info(f"📌 SKUs fantasma detectados (ventas sin SKU en inventario): {sku_fantasmas}")
            else:
                st.warning("⚠️ No se pudo hacer join con inventario: no se encontró columna SKU/sku en ambos datasets.")
        else:
            st.info("ℹ️ No se encontró inventario_central. Se omitió el join con inventario.")

        # ---------- JOIN con feedback por transaccion_id ----------
        if feedback_clean is not None:
            if "Transaccion_ID" in df_master.columns and "Transaccion_ID" in feedback_clean.columns:
                feedback_one = feedback_clean.drop_duplicates(subset=["Transaccion_ID"]).copy()

                df_master = df_master.merge(
                    feedback_one,
                    on="Transaccion_ID",
                    how="left",
                    suffixes=("", "_fb")
                )

                st.success("✅ Join aplicado: master + feedback (LEFT JOIN por transaccion_id).")
            else:
                st.warning("⚠️ No se pudo hacer join con feedback: falta columna transaccion_id en alguno.")
        else:
            st.info("ℹ️ No se encontró feedback_clientes. Se omitió el join con feedback.")

        st.subheader("📌 Vista previa Single Source of Truth")
        st.dataframe(df_master.head(50), use_container_width=True)

        st.subheader("⬇️ Descargar Single Source of Truth")
        master_csv = df_master.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar single_source_of_truth.csv",
            data=master_csv,
            file_name="single_source_of_truth.csv",
            mime="text/csv",
            use_container_width=True
        )


# ======================================================
#  TAB 2: EDA Dashboard (SEPARADO)
# ======================================================
with tab_eda:
    st.title("📊 EDA Dashboard")
    st.caption("Exploración univariada, bivariada y visualizaciones típicas de un dashboard.")

    # Dataset base del EDA
    df_eda = None

    # Usar df_master si existe (Single Source of Truth)
    if "df_master" in locals() and df_master is not None and not df_master.empty:
        df_eda = df_master.copy()
        st.success("✅ Dataset EDA: Single Source of Truth (df_master).")
    else:
        st.warning("⚠️ No hay Single Source of Truth disponible. Usando fallback al primer dataset limpio encontrado.")
        if transacciones_clean is not None:
            df_eda = transacciones_clean.copy()
            st.info("📌 Fallback: transacciones_clean")
        elif inventario_clean is not None:
            df_eda = inventario_clean.copy()
            st.info("📌 Fallback: inventario_clean")
        elif feedback_clean is not None:
            df_eda = feedback_clean.copy()
            st.info("📌 Fallback: feedback_clean")

    if df_eda is None or df_eda.empty:
        st.error("❌ No hay datos para EDA.")
        st.stop()

    # ==========================================
    # Filtros del dashboard
    # ==========================================
    st.subheader("🎛️ Filtros (EDA)")
    df_dash = df_eda.copy()

    # ======================================================
    #  PREGUNTA 1: MARGEN NEGATIVO (Rentabilidad)
    # ======================================================
    st.subheader("💰 Pregunta 1: Fuga de Capital y Rentabilidad (Márgenes negativos)")
    st.caption("Detecta SKUs vendidos con margen negativo y evalúa si es crítico por canal.")
    
    # Trabajamos sobre el dataset EDA base (idealmente df_master)
    df_profit_base = df_eda.copy()
    
    df_profit, meta_profit = add_profitability_features(df_profit_base)
    
    # Mostrar warnings de detección de columnas
    if meta_profit.get("warnings"):
        for w in meta_profit["warnings"]:
            st.warning(f"⚠️ {w}")
    
    # Si no se pudieron crear columnas, detenemos esta sección
    if "margen_usd" not in df_profit.columns:
        st.error("❌ No se pudieron calcular márgenes (faltan columnas base).")
    else:
        # KPIs
        kpis = profitability_summary(df_profit)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ingreso total (USD)", f"{kpis.get('total_ingreso_usd', 0)}")
        c2.metric("Margen total (USD)", f"{kpis.get('total_margen_usd', 0)}")
        c3.metric("# Transacciones con margen negativo", f"{kpis.get('transacciones_margen_negativo', 0)}")
        c4.metric("Impacto margen negativo (USD)", f"{kpis.get('impacto_margen_negativo_usd', 0)}")
    
        st.divider()
    
        # -------------------------
        # Tabla Top pérdidas
        # -------------------------
        st.markdown("### 📉 Top transacciones con peor margen")
        cols_show = []
    
        # Mostrar columnas útiles si existen
        for c in ["Transaccion_ID", "SKU", "Canal_Venta", "Ciudad_Destino", "Bodega_Origen"]:
            if c in df_profit.columns:
                cols_show.append(c)
    
        cols_show += ["ingreso_total_usd", "costo_total_usd", "margen_usd"]
    
        worst = (
            df_profit[df_profit["margen_usd"].notna()]
            .sort_values("margen_usd", ascending=True)
            .head(20)
        )
    
        st.dataframe(worst[cols_show], use_container_width=True)
    
        st.divider()
    
        # -------------------------
        # Análisis por SKU (si existe)
        # -------------------------
        if meta_profit.get("col_sku") is not None:
            sku_col = meta_profit["col_sku"]
    
            st.markdown("### 🧾 SKUs con margen total negativo (agregado)")
            sku_agg = (
                df_profit.groupby(sku_col)
                .agg(
                    transacciones=("margen_usd", "count"),
                    ingreso_total=("ingreso_total_usd", "sum"),
                    margen_total=("margen_usd", "sum"),
                )
                .sort_values("margen_total", ascending=True)
                .head(15)
            )
    
            st.dataframe(sku_agg, use_container_width=True)
            st.bar_chart(sku_agg["margen_total"])
    
        else:
            st.info("ℹ️ No se puede agrupar por SKU porque no se detectó una columna SKU.")


    # ===========================
    # FEATURES LOGÍSTICAS (P2)
    # ===========================
    df_dash = add_logistics_features(df_dash, sla_days=5)

    st.subheader("🚚 Crisis Logística y Cuellos de Botella (P2)")
    st.caption("Correlación entre Tiempo de Entrega y NPS bajo por Ciudad/Bodega.")
    
    corr_zone = corr_delivery_vs_nps_by_city_warehouse(df_dash, min_rows=30)
    
    if corr_zone.empty:
        st.warning("⚠️ No hay suficientes datos para calcular correlación por zona (min_rows=30).")
    else:
        st.write("### 📉 Zonas con correlación más negativa (más crítico)")
        st.dataframe(corr_zone.head(15), use_container_width=True)
    
        st.write("### 📊 Correlación por zona (Top 15 crítico)")
        chart_df = corr_zone.head(15).set_index("zona_operativa")["corr_tiempo_vs_nps"]
        st.bar_chart(chart_df)
    
    kpis_zone = kpis_logistics_by_city_warehouse(df_dash, min_rows=30)
    
    if not kpis_zone.empty:
        st.write("### 🧾 KPIs logísticos por zona (para decidir cambio de operador)")
        st.dataframe(kpis_zone.head(15), use_container_width=True)
    
        st.write("### 🏁 Ranking de zonas por score de riesgo logístico")
        st.bar_chart(kpis_zone.head(15).set_index("zona_operativa")["score_riesgo_logistico"])
    
    st.divider()

    # ===========================
    # DIAGNÓSTICO DE FIDELIDAD (P4)
    # ===========================
    st.subheader("🧾 Diagnóstico de Fidelidad (P4)")
    st.caption("¿Existen categorías con alta disponibilidad (stock alto) pero sentimiento cliente negativo?")

    # intentamos usar los datasets limpios cargados en la pestaña de limpieza
    inv = inventario_clean if 'inventario_clean' in locals() else None
    fb = feedback_clean if 'feedback_clean' in locals() else None

    if inv is None and fb is None:
        st.warning("No hay datasets limpios de inventario ni feedback disponibles para este diagnóstico.")
    else:
        # detectar columnas relevantes
        def find_col_ci(df, candidates):
            if df is None:
                return None
            for c in df.columns:
                if c.lower() in [x.lower() for x in candidates]:
                    return c
            return None

        stock_col = find_col_ci(inv, ["Stock_Actual", "stock_actual", "stock", "cantidad_stock"]) if inv is not None else None
        cat_col = find_col_ci(inv, ["categoria", "categoria_producto", "categoria_producto" ]) if inv is not None else None
        sku_inv = find_col_ci(inv, ["SKU", "sku", "Sku", "SKU_ID"]) if inv is not None else None

        sku_fb = find_col_ci(fb, ["SKU", "sku", "Sku", "SKU_ID"]) if fb is not None else None
        nps_fb_col = find_col_ci(fb, ["satisfaccion_NPS", "satisfaccion_nps", "nps", "NPS"]) if fb is not None else None

        # Preparar agregados inventario por categoría
        inv_agg = None
        if inv is not None and cat_col is not None and stock_col is not None:
            inv_temp = inv.copy()
            inv_temp[stock_col] = pd.to_numeric(inv_temp[stock_col], errors='coerce').fillna(0)
            inv_agg = (
                inv_temp.groupby(cat_col)[stock_col]
                .agg(total_stock='sum', avg_stock='mean', n_items='count')
                .reset_index()
                .sort_values('total_stock', ascending=False)
            )

        # Preparar agregados feedback (por SKU o categoría si es posible)
        fb_agg = None
        if fb is not None:
            fb_temp = fb.copy()
            if nps_fb_col is not None:
                fb_temp['nps_num'] = pd.to_numeric(fb_temp[nps_fb_col], errors='coerce')
            else:
                fb_temp['nps_num'] = pd.NA

            # ratings
            rating_prod_col = find_col_ci(fb_temp, ["Rating_Producto", "rating_producto", "Rating_Producto", "rating"]) 
            rating_log_col = find_col_ci(fb_temp, ["Rating_Logistica", "rating_logistica", "rating_logistica"]) 
            if rating_prod_col is not None:
                fb_temp['rating_producto_num'] = pd.to_numeric(fb_temp[rating_prod_col], errors='coerce')
            else:
                fb_temp['rating_producto_num'] = pd.NA

            if rating_log_col is not None:
                fb_temp['rating_logistica_num'] = pd.to_numeric(fb_temp[rating_log_col], errors='coerce')
            else:
                fb_temp['rating_logistica_num'] = pd.NA

            # si tenemos SKU en ambos y categoria en inventario, join para obtener categoria por feedback
            if sku_fb is not None and sku_inv is not None and inv is not None and cat_col is not None:
                merged = fb_temp.merge(inv[[sku_inv, cat_col]].drop_duplicates(), left_on=sku_fb, right_on=sku_inv, how='left')
                grp = merged.groupby(cat_col).agg(
                    n_feedback=('nps_num','count'),
                    avg_nps=('nps_num','mean'),
                    avg_rating_producto=('rating_producto_num','mean'),
                    avg_rating_logistica=('rating_logistica_num','mean')
                ).reset_index()
                pct = merged.groupby(cat_col)['nps_num'].apply(lambda x: (x<=6).mean() if x.notna().any() else np.nan).reset_index(name='pct_nps_bajo')
                fb_agg = grp.merge(pct, on=cat_col, how='left')
                if 'pct_nps_bajo' in fb_agg.columns:
                    fb_agg['pct_nps_bajo'] = (fb_agg['pct_nps_bajo'] * 100).round(2)
            else:
                # fallback: agregamos por SKU si no hay categoria
                if sku_fb is not None:
                    fb_agg = (
                        fb_temp.groupby(sku_fb)
                        .agg(n_feedback=('nps_num','count'), avg_nps=('nps_num','mean'))
                        .reset_index()
                    )

        # ------------------ GRAFICA 1: Avg NPS por categoria ------------------
        if inv_agg is not None and fb_agg is not None and cat_col in inv_agg.columns and cat_col in fb_agg.columns:
            merged_cat = inv_agg.merge(fb_agg, on=cat_col, how='left')
            merged_cat['avg_nps'] = pd.to_numeric(merged_cat.get('avg_nps', pd.Series([np.nan]*len(merged_cat))), errors='coerce')

            # permitir elegir métrica de sentimiento para graficar
            metrics_available = []
            if 'avg_nps' in merged_cat.columns:
                metrics_available.append(('avg_nps','Satisfacción NPS'))
            if 'avg_rating_producto' in merged_cat.columns:
                metrics_available.append(('avg_rating_producto','Rating Producto'))
            if 'avg_rating_logistica' in merged_cat.columns:
                metrics_available.append(('avg_rating_logistica','Rating Logística'))

            chosen_label = 'avg_nps'
            sel_label = 'Satisfacción NPS'
            if metrics_available:
                opts = [lab for (_,lab) in metrics_available]
                sel_label = st.selectbox('Métrica de sentimiento a graficar', opts, index=0)
                label_map = {lab:col for (col,lab) in metrics_available}
                chosen_label = label_map.get(sel_label, 'avg_nps')

            st.markdown("**1) Promedio de Sentimiento por Categoría vs Stock**")
            fig1 = px.bar(
                merged_cat.sort_values(chosen_label, ascending=True),
                x=cat_col,
                y=chosen_label,
                color='total_stock',
                color_continuous_scale='RdYlGn_r',
                labels={cat_col: 'Categoría', chosen_label: sel_label, 'total_stock': 'Stock total'},
                title=f'{sel_label} por Categoría (color = stock total)'
            )
            fig1.update_layout(xaxis={'categoryorder':'total descending'}, height=450)
            st.plotly_chart(fig1, use_container_width=True)

            # Barra auxiliar: stock por categoria
            st.markdown("**2) Stock total por Categoría**")
            fig2 = px.bar(merged_cat.sort_values('total_stock', ascending=False), x=cat_col, y='total_stock', title='Stock total por Categoría')
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

            # Scatter: stock vs métrica seleccionada
            st.markdown(f"**3) Stock vs {sel_label} (bubble size = n_feedback)**")
            merged_cat['n_feedback'] = merged_cat.get('n_feedback', merged_cat.get('n_items', 0)).fillna(0)
            fig3 = px.scatter(
                merged_cat,
                x='total_stock',
                y=chosen_label,
                size='n_feedback',
                hover_name=cat_col,
                title=f'Stock total vs {sel_label} por Categoría',
                labels={'total_stock':'Stock total', chosen_label:sel_label}
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown("**Interpretación rápida:** categorías con alto stock y sentimiento (NPS/Rating) bajo aparecen con alta stock a la derecha y valores de sentimiento bajos. Esas categorías son candidatas a revisar: calidad de producto, desajuste entre expectativas y descripción, o problemas logísticos/precio.")

        else:
            st.info("No se pudieron construir las gráficas por falta de columnas comunes (categoria/stock en inventario y NPS en feedback). Si existe SKU en ambos datasets, súbelos para emparejar datos.")


    st.divider()

    # -------------------------
    # Comparación por canal (Online vs otros)
    # -------------------------
    if meta_profit.get("col_channel") is not None:
        canal_col = meta_profit["col_channel"]

        st.markdown("### 🌐 Comparación de margen por canal")
        canal_agg = (
            df_profit.groupby(canal_col)
            .agg(
                transacciones=("margen_usd", "count"),
                ingreso_total=("ingreso_total_usd", "sum"),
                margen_total=("margen_usd", "sum"),
                pct_margen_neg=("margen_negativo", "mean"),
            )
        )

        canal_agg["pct_margen_neg"] = (canal_agg["pct_margen_neg"] * 100).round(2)

        st.dataframe(canal_agg, use_container_width=True)

        st.markdown("**% transacciones con margen negativo por canal**")
        st.bar_chart(canal_agg["pct_margen_neg"])

    else:
        st.info("ℹ️ No se puede comparar por canal porque no se detectó Canal_Venta.")


    with st.expander("🔎 Configurar filtros", expanded=True):
        cat_cols = df_dash.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df_dash.select_dtypes(include=[np.number]).columns.tolist()

        col_filter_cat = st.selectbox(
            "Filtro categórico (opcional)",
            ["(ninguno)"] + cat_cols
        )

        if col_filter_cat != "(ninguno)":
            options = sorted(df_dash[col_filter_cat].dropna().astype(str).unique().tolist())
            selected_opts = st.multiselect(
                f"Valores en {col_filter_cat}",
                options,
                default=options[: min(10, len(options))]
            )
            if selected_opts:
                df_dash = df_dash[df_dash[col_filter_cat].astype(str).isin(selected_opts)]

        col_filter_num = st.selectbox(
            "Filtro numérico por rango (opcional)",
            ["(ninguno)"] + num_cols
        )

        if col_filter_num != "(ninguno)":
            s = pd.to_numeric(df_dash[col_filter_num], errors="coerce").dropna()
            if not s.empty:
                min_v, max_v = float(s.min()), float(s.max())
                r = st.slider(
                    f"Rango para {col_filter_num}",
                    min_value=min_v,
                    max_value=max_v,
                    value=(min_v, max_v)
                )
                df_dash = df_dash[(df_dash[col_filter_num] >= r[0]) & (df_dash[col_filter_num] <= r[1])]

    st.write(f"📌 Filas después de filtros: **{len(df_dash)}**")
    if df_dash.empty:
        st.warning("⚠️ Con los filtros actuales no hay filas para graficar.")
        st.stop()

    # ==========================================
    # KPIs
    # ==========================================
    st.subheader("📌 KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Filas", f"{df_dash.shape[0]}")
    k2.metric("Columnas", f"{df_dash.shape[1]}")
    k3.metric("Nulos totales", f"{int(df_dash.isnull().sum().sum())}")
    k4.metric("Duplicados", f"{int(df_dash.duplicated().sum())}")

    extra = st.columns(4)

    if "sku_en_inventario" in df_dash.columns:
        extra[0].metric("% SKU en inventario", f"{round(df_dash['sku_en_inventario'].mean()*100,2)}%")

    if "rating_producto" in df_dash.columns:
        extra[1].metric("Rating promedio", f"{round(pd.to_numeric(df_dash['rating_producto'], errors='coerce').mean(),2)}")

    if "satisfaccion_NPS" in df_dash.columns:
        extra[2].metric("NPS promedio", f"{round(pd.to_numeric(df_dash['satisfaccion_NPS'], errors='coerce').mean(),2)}")

    if "ticket_soporte_abierto" in df_dash.columns:
        pct_tickets = round((df_dash["ticket_soporte_abierto"].astype(str).str.upper() == "SI").mean() * 100, 2)
        extra[3].metric("% con ticket soporte", f"{pct_tickets}%")

    st.divider()

    # ==========================================
    # Univariado numérico
    # ==========================================
    st.subheader("📈 Univariado (Numéricas)")
    num_cols = df_dash.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols:
        st.info("No hay columnas numéricas.")
    else:
        col_num = st.selectbox("Columna numérica", num_cols)

        s = pd.to_numeric(df_dash[col_num], errors="coerce")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Media", f"{s.mean():.2f}" if not np.isnan(s.mean()) else "N/A")
        c2.metric("Mediana", f"{s.median():.2f}" if not np.isnan(s.median()) else "N/A")
        c3.metric("Std", f"{s.std():.2f}" if not np.isnan(s.std()) else "N/A")
        c4.metric("Skew", f"{s.skew():.2f}" if not np.isnan(s.skew()) else "N/A")

        st.write("**Histograma (bins)**")
        st.bar_chart(s.dropna().value_counts(bins=20).sort_index())

        st.write("**Describe**")
        st.dataframe(s.describe().to_frame().T, use_container_width=True)

    st.divider()

    # ==========================================
    # Univariado categórico
    # ==========================================
    st.subheader("🧩 Univariado (Categóricas)")
    cat_cols = df_dash.select_dtypes(include=["object", "category"]).columns.tolist()

    if not cat_cols:
        st.info("No hay columnas categóricas.")
    else:
        col_cat = st.selectbox("Columna categórica", cat_cols)

        freq = (
            df_dash[col_cat]
            .astype(str)
            .fillna("N/A")
            .value_counts()
            .head(15)
        )

        st.write("**Top 15**")
        st.dataframe(freq.reset_index().rename(columns={"index": col_cat, col_cat: "conteo"}), use_container_width=True)

        st.write("**Gráfico**")
        st.bar_chart(freq)

    st.divider()

    # ==========================================
    # Bivariado num vs num
    # ==========================================
    st.subheader("🔁 Bivariado (Numérica vs Numérica)")
    num_cols = df_dash.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        st.info("Se requieren al menos 2 numéricas.")
    else:
        x_col = st.selectbox("Eje X", num_cols, index=0)
        y_col = st.selectbox("Eje Y", num_cols, index=1)

        scatter_df = df_dash[[x_col, y_col]].dropna()
        st.scatter_chart(scatter_df, x=x_col, y=y_col)

        corr_val = scatter_df[x_col].corr(scatter_df[y_col])
        st.metric("Correlación (Pearson)", f"{corr_val:.3f}" if corr_val is not None else "N/A")

    st.divider()

    # ==========================================
    # Bivariado cat vs num
    # ==========================================
    st.subheader("📦 Bivariado (Categórica vs Numérica)")
    cat_cols = df_dash.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df_dash.select_dtypes(include=[np.number]).columns.tolist()

    if not cat_cols or not num_cols:
        st.info("Se requiere al menos 1 categórica y 1 numérica.")
    else:
        col_cat2 = st.selectbox("Categoría", cat_cols)
        col_num2 = st.selectbox("Métrica numérica", num_cols)

        grouped = (
            df_dash.groupby(col_cat2)[col_num2]
            .mean(numeric_only=True)
            .sort_values(ascending=False)
            .head(15)
        )

        st.dataframe(grouped.reset_index().rename(columns={col_num2: f"promedio_{col_num2}"}), use_container_width=True)
        st.bar_chart(grouped)

    st.divider()

    # ==========================================
    # Matriz correlación
    # ==========================================
    st.subheader("🧠 Matriz de Correlación")
    num_cols = df_dash.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        st.info("No hay suficientes numéricas para correlación.")
    else:
        corr = df_dash[num_cols].corr(numeric_only=True)
        st.dataframe(corr, use_container_width=True)

    st.divider()

    st.subheader("📄 Vista previa del dataset filtrado (EDA)")
    st.dataframe(df_dash.head(100), use_container_width=True)





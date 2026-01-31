import pandas as pd
import numpy as np
import streamlit as st


def detect_date_candidates(df: pd.DataFrame):
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


def detect_category_candidates(df: pd.DataFrame):
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


def detect_bodega_candidates(df: pd.DataFrame):
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


def apply_filters_ui(df: pd.DataFrame, file_name: str, key_prefix: str = '') -> pd.DataFrame:
    """Muestra UI para seleccionar filtros y aplica las selecciones sobre el df limpio.

    Retorna el DataFrame filtrado.
    """
    if df is None or df.empty:
        st.info("Dataset limpio vacío — no hay filtros aplicables.")
        return df

    date_candidates = detect_date_candidates(df)
    cat_candidates = detect_category_candidates(df)
    bod_candidates = detect_bodega_candidates(df)

    sel_date = st.selectbox(f"Columna de fecha ({file_name})", ['(ninguna)'] + date_candidates, key=f"{key_prefix}_sel_date")
    date_range = None
    if sel_date and sel_date != '(ninguna)':
        try:
            df[sel_date] = pd.to_datetime(df[sel_date], errors='coerce')
            min_d, max_d = df[sel_date].min(), df[sel_date].max()
            if pd.isna(min_d) or pd.isna(max_d):
                st.warning(f"La columna {sel_date} no contiene fechas reconocibles.")
            else:
                date_range = st.date_input(f"Rango de {sel_date}", [min_d, max_d], key=f"{key_prefix}_date_range")
        except Exception:
            st.warning(f"No se pudo convertir {sel_date} a datetime.")

    sel_cat = st.selectbox(f"Columna de categoría ({file_name})", ['(ninguna)'] + cat_candidates, key=f"{key_prefix}_sel_cat")
    cat_vals = []
    if sel_cat and sel_cat != '(ninguna)':
        cat_vals = st.multiselect(f"Valores en {sel_cat}", df[sel_cat].dropna().unique().tolist(), key=f"{key_prefix}_cat_vals")

    sel_bod = st.selectbox(f"Columna de bodega/tienda ({file_name})", ['(ninguna)'] + bod_candidates, key=f"{key_prefix}_sel_bod")
    bod_vals = []
    if sel_bod and sel_bod != '(ninguna)':
        bod_vals = st.multiselect(f"Valores en {sel_bod}", df[sel_bod].dropna().unique().tolist(), key=f"{key_prefix}_bod_vals")

    df_filtered = df.copy()
    if sel_date and sel_date != '(ninguna)' and date_range and len(date_range) == 2:
        start, end = date_range[0], date_range[1]
        df_filtered = df_filtered[(df_filtered[sel_date].dt.date >= start) & (df_filtered[sel_date].dt.date <= end)]

    if sel_cat and sel_cat != '(ninguna)' and cat_vals:
        df_filtered = df_filtered[df_filtered[sel_cat].isin(cat_vals)]

    if sel_bod and sel_bod != '(ninguna)' and bod_vals:
        df_filtered = df_filtered[df_filtered[sel_bod].isin(bod_vals)]

    return df_filtered


def render_filters_panel(df: pd.DataFrame, file_name: str, key_prefix: str, report_func):
    """Renderiza la UI de filtros y muestra el healthcheck usando `report_func` callback.

    `report_func` debe ser una función que reciba un DataFrame y devuelva (report_df, resumen_dict).
    Devuelve el DataFrame filtrado.
    """
    with st.expander("🔎 Aplicar filtros al dataset limpio (fecha/categoría/bodega)", expanded=False):
        df_filtered = apply_filters_ui(df, file_name, key_prefix=key_prefix)

        try:
            report_filt, resumen_filt = report_func(df_filtered)
        except Exception:
            # si falla el callback, construimos un resumen mínimo
            report_filt = pd.DataFrame()
            resumen_filt = {
                "filas": int(df_filtered.shape[0]) if df_filtered is not None else 0,
                "columnas": int(df_filtered.shape[1]) if df_filtered is not None else 0,
                "duplicados": int(df_filtered.duplicated().sum()) if df_filtered is not None else 0,
                "pct_nulos_total": round((df_filtered.isnull().sum().sum() / (df_filtered.size)) * 100, 2) if df_filtered is not None and df_filtered.size else 0.0
            }

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Filas (filtradas)", resumen_filt.get("filas", 0))
        m2.metric("Columnas", resumen_filt.get("columnas", 0))
        m3.metric("Duplicados", resumen_filt.get("duplicados", 0))
        m4.metric("% Nulos total", f"{resumen_filt.get('pct_nulos_total', 0.0)}%")

        if not report_filt.empty:
            st.dataframe(report_filt, use_container_width=True)
        else:
            st.write("No hay reporte detallado disponible para el subset filtrado.")

        # descarga del subset filtrado
        sub_csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Descargar subset filtrado: {file_name}",
            data=sub_csv,
            file_name=f"{file_name.replace('.csv','').replace('.xlsx','')}_filtered.csv",
            mime="text/csv",
            use_container_width=True
        )

        return df_filtered

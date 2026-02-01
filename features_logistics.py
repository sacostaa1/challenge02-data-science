import pandas as pd
import numpy as np


def _find_col(df: pd.DataFrame, candidates: list[str]):
    """Retorna el primer nombre de columna que exista en df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_logistics_features(df: pd.DataFrame, sla_days: int = 5) -> pd.DataFrame:
    """
    Agrega features mínimas para análisis logístico con tus columnas reales:
    - tiempo_entrega_dias (desde Tiempo_Entrega_Real)
    - nps (desde Satisfaccion_NPS)
    - nps_bajo (<= 6)
    - sla_incumplido (tiempo_entrega_dias > sla_days)

    NOTA:
    - NO crea zona_operativa porque el análisis se hará por (Ciudad_Destino, Bodega_Origen)
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # -----------------------------
    # Detectar columnas reales
    # -----------------------------
    col_city = _find_col(out, ["Ciudad_Destino", "Ciudad", "ciudad"])
    col_warehouse = _find_col(out, ["Bodega_Origen", "Bodega", "bodega"])
    col_nps = _find_col(out, ["Satisfaccion_NPS", "satisfaccion_NPS", "NPS", "nps"])
    col_lead = _find_col(out, ["Tiempo_Entrega_Real", "Tiempo_Entrega", "tiempo_entrega"])

    # -----------------------------
    # 1) Tiempo de entrega (días)
    # -----------------------------
    if col_lead is not None:
        out["tiempo_entrega_dias"] = pd.to_numeric(out[col_lead], errors="coerce")
    else:
        # Si no existe, dejamos NaN (no inventamos con fechas)
        out["tiempo_entrega_dias"] = np.nan

    # Tiempos negativos no tienen sentido
    out.loc[out["tiempo_entrega_dias"] < 0, "tiempo_entrega_dias"] = np.nan

    # -----------------------------
    # 2) NPS numérico
    # -----------------------------
    if col_nps is not None:
        out["nps"] = pd.to_numeric(out[col_nps], errors="coerce")
    else:
        out["nps"] = np.nan

    # -----------------------------
    # 3) NPS bajo (detractores)
    # -----------------------------
    out["nps_bajo"] = (out["nps"] <= 6).astype(int)

    # -----------------------------
    # 4) SLA incumplido
    # -----------------------------
    out["sla_incumplido"] = (out["tiempo_entrega_dias"] > sla_days).astype(int)

    # -----------------------------
    # 5) Asegurar columnas base (para agrupaciones)
    # -----------------------------
    # Si no existen, las creamos como NaN para evitar errores downstream
    if col_city is None:
        out["Ciudad_Destino"] = np.nan
    if col_warehouse is None:
        out["Bodega_Origen"] = np.nan

    return out


def corr_delivery_vs_nps_by_city_warehouse(
    df: pd.DataFrame,
    min_rows: int = 30
) -> pd.DataFrame:
    """
    Correlación (Pearson) entre tiempo_entrega_dias y nps por combinación:
    (Ciudad_Destino, Bodega_Origen)

    Devuelve un ranking de los pares más críticos:
    - correlación más NEGATIVA (a mayor tiempo, peor NPS)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    needed = ["Ciudad_Destino", "Bodega_Origen", "tiempo_entrega_dias", "nps"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()

    base = df[needed].copy()
    base["tiempo_entrega_dias"] = pd.to_numeric(base["tiempo_entrega_dias"], errors="coerce")
    base["nps"] = pd.to_numeric(base["nps"], errors="coerce")

    # Solo filas completas
    base = base.dropna(subset=["Ciudad_Destino", "Bodega_Origen", "tiempo_entrega_dias", "nps"])

    rows = []
    grouped = base.groupby(["Ciudad_Destino", "Bodega_Origen"])

    for (city, wh), g in grouped:
        if len(g) < min_rows:
            continue

        corr = g["tiempo_entrega_dias"].corr(g["nps"])

        rows.append({
            "Ciudad_Destino": city,
            "Bodega_Origen": wh,
            "n": int(len(g)),
            "corr_tiempo_vs_nps": float(corr) if corr is not None else np.nan,
            "avg_tiempo_entrega": float(g["tiempo_entrega_dias"].mean()),
            "avg_nps": float(g["nps"].mean()),
            "pct_nps_bajo": float((g["nps"] <= 6).mean() * 100)
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Más crítico: correlación más negativa
    out = out.sort_values(by="corr_tiempo_vs_nps", ascending=True).reset_index(drop=True)
    return out


def kpis_logistics_by_city_warehouse(
    df: pd.DataFrame,
    min_rows: int = 30
) -> pd.DataFrame:
    """
    KPIs logísticos por (Ciudad_Destino, Bodega_Origen):
    - promedio tiempo entrega
    - % NPS bajo
    - % SLA incumplido
    - score_riesgo_logistico
    """
    if df is None or df.empty:
        return pd.DataFrame()

    needed = ["Ciudad_Destino", "Bodega_Origen", "tiempo_entrega_dias", "nps_bajo", "sla_incumplido"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()

    base = df.dropna(subset=["Ciudad_Destino", "Bodega_Origen"]).copy()

    g = base.groupby(["Ciudad_Destino", "Bodega_Origen"])

    out = g.agg(
        n=("tiempo_entrega_dias", "size"),
        avg_tiempo_entrega=("tiempo_entrega_dias", "mean"),
        pct_nps_bajo=("nps_bajo", "mean"),
        pct_sla_incumplido=("sla_incumplido", "mean"),
    ).reset_index()

    out = out[out["n"] >= min_rows].copy()

    out["pct_nps_bajo"] = (out["pct_nps_bajo"] * 100).round(2)
    out["pct_sla_incumplido"] = (out["pct_sla_incumplido"] * 100).round(2)

    # Score simple para ranking (más alto = peor)
    out["score_riesgo_logistico"] = (
        out["avg_tiempo_entrega"].fillna(0) * 0.4
        + out["pct_nps_bajo"].fillna(0) * 0.4
        + out["pct_sla_incumplido"].fillna(0) * 0.2
    ).round(2)

    out = out.sort_values(by="score_riesgo_logistico", ascending=False).reset_index(drop=True)
    return out

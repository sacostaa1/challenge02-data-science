import pandas as pd
import numpy as np


def _find_col(df: pd.DataFrame, candidates: list[str]):
    """Retorna el primer nombre de columna que exista en df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_datetime_safe(s: pd.Series):
    return pd.to_datetime(s, errors="coerce")


def add_logistics_features(df: pd.DataFrame, sla_days: int = 5) -> pd.DataFrame:
    """
    Agrega features mínimas y valiosas para análisis logístico:
    - tiempo_entrega_dias
    - nps (numérico)
    - nps_bajo
    - zona_operativa (ciudad|bodega)
    - sla_incumplido
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # -----------------------------
    # Detectar columnas relevantes
    # -----------------------------
    col_city = _find_col(out, ["Ciudad", "ciudad", "CITY", "city"])
    col_warehouse = _find_col(out, ["Bodega", "bodega", "Warehouse", "warehouse", "Centro_Distribucion"])
    col_nps = _find_col(out, ["satisfaccion_NPS", "NPS", "nps"])

    # Posibles fechas (dependen del dataset)
    col_fecha_compra = _find_col(out, ["Fecha_Compra", "fecha_compra", "Fecha", "fecha"])
    col_fecha_despacho = _find_col(out, ["Fecha_Despacho", "fecha_despacho", "Fecha_Salida", "fecha_salida"])
    col_fecha_entrega = _find_col(out, ["Fecha_Entrega", "fecha_entrega", "Entrega", "entrega"])

    # -----------------------------
    # 1) Tiempo de entrega (días)
    # -----------------------------
    # Si ya existe una columna "Tiempo_Entrega" o similar, la usamos.
    col_lead = _find_col(out, ["Tiempo_Entrega", "tiempo_entrega", "Tiempo_Entrega_Dias", "lead_time_dias"])

    if col_lead is not None:
        out["tiempo_entrega_dias"] = pd.to_numeric(out[col_lead], errors="coerce")
    else:
        # Calculamos usando fechas si están disponibles
        tiempo = pd.Series([np.nan] * len(out))

        if col_fecha_entrega and col_fecha_despacho:
            f_ent = _to_datetime_safe(out[col_fecha_entrega])
            f_des = _to_datetime_safe(out[col_fecha_despacho])
            tiempo = (f_ent - f_des).dt.total_seconds() / (3600 * 24)

        elif col_fecha_entrega and col_fecha_compra:
            f_ent = _to_datetime_safe(out[col_fecha_entrega])
            f_com = _to_datetime_safe(out[col_fecha_compra])
            tiempo = (f_ent - f_com).dt.total_seconds() / (3600 * 24)

        out["tiempo_entrega_dias"] = pd.to_numeric(tiempo, errors="coerce")

    # Limpieza básica: tiempos negativos no tienen sentido
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
    # 4) Zona operativa (Ciudad | Bodega)
    # -----------------------------
    if col_city and col_warehouse:
        out["zona_operativa"] = (
            out[col_city].astype(str).fillna("N/A").str.strip()
            + " | " +
            out[col_warehouse].astype(str).fillna("N/A").str.strip()
        )
    elif col_city:
        out["zona_operativa"] = out[col_city].astype(str).fillna("N/A").str.strip()
    elif col_warehouse:
        out["zona_operativa"] = out[col_warehouse].astype(str).fillna("N/A").str.strip()
    else:
        out["zona_operativa"] = "N/A"

    # -----------------------------
    # 5) SLA incumplido
    # -----------------------------
    out["sla_incumplido"] = (out["tiempo_entrega_dias"] > sla_days).astype(int)

    return out


def zone_corr_delivery_vs_nps(df: pd.DataFrame, min_rows: int = 30) -> pd.DataFrame:
    """
    Calcula correlación (Pearson) entre tiempo_entrega_dias y nps por zona_operativa.
    Devuelve ranking de zonas con mayor correlación negativa (más preocupante).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    needed = ["zona_operativa", "tiempo_entrega_dias", "nps"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()

    base = df[needed].copy()
    base["tiempo_entrega_dias"] = pd.to_numeric(base["tiempo_entrega_dias"], errors="coerce")
    base["nps"] = pd.to_numeric(base["nps"], errors="coerce")
    base = base.dropna(subset=["zona_operativa", "tiempo_entrega_dias", "nps"])

    rows = []
    for zone, g in base.groupby("zona_operativa"):
        if len(g) < min_rows:
            continue
        corr = g["tiempo_entrega_dias"].corr(g["nps"])
        rows.append({
            "zona_operativa": zone,
            "n": int(len(g)),
            "corr_tiempo_vs_nps": float(corr) if corr is not None else np.nan,
            "avg_tiempo_entrega": float(g["tiempo_entrega_dias"].mean()),
            "avg_nps": float(g["nps"].mean())
        })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    # Más crítico: correlación más NEGATIVA
    out = out.sort_values(by="corr_tiempo_vs_nps", ascending=True).reset_index(drop=True)
    return out


def zone_kpis_logistics(df: pd.DataFrame, min_rows: int = 30) -> pd.DataFrame:
    """
    KPIs por zona:
    - promedio de tiempo entrega
    - % NPS bajo
    - % SLA incumplido
    """
    if df is None or df.empty:
        return pd.DataFrame()

    needed = ["zona_operativa", "tiempo_entrega_dias", "nps_bajo", "sla_incumplido"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()

    g = df.dropna(subset=["zona_operativa"]).groupby("zona_operativa")

    out = g.agg(
        n=("zona_operativa", "size"),
        avg_tiempo_entrega=("tiempo_entrega_dias", "mean"),
        pct_nps_bajo=("nps_bajo", "mean"),
        pct_sla_incumplido=("sla_incumplido", "mean"),
    ).reset_index()

    out = out[out["n"] >= min_rows].copy()

    # % bonitos
    out["pct_nps_bajo"] = (out["pct_nps_bajo"] * 100).round(2)
    out["pct_sla_incumplido"] = (out["pct_sla_incumplido"] * 100).round(2)

    # ranking por combinación de "malo"
    # (esto no es una columna auxiliar, es un score útil)
    out["score_riesgo_logistico"] = (
        out["avg_tiempo_entrega"].fillna(0) * 0.4
        + out["pct_nps_bajo"].fillna(0) * 0.4
        + out["pct_sla_incumplido"].fillna(0) * 0.2
    ).round(2)

    out = out.sort_values(by="score_riesgo_logistico", ascending=False).reset_index(drop=True)
    return out

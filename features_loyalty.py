import pandas as pd
import numpy as np


def _find_col(df: pd.DataFrame, candidates: list[str]):
    """Retorna el primer nombre de columna que exista en df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_loyalty_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features para diagnóstico de fidelidad:
    - stock_actual_num
    - rating_producto_num
    - rating_logistica_num
    - nps_num
    - sentimiento_negativo (flag)
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    col_cat = _find_col(out, ["Categoria", "categoria", "CATEGORY", "category"])
    col_stock = _find_col(out, ["Stock_Actual", "stock_actual", "Stock", "stock"])
    col_rp = _find_col(out, ["Rating_Producto", "rating_producto", "RatingProducto"])
    col_rl = _find_col(out, ["Rating_Logistica", "rating_logistica", "RatingLogistica"])
    col_nps = _find_col(out, ["Satisfaccion_NPS", "satisfaccion_nps", "NPS", "nps"])

    # Normalizar nombres mínimos
    if col_cat is None:
        out["Categoria"] = "N/A"
    else:
        out["Categoria"] = out[col_cat].astype(str).fillna("N/A").str.strip()

    # Convertir a numérico
    out["stock_actual_num"] = pd.to_numeric(out[col_stock], errors="coerce") if col_stock else np.nan
    out["rating_producto_num"] = pd.to_numeric(out[col_rp], errors="coerce") if col_rp else np.nan
    out["rating_logistica_num"] = pd.to_numeric(out[col_rl], errors="coerce") if col_rl else np.nan
    out["nps_num"] = pd.to_numeric(out[col_nps], errors="coerce") if col_nps else np.nan

    # Sentimiento negativo (puedes ajustar thresholds)
    # - rating producto <= 2 (malo)
    # - o NPS <= 6 (detractor)
    out["sentimiento_negativo"] = (
        (out["rating_producto_num"] <= 2) |
        (out["nps_num"] <= 6)
    ).astype(int)

    return out


def category_loyalty_kpis(df: pd.DataFrame, min_rows: int = 30) -> pd.DataFrame:
    """
    KPIs por categoría para detectar paradojas:
    - stock promedio alto
    - ratings / nps bajos
    - % sentimiento negativo alto
    """
    if df is None or df.empty:
        return pd.DataFrame()

    needed = ["Categoria", "stock_actual_num", "rating_producto_num", "rating_logistica_num", "nps_num", "sentimiento_negativo"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()

    base = df.copy()
    base = base.dropna(subset=["Categoria"])

    g = base.groupby("Categoria")

    out = g.agg(
        n=("Categoria", "size"),
        stock_prom=("stock_actual_num", "mean"),
        stock_mediana=("stock_actual_num", "median"),
        stock_total=("stock_actual_num", "sum"),
        rating_producto_prom=("rating_producto_num", "mean"),
        rating_logistica_prom=("rating_logistica_num", "mean"),
        nps_prom=("nps_num", "mean"),
        pct_sentimiento_negativo=("sentimiento_negativo", "mean"),
    ).reset_index()

    out = out[out["n"] >= min_rows].copy()

    # Formato bonito
    out["stock_prom"] = out["stock_prom"].round(2)
    out["stock_mediana"] = out["stock_mediana"].round(2)
    out["stock_total"] = out["stock_total"].round(2)
    out["rating_producto_prom"] = out["rating_producto_prom"].round(2)
    out["rating_logistica_prom"] = out["rating_logistica_prom"].round(2)
    out["nps_prom"] = out["nps_prom"].round(2)
    out["pct_sentimiento_negativo"] = (out["pct_sentimiento_negativo"] * 100).round(2)

    return out


def category_paradox_ranking(df_cat: pd.DataFrame) -> pd.DataFrame:
    """
    Ranking de "paradoja": mucho stock + mala percepción.
    Score alto = más urgente revisar (posible calidad o sobrecosto).
    """
    if df_cat is None or df_cat.empty:
        return pd.DataFrame()

    out = df_cat.copy()

    # Normalizar stock para score (min-max)
    if out["stock_prom"].nunique() > 1:
        out["stock_norm"] = (out["stock_prom"] - out["stock_prom"].min()) / (out["stock_prom"].max() - out["stock_prom"].min())
    else:
        out["stock_norm"] = 0.0

    # Convertir rating producto a "riesgo" (rating bajo = riesgo alto)
    # rating ideal 5, malo 1
    out["rating_riesgo"] = 1 - ((out["rating_producto_prom"] - 1) / 4)
    out["rating_riesgo"] = out["rating_riesgo"].clip(0, 1)

    # % sentimiento negativo también es riesgo directo
    out["neg_norm"] = (out["pct_sentimiento_negativo"] / 100).clip(0, 1)

    # Score final: ponderación
    out["score_paradoja"] = (
        out["stock_norm"] * 0.45 +
        out["rating_riesgo"] * 0.35 +
        out["neg_norm"] * 0.20
    ).round(3)

    out = out.sort_values(by="score_paradoja", ascending=False).reset_index(drop=True)

    # columnas útiles al frente
    cols_front = [
        "Categoria", "n", "stock_prom", "stock_total",
        "rating_producto_prom", "nps_prom",
        "pct_sentimiento_negativo", "score_paradoja"
    ]
    cols_front = [c for c in cols_front if c in out.columns]

    return out[cols_front]

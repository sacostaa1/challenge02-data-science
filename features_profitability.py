# features_profitability.py
import pandas as pd
import numpy as np


def _find_col(df: pd.DataFrame, candidates: list[str]):
    """Retorna el primer nombre de columna que exista en df, o None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def add_profitability_features(df_master: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Agrega features mínimas para responder:
    1) SKUs con margen negativo y si es crítico por canal.

    Columnas creadas:
    - ingreso_total_usd
    - costo_total_usd
    - margen_usd
    - margen_negativo (bool)

    Retorna:
    - df_enriched
    - meta (dict con nombres de columnas detectadas y warnings)
    """
    df = df_master.copy()
    meta = {"warnings": []}

    # -------------------------
    # Detectar columnas base
    # -------------------------
    col_qty = _find_col(df, ["Cantidad", "Cantidad_Vendida", "cantidad_vendida", "qty", "unidades"])
    col_price = _find_col(df, ["Precio_Venta_Final", "precio_venta_final", "Precio_Unitario", "precio_unitario", "Precio_USD"])
    col_cost = _find_col(df, ["Costo_Unitario_USD", "costo_unitario_usd", "Costo_Unitario", "costo_unitario"])
    col_ship = _find_col(df, ["Costo_Envio", "costo_envio", "Shipping_Cost", "shipping_cost"])
    col_channel = _find_col(df, ["Canal_Venta", "canal_venta", "Canal", "canal"])
    col_sku = _find_col(df, ["SKU", "sku", "Sku", "SKU_ID"])

    meta["col_qty"] = col_qty
    meta["col_price"] = col_price
    meta["col_cost"] = col_cost
    meta["col_ship"] = col_ship
    meta["col_channel"] = col_channel
    meta["col_sku"] = col_sku

    # Validación mínima
    required = {"Cantidad": col_qty, "Precio_Venta": col_price, "Costo_Unitario": col_cost}
    missing = [k for k, v in required.items() if v is None]

    if missing:
        meta["warnings"].append(
            f"No se pueden calcular márgenes: faltan columnas requeridas: {missing}"
        )
        # Retornamos df sin cambios (no rompemos el main)
        return df, meta

    # -------------------------
    # Convertir a numérico
    # -------------------------
    qty = _to_num(df[col_qty]).fillna(0)
    price = _to_num(df[col_price]).fillna(0)
    cost_unit = _to_num(df[col_cost]).fillna(np.nan)

    # shipping puede no existir
    if col_ship is not None:
        ship = _to_num(df[col_ship]).fillna(0)
    else:
        ship = pd.Series(0, index=df.index)

    # -------------------------
    # Features VALIOSAS (mínimas)
    # -------------------------
    df["ingreso_total_usd"] = (qty * price).round(2)

    # costo_total_usd = (qty * costo_unitario) + shipping
    # Si costo_unitario es NaN, el costo total queda NaN -> margen NaN
    df["costo_total_usd"] = ((qty * cost_unit) + ship).round(2)

    df["margen_usd"] = (df["ingreso_total_usd"] - df["costo_total_usd"]).round(2)

    # margen negativo solo si margen es numérico válido
    df["margen_negativo"] = df["margen_usd"].apply(lambda x: bool(x < 0) if pd.notna(x) else False)

    # -------------------------
    # Warnings útiles
    # -------------------------
    if df["margen_usd"].isna().mean() > 0.05:
        meta["warnings"].append(
            "Hay más de 5% de márgenes NaN. Posible falta de costos unitarios en inventario o SKUs fantasma."
        )

    # Si no hay canal o sku, avisamos pero no bloqueamos
    if col_channel is None:
        meta["warnings"].append("No se detectó columna de Canal_Venta. El análisis por canal será limitado.")
    if col_sku is None:
        meta["warnings"].append("No se detectó columna SKU. El análisis por SKU será limitado.")

    return df, meta


def profitability_summary(df: pd.DataFrame) -> dict:
    """
    KPIs rápidos para la pregunta 1.
    Espera columnas ya creadas:
    - ingreso_total_usd
    - margen_usd
    - margen_negativo
    """
    if "ingreso_total_usd" not in df.columns or "margen_usd" not in df.columns:
        return {}

    total_ingreso = float(pd.to_numeric(df["ingreso_total_usd"], errors="coerce").fillna(0).sum())
    total_margen = float(pd.to_numeric(df["margen_usd"], errors="coerce").fillna(0).sum())
    neg_count = int(df.get("margen_negativo", pd.Series(False, index=df.index)).sum())

    # Impacto de margen negativo (sumatoria de márgenes negativos)
    margen_neg_total = float(
        pd.to_numeric(df.loc[df.get("margen_negativo", False), "margen_usd"], errors="coerce")
        .fillna(0)
        .sum()
    )

    return {
        "total_ingreso_usd": round(total_ingreso, 2),
        "total_margen_usd": round(total_margen, 2),
        "transacciones_margen_negativo": neg_count,
        "impacto_margen_negativo_usd": round(margen_neg_total, 2),
    }

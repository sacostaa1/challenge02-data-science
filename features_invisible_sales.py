import pandas as pd
import numpy as np


# ======================================================
# Helpers
# ======================================================
def _find_col(df: pd.DataFrame, candidates: list[str]):
    """Retorna el primer nombre de columna que exista en df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_numeric_safe(s: pd.Series):
    return pd.to_numeric(s, errors="coerce")


# ======================================================
# 1) Feature Engineering para Venta Invisible
# ======================================================
def add_invisible_sales_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features necesarias para responder:
    "¿Cuánto dinero proviene de ventas con SKUs que no están en inventario?"

    Columnas esperadas en el master:
    - Cantidad_Vendida
    - Precio_Venta_Final
    - sku_en_inventario (boolean)
    """

    if df is None or df.empty:
        return df

    out = df.copy()

    # Detectar columnas clave
    col_qty = _find_col(out, ["Cantidad_Vendida", "cantidad_vendida", "Qty", "qty", "cantidad"])
    col_price = _find_col(out, ["Precio_Venta_Final", "precio_venta_final", "Precio", "precio", "precio_final"])
    col_flag = _find_col(out, ["sku_en_inventario", "SKU_en_inventario", "sku_in_inventory"])

    # Normalizar numéricos
    if col_qty is not None:
        out["cantidad"] = _to_numeric_safe(out[col_qty])
    else:
        out["cantidad"] = np.nan

    if col_price is not None:
        out["precio_final"] = _to_numeric_safe(out[col_price])
    else:
        out["precio_final"] = np.nan

    # Ingreso bruto por transacción
    out["ingreso_usd"] = (out["cantidad"].fillna(0) * out["precio_final"].fillna(0)).astype(float)

    # Flag de SKU en inventario
    # Si no existe, asumimos que NO hay forma de validar y lo dejamos como NaN
    if col_flag is not None:
        # Puede venir como True/False o "TRUE"/"FALSE"
        if out[col_flag].dtype == object:
            out["sku_en_inventario_flag"] = (
                out[col_flag].astype(str).str.strip().str.lower().map({"true": True, "false": False})
            )
        else:
            out["sku_en_inventario_flag"] = out[col_flag].astype("boolean")
    else:
        out["sku_en_inventario_flag"] = pd.Series([pd.NA] * len(out), dtype="boolean")

    # Venta invisible: sku no está en inventario
    out["venta_invisible"] = (out["sku_en_inventario_flag"] == False).astype(int)

    # Ingreso en riesgo: solo donde venta_invisible=1
    out["ingreso_en_riesgo_usd"] = np.where(out["venta_invisible"] == 1, out["ingreso_usd"], 0.0)

    return out


# ======================================================
# 2) KPIs globales de Venta Invisible
# ======================================================
def invisible_sales_summary(df: pd.DataFrame) -> dict:
    """
    Retorna un resumen con métricas globales:
    - ingreso_total
    - ingreso_en_riesgo
    - pct_ingreso_en_riesgo
    - n_transacciones_total
    - n_transacciones_invisibles
    - pct_transacciones_invisibles
    """

    if df is None or df.empty:
        return {}

    needed = ["ingreso_usd", "ingreso_en_riesgo_usd", "venta_invisible"]
    for c in needed:
        if c not in df.columns:
            return {}

    base = df.copy()

    ingreso_total = float(base["ingreso_usd"].fillna(0).sum())
    ingreso_riesgo = float(base["ingreso_en_riesgo_usd"].fillna(0).sum())

    n_total = int(len(base))
    n_invisible = int(base["venta_invisible"].fillna(0).sum())

    pct_ingreso_riesgo = float((ingreso_riesgo / ingreso_total) * 100) if ingreso_total > 0 else 0.0
    pct_trans_invisible = float((n_invisible / n_total) * 100) if n_total > 0 else 0.0

    return {
        "ingreso_total_usd": round(ingreso_total, 2),
        "ingreso_en_riesgo_usd": round(ingreso_riesgo, 2),
        "pct_ingreso_en_riesgo": round(pct_ingreso_riesgo, 2),
        "n_transacciones_total": n_total,
        "n_transacciones_invisibles": n_invisible,
        "pct_transacciones_invisibles": round(pct_trans_invisible, 2),
    }


# ======================================================
# 3) Ranking por SKU "fantasma"
# ======================================================
def invisible_sales_by_sku(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Agrupa ventas invisibles por SKU_ID.
    Devuelve los SKUs con mayor ingreso en riesgo.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    col_sku = _find_col(df, ["SKU_ID", "SKU", "sku", "Sku"])
    if col_sku is None:
        return pd.DataFrame()

    needed = ["venta_invisible", "ingreso_en_riesgo_usd", "ingreso_usd"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()

    base = df.copy()
    base = base[base["venta_invisible"] == 1].copy()

    if base.empty:
        return pd.DataFrame()

    out = base.groupby(col_sku).agg(
        n_transacciones=(col_sku, "size"),
        ingreso_en_riesgo_usd=("ingreso_en_riesgo_usd", "sum"),
        ingreso_total_usd=("ingreso_usd", "sum"),
    ).reset_index()

    out = out.sort_values(by="ingreso_en_riesgo_usd", ascending=False).head(top_n).reset_index(drop=True)
    out = out.rename(columns={col_sku: "SKU_ID"})

    # Formato bonito
    out["ingreso_en_riesgo_usd"] = out["ingreso_en_riesgo_usd"].round(2)
    out["ingreso_total_usd"] = out["ingreso_total_usd"].round(2)

    return out


# ======================================================
# 4) Ranking por Canal y Ciudad
# ======================================================
def invisible_sales_by_channel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ventas invisibles por Canal_Venta:
    - n
    - ingreso_en_riesgo_usd
    - pct_ingreso_riesgo_del_canal
    """

    if df is None or df.empty:
        return pd.DataFrame()

    col_channel = _find_col(df, ["Canal_Venta", "canal_venta", "canal", "Channel"])
    if col_channel is None:
        return pd.DataFrame()

    needed = ["ingreso_usd", "ingreso_en_riesgo_usd"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()

    base = df.copy()

    out = base.groupby(col_channel).agg(
        n_transacciones=(col_channel, "size"),
        ingreso_total_usd=("ingreso_usd", "sum"),
        ingreso_en_riesgo_usd=("ingreso_en_riesgo_usd", "sum"),
    ).reset_index()

    out["pct_ingreso_riesgo"] = np.where(
        out["ingreso_total_usd"] > 0,
        (out["ingreso_en_riesgo_usd"] / out["ingreso_total_usd"]) * 100,
        0.0
    )

    out["ingreso_total_usd"] = out["ingreso_total_usd"].round(2)
    out["ingreso_en_riesgo_usd"] = out["ingreso_en_riesgo_usd"].round(2)
    out["pct_ingreso_riesgo"] = out["pct_ingreso_riesgo"].round(2)

    out = out.sort_values(by="ingreso_en_riesgo_usd", ascending=False).reset_index(drop=True)
    out = out.rename(columns={col_channel: "Canal_Venta"})
    return out


def invisible_sales_by_city(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Ventas invisibles por Ciudad_Destino.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    col_city = _find_col(df, ["Ciudad_Destino", "ciudad_destino", "Ciudad", "ciudad"])
    if col_city is None:
        return pd.DataFrame()

    needed = ["ingreso_usd", "ingreso_en_riesgo_usd"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()

    base = df.copy()

    out = base.groupby(col_city).agg(
        n_transacciones=(col_city, "size"),
        ingreso_total_usd=("ingreso_usd", "sum"),
        ingreso_en_riesgo_usd=("ingreso_en_riesgo_usd", "sum"),
    ).reset_index()

    out["pct_ingreso_riesgo"] = np.where(
        out["ingreso_total_usd"] > 0,
        (out["ingreso_en_riesgo_usd"] / out["ingreso_total_usd"]) * 100,
        0.0
    )

    out["ingreso_total_usd"] = out["ingreso_total_usd"].round(2)
    out["ingreso_en_riesgo_usd"] = out["ingreso_en_riesgo_usd"].round(2)
    out["pct_ingreso_riesgo"] = out["pct_ingreso_riesgo"].round(2)

    out = out.sort_values(by="ingreso_en_riesgo_usd", ascending=False).head(top_n).reset_index(drop=True)
    out = out.rename(columns={col_city: "Ciudad_Destino"})
    return out


# ======================================================
# 5) Dataset filtrado: solo ventas invisibles (para inspección)
# ======================================================
def get_invisible_transactions(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    Retorna transacciones invisibles ordenadas por ingreso_usd descendente.
    Útil para auditoría rápida.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    needed = ["venta_invisible", "ingreso_usd", "ingreso_en_riesgo_usd"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()

    base = df[df["venta_invisible"] == 1].copy()
    if base.empty:
        return pd.DataFrame()

    # Ordenar por mayor ingreso
    base = base.sort_values(by="ingreso_usd", ascending=False).head(top_n).reset_index(drop=True)
    return base

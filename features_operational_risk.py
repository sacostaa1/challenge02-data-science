import pandas as pd
import numpy as np


# -----------------------------
# Helpers
# -----------------------------
def _find_col(df: pd.DataFrame, candidates: list[str]):
    """Retorna el primer nombre de columna que exista en df."""
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_datetime_mixed(series: pd.Series) -> pd.Series:
    """
    Convierte fechas con formatos mixtos:
    - dd/mm/yyyy (día primero)
    - yyyy-mm-dd
    - otros parseables
    """
    if series is None:
        return pd.Series(dtype="datetime64[ns]")

    s = series.astype(str).str.strip()

    # intento 1: dayfirst=True (sirve para 29/07/2025)
    dt1 = pd.to_datetime(s, errors="coerce", dayfirst=True)

    # intento 2: normal (sirve para 2025-11-16)
    dt2 = pd.to_datetime(s, errors="coerce", dayfirst=False)

    # combinar: si dt1 es NaT, usar dt2
    return dt1.fillna(dt2)


def _normalize_yes_no(series: pd.Series) -> pd.Series:
    """
    Normaliza valores tipo SI/NO, True/False, 1/0.
    Retorna 1 si "sí", 0 si "no", NaN si no se puede.
    """
    if series is None:
        return pd.Series(dtype="float")

    s = series.astype(str).str.strip().str.upper()

    yes_vals = {"SI", "SÍ", "YES", "TRUE", "1"}
    no_vals = {"NO", "FALSE", "0"}

    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.isin(yes_vals)] = 1.0
    out[s.isin(no_vals)] = 0.0
    return out


# ============================================================
# 1) Features Operativo-Riesgo
# ============================================================
def add_operational_risk_features(
    df: pd.DataFrame,
    reference_date: str | None = None,
    stale_days_threshold: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """
    Agrega features para Storytelling de Riesgo Operativo (P5):
    - dias_desde_revision
    - revision_desactualizada (1 si dias_desde_revision >= threshold)
    - ticket_soporte_bin (1 si Ticket_Soporte_Abierto == SI)
    - nps (numérico)
    - nps_bajo (1 si nps <= 6)
    - bodega_origen_clean

    Importante:
    - Usa Fecha_Venta como referencia por fila.
    - Si Ultima_Revision > Fecha_Venta (dato inconsistente), se corrige con clip(0)
      en vez de convertir a NaN para no perder filas.
    """
    meta = {"warnings": []}

    if df is None or df.empty:
        meta["warnings"].append("Dataset vacío: no se agregaron features.")
        return df, meta

    out = df.copy()

    # Columnas esperadas según tu CSV general
    col_warehouse = _find_col(out, ["Bodega_Origen"])
    col_last_review = _find_col(out, ["Ultima_Revision"])
    col_ticket = _find_col(out, ["Ticket_Soporte_Abierto"])
    col_nps = _find_col(out, ["Satisfaccion_NPS"])
    col_fecha_venta = _find_col(out, ["Fecha_Venta"])

    # Validaciones
    if col_warehouse is None:
        meta["warnings"].append("No se encontró columna Bodega_Origen.")
    if col_last_review is None:
        meta["warnings"].append("No se encontró columna Ultima_Revision.")
    if col_ticket is None:
        meta["warnings"].append("No se encontró columna Ticket_Soporte_Abierto.")
    if col_nps is None:
        meta["warnings"].append("No se encontró columna Satisfaccion_NPS.")
    if col_fecha_venta is None:
        meta["warnings"].append("No se encontró columna Fecha_Venta (se usará reference_date o hoy).")

    # -----------------------------
    # 1) dias_desde_revision
    # -----------------------------
    if col_last_review is not None:
        last_dt = _to_datetime_mixed(out[col_last_review])

        # Caso ideal: usar Fecha_Venta por fila
        if col_fecha_venta is not None:
            sale_dt = _to_datetime_mixed(out[col_fecha_venta])

            out["dias_desde_revision"] = (sale_dt - last_dt).dt.days

            # Corrección: si sale negativo por inconsistencias, lo dejamos en 0
            out["dias_desde_revision"] = pd.to_numeric(out["dias_desde_revision"], errors="coerce")
            out["dias_desde_revision"] = out["dias_desde_revision"].clip(lower=0)

        else:
            # fallback: usar reference_date o "hoy"
            if reference_date is None:
                ref_dt = pd.Timestamp.today().normalize()
            else:
                ref_dt = pd.to_datetime(reference_date, errors="coerce")
                if pd.isna(ref_dt):
                    ref_dt = pd.Timestamp.today().normalize()
                    meta["warnings"].append("reference_date inválida, usando hoy().")

            out["dias_desde_revision"] = (ref_dt - last_dt).dt.days
            out["dias_desde_revision"] = pd.to_numeric(out["dias_desde_revision"], errors="coerce")
            out["dias_desde_revision"] = out["dias_desde_revision"].clip(lower=0)

    else:
        out["dias_desde_revision"] = np.nan

    # -----------------------------
    # 2) revision_desactualizada
    # -----------------------------
    out["revision_desactualizada"] = (
        pd.to_numeric(out["dias_desde_revision"], errors="coerce") >= stale_days_threshold
    ).astype(int)

    # -----------------------------
    # 3) ticket_soporte_bin
    # -----------------------------
    if col_ticket is not None:
        out["ticket_soporte_bin"] = _normalize_yes_no(out[col_ticket])
    else:
        out["ticket_soporte_bin"] = np.nan

    # -----------------------------
    # 4) NPS numérico + nps_bajo
    # -----------------------------
    if col_nps is not None:
        out["nps"] = pd.to_numeric(out[col_nps], errors="coerce")
    else:
        out["nps"] = np.nan

    out["nps_bajo"] = (out["nps"] <= 6).astype(int)

    # -----------------------------
    # 5) Warehouse clean
    # -----------------------------
    if col_warehouse is not None:
        out["bodega_origen_clean"] = out[col_warehouse].astype(str).str.strip().replace({"": "N/A"})
    else:
        out["bodega_origen_clean"] = "N/A"

    return out, meta


# ============================================================
# 2) KPIs por Bodega
# ============================================================
def operational_risk_by_warehouse(
    df: pd.DataFrame,
    min_rows: int = 30,
    blind_percentile: int = 80,
) -> pd.DataFrame:
    """
    Agrega KPIs por bodega:
    - avg_dias_desde_revision
    - pct_revision_desactualizada
    - pct_tickets
    - avg_nps
    - pct_nps_bajo
    - score_riesgo_operativo
    - operando_a_ciegas (flag relativo por percentil)

    Score (más alto = más riesgo):
      0.35 * avg_dias_desde_revision_norm
    + 0.25 * pct_revision_desactualizada
    + 0.25 * pct_tickets
    + 0.15 * pct_nps_bajo
    """
    if df is None or df.empty:
        return pd.DataFrame()

    needed = [
        "bodega_origen_clean",
        "dias_desde_revision",
        "revision_desactualizada",
        "ticket_soporte_bin",
        "nps",
        "nps_bajo",
    ]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()

    base = df[needed].copy()

    # Numeric safe
    base["dias_desde_revision"] = pd.to_numeric(base["dias_desde_revision"], errors="coerce")
    base["revision_desactualizada"] = pd.to_numeric(base["revision_desactualizada"], errors="coerce")
    base["ticket_soporte_bin"] = pd.to_numeric(base["ticket_soporte_bin"], errors="coerce")
    base["nps"] = pd.to_numeric(base["nps"], errors="coerce")
    base["nps_bajo"] = pd.to_numeric(base["nps_bajo"], errors="coerce")

    g = base.groupby("bodega_origen_clean")

    out = g.agg(
        n_valid=("dias_desde_revision", "count"),
        avg_dias_desde_revision=("dias_desde_revision", "mean"),
        pct_revision_desactualizada=("revision_desactualizada", "mean"),
        pct_tickets=("ticket_soporte_bin", "mean"),
        avg_nps=("nps", "mean"),
        pct_nps_bajo=("nps_bajo", "mean"),
    ).reset_index()

    out = out[out["n_valid"] >= min_rows].copy()
    if out.empty:
        return out

    # Convertir a % bonitos
    out["pct_revision_desactualizada"] = (out["pct_revision_desactualizada"] * 100).round(2)
    out["pct_tickets"] = (out["pct_tickets"] * 100).round(2)
    out["pct_nps_bajo"] = (out["pct_nps_bajo"] * 100).round(2)

    # Normalización simple para días (0..1)
    d = out["avg_dias_desde_revision"].copy()
    d_min, d_max = float(d.min()), float(d.max())
    if d_max > d_min:
        out["avg_dias_desde_revision_norm"] = ((d - d_min) / (d_max - d_min)).round(4)
    else:
        out["avg_dias_desde_revision_norm"] = 0.0

    # Score completo (más alto = más riesgo)
    out["score_riesgo_operativo"] = (
        out["avg_dias_desde_revision_norm"].fillna(0) * 0.35
        + (out["pct_revision_desactualizada"].fillna(0) / 100) * 0.25
        + (out["pct_tickets"].fillna(0) / 100) * 0.25
        + (out["pct_nps_bajo"].fillna(0) / 100) * 0.15
    ).round(4)

    # Operando a ciegas: percentil sobre avg_dias_desde_revision
    blind_percentile = int(blind_percentile)
    blind_percentile = max(0, min(100, blind_percentile))

    threshold = np.nanpercentile(out["avg_dias_desde_revision"], blind_percentile)
    out["threshold_blind_days"] = float(threshold)
    out["operando_a_ciegas"] = (out["avg_dias_desde_revision"] >= threshold).astype(int)

    out = out.sort_values("score_riesgo_operativo", ascending=False).reset_index(drop=True)
    return out


# ============================================================
# 3) Dataset para Scatter
# ============================================================
def operational_risk_scatter_df(
    df: pd.DataFrame,
    min_rows: int = 30,
    blind_percentile: int = 80,
) -> pd.DataFrame:
    """
    Devuelve tabla por bodega lista para scatter:
    x = avg_dias_desde_revision
    y = pct_tickets
    size/color = avg_nps
    """
    agg = operational_risk_by_warehouse(df, min_rows=min_rows, blind_percentile=blind_percentile)
    if agg.empty:
        return agg

    agg["pct_tickets_frac"] = (agg["pct_tickets"] / 100.0).round(4)
    agg["pct_nps_bajo_frac"] = (agg["pct_nps_bajo"] / 100.0).round(4)
    return agg

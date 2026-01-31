import pandas as pd
import numpy as np
import re
import unicodedata


def remove_accents(text: str) -> str:
    """
    Elimina tildes/acentos de un texto.
    Ej: 'Medellín' -> 'Medellin'
    """
    if text is None:
        return text
    text = str(text)
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def normalize_ciudad_value(x):
    """
    Normaliza nombres de ciudades:
    - Convierte abreviaturas (BOG, MED) a nombre completo
    - Quita tildes
    - Capitaliza (Mayúscula inicial)
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if s.lower() in ["nan", "none", "null", ""]:
        return np.nan

    # Quitar tildes
    s = remove_accents(s)

    # Limpiar caracteres raros (mantener letras/espacios)
    s = re.sub(r"[^a-zA-Z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Normalizar a minúsculas para mapear
    s_lower = s.lower()

    # Mapeo de abreviaturas comunes
    mapping = {
        "bog": "bogota",
        "bta": "bogota",
        "bogota d c": "bogota",
        "bogota dc": "bogota",
        "med": "medellin",
        "mde": "medellin",
        "cali": "cali",
        "ctg": "cartagena",
        "baq": "barranquilla",
    }

    if s_lower in mapping:
        s_lower = mapping[s_lower]

    # Capitalizar: "bogota" -> "Bogota"
    return s_lower.capitalize()


def clean_transacciones_logistica(df: pd.DataFrame):
    """
    Limpieza específica para dataset de transacciones logísticas.
    Devuelve (df_clean, decisiones_df).
    """
    decisiones = []
    df_clean = df.copy()

    before_rows = len(df_clean)

    # =========================================================
    # 1) cantidad_vendida: eliminar registros con valor -5
    # =========================================================
    cantidad_cols = [c for c in df_clean.columns if c.lower() in ["cantidad_vendida", "cantidad", "qty_vendida"]]
    if cantidad_cols:
        cantidad_col = cantidad_cols[0]
        cantidad_num = pd.to_numeric(df_clean[cantidad_col], errors="coerce")

        mask_neg_5 = (cantidad_num == -5)
        removed = int(mask_neg_5.sum())

        df_clean = df_clean.loc[~mask_neg_5].copy()

        decisiones.append({
            "Acción": "Eliminación de registros por cantidad negativa",
            "Columna": cantidad_col,
            "Registros afectados": removed,
            "Justificación": (
                "Se eliminaron registros donde cantidad_vendida = -5, "
                "ya que representa un valor inválido para ventas y puede distorsionar análisis."
            )
        })

        # Reasignar limpio como numérico (opcional, pero recomendado)
        df_clean[cantidad_col] = pd.to_numeric(df_clean[cantidad_col], errors="coerce")

    # =========================================================
    # 2) costo_envio: nulos -> 0
    # =========================================================
    costo_envio_cols = [c for c in df_clean.columns if c.lower() in ["costo_envio", "costo_envio_usd", "shipping_cost"]]
    if costo_envio_cols:
        costo_envio_col = costo_envio_cols[0]
        costo_num = pd.to_numeric(df_clean[costo_envio_col], errors="coerce")

        nulls_before = int(costo_num.isna().sum())

        df_clean[costo_envio_col] = costo_num.fillna(0)

        decisiones.append({
            "Acción": "Imputación de costo_envio nulo",
            "Columna": costo_envio_col,
            "Registros afectados": nulls_before,
            "Justificación": (
                "Se imputaron valores nulos de costo_envio con 0, interpretando "
                "que no hubo costo registrado o el envío fue gratuito."
            )
        })

    # =========================================================
    # 3) estado_envio: nulos -> moda
    # =========================================================
    estado_cols = [c for c in df_clean.columns if c.lower() in ["estado_envio", "estado", "shipping_status"]]
    if estado_cols:
        estado_col = estado_cols[0]

        nulls_before = int(df_clean[estado_col].isna().sum())

        # moda (ignorando nulos)
        moda_series = df_clean[estado_col].mode(dropna=True)
        moda_value = moda_series.iloc[0] if not moda_series.empty else "desconocido"

        df_clean[estado_col] = df_clean[estado_col].fillna(moda_value)

        decisiones.append({
            "Acción": "Imputación de estado_envio nulo con moda",
            "Columna": estado_col,
            "Registros afectados": nulls_before,
            "Justificación": (
                "Se imputaron valores nulos en estado_envio usando la moda "
                f"('{moda_value}'), ya que es una variable categórica y la moda "
                "preserva la distribución más frecuente."
            )
        })

    # =========================================================
    # 4) ciudad_destino: normalización (abreviaturas, mayúscula inicial, sin tilde)
    # =========================================================
    ciudad_cols = [c for c in df_clean.columns if c.lower() in ["ciudad_destino", "ciudad", "destino_ciudad"]]
    if ciudad_cols:
        ciudad_col = ciudad_cols[0]
        before_unique = df_clean[ciudad_col].nunique(dropna=True)

        df_clean[ciudad_col] = df_clean[ciudad_col].apply(normalize_ciudad_value)

        after_unique = df_clean[ciudad_col].nunique(dropna=True)

        decisiones.append({
            "Acción": "Normalización de ciudad_destino",
            "Columna": ciudad_col,
            "Registros afectados": int(len(df_clean)),
            "Justificación": (
                "Se normalizaron nombres de ciudades para evitar duplicados por variaciones "
                "de escritura (abreviaturas como BOG/MED, tildes, mayúsculas/minúsculas). "
                f"Únicos antes: {before_unique}, únicos después: {after_unique}."
            )
        })

    after_rows = len(df_clean)
    removed_rows_total = before_rows - after_rows

    if removed_rows_total > 0:
        decisiones.append({
            "Acción": "Resumen de eliminación de filas",
            "Columna": "(dataset)",
            "Registros afectados": removed_rows_total,
            "Justificación": (
                "Se eliminaron filas únicamente por reglas explícitas (cantidad_vendida = -5)."
            )
        })

    return df_clean, pd.DataFrame(decisiones)

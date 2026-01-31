import pandas as pd
import numpy as np


def normalize_ticket_value(x):
    """
    Normaliza ticket_soporte_abierto a 'SI' / 'NO'
    - 1 -> SI
    - 0 -> NO
    - Sí/Si -> SI
    - No -> NO
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower()

    if s in ["1", "si", "sí", "s", "true", "t", "yes", "y"]:
        return "SI"
    if s in ["0", "no", "n", "false", "f"]:
        return "NO"

    # Si viene algo raro, lo dejamos como NaN para que se note en métricas
    return np.nan


def clean_feedback_clientes(df: pd.DataFrame):
    """
    Limpieza específica para dataset feedback_clientes.
    Devuelve (df_clean, decisiones_df).
    """
    decisiones = []
    df_clean = df.copy()

    before_rows = len(df_clean)

    # =========================================================
    # 1) rating_producto: >5 -> 5 (incluye el 99)
    # =========================================================
    rating_cols = [c for c in df_clean.columns if c.lower() in ["rating_producto", "rating", "calificacion_producto"]]
    if rating_cols:
        rating_col = rating_cols[0]
        rating_num = pd.to_numeric(df_clean[rating_col], errors="coerce")

        affected = int((rating_num > 5).sum())
        df_clean.loc[rating_num > 5, rating_col] = 5

        decisiones.append({
            "Acción": "Cap de rating (valores fuera de rango)",
            "Columna": rating_col,
            "Registros afectados": affected,
            "Justificación": (
                "Se normalizó rating_producto para que no supere 5. "
                "Esto corrige valores inválidos (ej: 99) manteniendo el máximo permitido."
            )
        })

        df_clean[rating_col] = pd.to_numeric(df_clean[rating_col], errors="coerce")

    # =========================================================
    # 2) comentario_texto: N/A y --- -> N/A
    # =========================================================
    comment_cols = [c for c in df_clean.columns if c.lower() in ["comentario_texto", "comentario", "feedback_texto"]]
    if comment_cols:
        comment_col = comment_cols[0]

        before_na = int(df_clean[comment_col].isna().sum())

        # normalización: valores tipo N/A, --- y similares
        def normalize_comment(x):
            if pd.isna(x):
                return "N/A"
            s = str(x).strip()
            if s.lower() in ["n/a", "na", "---", "--", "-", "sin comentario", "none", "null", "nan", ""]:
                return "N/A"
            return s

        df_clean[comment_col] = df_clean[comment_col].apply(normalize_comment)

        affected = int((df_clean[comment_col] == "N/A").sum())

        decisiones.append({
            "Acción": "Normalización de comentarios vacíos",
            "Columna": comment_col,
            "Registros afectados": affected,
            "Justificación": (
                "Se unificaron registros sin comentario ('N/A', '---', vacíos o nulos) "
                "como 'N/A' para indicar ausencia de texto y evitar ruido en NLP."
            )
        })

        # (opcional) info adicional de nulos antes
        if before_na > 0:
            decisiones.append({
                "Acción": "Imputación de comentario nulo",
                "Columna": comment_col,
                "Registros afectados": before_na,
                "Justificación": "Los comentarios nulos se interpretan como ausencia de comentario y se imputan como 'N/A'."
            })

    # =========================================================
    # 3) ticket_soporte_abierto: normalizar a SI/NO
    # =========================================================
    ticket_cols = [c for c in df_clean.columns if c.lower() in ["ticket_soporte_abierto", "ticket_abierto", "soporte_abierto"]]
    if ticket_cols:
        ticket_col = ticket_cols[0]

        nulls_before = int(df_clean[ticket_col].isna().sum())

        df_clean[ticket_col] = df_clean[ticket_col].apply(normalize_ticket_value)

        nulls_after = int(df_clean[ticket_col].isna().sum())
        affected = int(len(df_clean) - nulls_after)

        decisiones.append({
            "Acción": "Normalización booleana ticket_soporte_abierto",
            "Columna": ticket_col,
            "Registros afectados": affected,
            "Justificación": (
                "Se normalizó el campo a valores consistentes 'SI'/'NO'. "
                "Se mapearon 1->SI y 0->NO, además de variantes tipo Sí/No."
            )
        })

        if nulls_after > 0 and nulls_after != nulls_before:
            decisiones.append({
                "Acción": "Valores no reconocidos en ticket_soporte_abierto",
                "Columna": ticket_col,
                "Registros afectados": nulls_after,
                "Justificación": (
                    "Algunos registros no pudieron mapearse a SI/NO y quedaron como nulos para revisión."
                )
            })

    # =========================================================
    # 4) edad: eliminar registro atípico 195
    # =========================================================
    edad_cols = [c for c in df_clean.columns if c.lower() in ["edad", "age"]]
    if edad_cols:
        edad_col = edad_cols[0]
        edad_num = pd.to_numeric(df_clean[edad_col], errors="coerce")

        mask_195 = (edad_num == 195)
        removed = int(mask_195.sum())

        df_clean = df_clean.loc[~mask_195].copy()

        decisiones.append({
            "Acción": "Eliminación de registro atípico de edad",
            "Columna": edad_col,
            "Registros afectados": removed,
            "Justificación": (
                "Se eliminó el registro con edad=195 por ser un valor imposible/no confiable "
                "y potencialmente error de digitación."
            )
        })

        df_clean[edad_col] = pd.to_numeric(df_clean[edad_col], errors="coerce")

    # =========================================================
    # Resumen eliminación
    # =========================================================
    after_rows = len(df_clean)
    removed_rows_total = before_rows - after_rows

    if removed_rows_total > 0:
        decisiones.append({
            "Acción": "Resumen de eliminación de filas",
            "Columna": "(dataset)",
            "Registros afectados": removed_rows_total,
            "Justificación": "Se eliminaron filas únicamente por reglas explícitas (edad=195)."
        })

    return df_clean, pd.DataFrame(decisiones)

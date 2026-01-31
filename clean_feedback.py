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
    # 1) rating_producto: >5 -> 5 (incluye 99)
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
    #    ⚠️ IMPORTANTE: NO tocamos recomienda_marca
    # =========================================================
    comment_cols = [c for c in df_clean.columns if c.lower() in ["comentario_texto", "comentario", "feedback_texto"]]
    if comment_cols:
        comment_col = comment_cols[0]

        nulls_before = int(df_clean[comment_col].isna().sum())

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

        if nulls_before > 0:
            decisiones.append({
                "Acción": "Imputación de comentario nulo",
                "Columna": comment_col,
                "Registros afectados": nulls_before,
                "Justificación": "Los comentarios nulos se interpretan como ausencia de comentario y se imputan como 'N/A'."
            })

    # =========================================================
    # 3) ticket_soporte_abierto: normalizar a SI/NO
    # =========================================================
    ticket_cols = [c for c in df_clean.columns if c.lower() in ["ticket_soporte_abierto", "ticket_abierto", "soporte_abierto"]]
    if ticket_cols:
        ticket_col = ticket_cols[0]

        df_clean[ticket_col] = df_clean[ticket_col].apply(normalize_ticket_value)

        affected = int(df_clean[ticket_col].notna().sum())

        decisiones.append({
            "Acción": "Normalización booleana ticket_soporte_abierto",
            "Columna": ticket_col,
            "Registros afectados": affected,
            "Justificación": (
                "Se normalizó el campo a valores consistentes 'SI'/'NO'. "
                "Se mapearon 1->SI y 0->NO, además de variantes tipo Sí/No."
            )
        })

    # =========================================================
    # 4) edad: eliminar outliers por IQR (en vez de solo 195)
    # =========================================================
    edad_cols = [c for c in df_clean.columns if c.lower() in ["edad", "age"]]
    if edad_cols:
        edad_col = edad_cols[0]
        edad_num = pd.to_numeric(df_clean[edad_col], errors="coerce")

        s = edad_num.dropna()

        removed = 0
        if not s.empty:
            Q1 = s.quantile(0.25)
            Q3 = s.quantile(0.75)
            IQR = Q3 - Q1

            if IQR > 0 and not pd.isna(IQR):
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                mask_out = (edad_num < lower) | (edad_num > upper)
                removed = int(mask_out.sum())

                df_clean = df_clean.loc[~mask_out].copy()

                decisiones.append({
                    "Acción": "Eliminación de outliers de edad (IQR)",
                    "Columna": edad_col,
                    "Registros afectados": removed,
                    "Justificación": (
                        "Se eliminaron registros con edad atípica usando regla IQR "
                        "para evitar valores imposibles o errores de captura. "
                        f"Umbrales: [{lower:.2f}, {upper:.2f}]."
                    )
                })
            else:
                decisiones.append({
                    "Acción": "Revisión de outliers edad (sin acción)",
                    "Columna": edad_col,
                    "Registros afectados": 0,
                    "Justificación": "No se aplicó IQR porque la dispersión (IQR) fue 0 o inválida."
                })

        # mantener edad como numérica en el df final
        if edad_col in df_clean.columns:
            df_clean[edad_col] = pd.to_numeric(df_clean[edad_col], errors="coerce")

    # =========================================================
    # 5) satisfaccion_NPS: asegurar valores positivos con abs()
    # =========================================================
    nps_cols = [c for c in df_clean.columns if c.lower() in ["satisfaccion_nps", "nps", "satisfaccion"]]
    if nps_cols:
        nps_col = nps_cols[0]
        nps_num = pd.to_numeric(df_clean[nps_col], errors="coerce")

        affected = int((nps_num < 0).sum())

        df_clean[nps_col] = nps_num.abs()

        decisiones.append({
            "Acción": "Normalización de NPS a valores positivos",
            "Columna": nps_col,
            "Registros afectados": affected,
            "Justificación": (
                "Se corrigieron valores negativos en satisfaccion_NPS aplicando valor absoluto. "
                "Esto asegura interpretabilidad consistente en la métrica."
            )
        })

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
            "Justificación": "Se eliminaron filas únicamente por reglas explícitas (outliers de edad)."
        })

    return df_clean, pd.DataFrame(decisiones)

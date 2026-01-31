import pandas as pd
import numpy as np
import re


def normalize_categoria_value(x):
    """
    Normaliza valores similares en columnas de categoría.
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower()

    # reemplazo directo para ruido
    if s in ["???", "??", "?", "nan", "none", "null", "sin dato", "desconocido"]:
        return "otros"

    # limpiar caracteres (mantener letras/números/espacios)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # normalización fuerte: smartphone (todas sus variantes)
    if s.replace(" ", "") in ["smartphone", "smartphones"]:
        return "smartphone"

    mapping = {
        "laptop": "laptop",
        "laptops": "laptop",
        "tablet": "tablet",
        "tablets": "tablet",
        "accesorio": "accesorios",
        "accesorios": "accesorios",
    }

    return mapping.get(s, s)


def clean_inventario_central(df: pd.DataFrame):
    """
    Limpieza específica para datasets de inventario.
    Devuelve (df_clean, decisiones_df).
    """
    decisiones = []
    df_clean = df.copy()

    before_rows = len(df_clean)

    # 1) Normalizar columna Categoria
    categoria_cols = [c for c in df_clean.columns if c.lower() in ["categoria", "categoría"]]
    if categoria_cols:
        cat_col = categoria_cols[0]
        before_unique = df_clean[cat_col].nunique(dropna=True)

        df_clean[cat_col] = df_clean[cat_col].apply(normalize_categoria_value)

        after_unique = df_clean[cat_col].nunique(dropna=True)
        decisiones.append({
            "Acción": "Normalización categórica",
            "Columna": cat_col,
            "Registros afectados": int(len(df_clean)),
            "Justificación": (
                "Se normalizaron valores similares (ej: smartphone/smartphones) "
                "para evitar fragmentación en análisis por categoría. "
                f"Únicos antes: {before_unique}, únicos después: {after_unique}."
            )
        })

        replaced_otros = int((df_clean[cat_col] == "otros").sum())
        decisiones.append({
            "Acción": "Reasignación de categoría no informativa",
            "Columna": cat_col,
            "Registros afectados": replaced_otros,
            "Justificación": "Valores como '???' no aportan significado, se renombraron a 'otros'."
        })

    # 2) Stock: nulos -> 0, negativos se quedan
    stock_cols = [c for c in df_clean.columns if c.lower() in ["stock_actual", "stock", "cantidad_stock"]]
    if stock_cols:
        stock_col = stock_cols[0]
        stock_numeric = pd.to_numeric(df_clean[stock_col], errors="coerce")

        nulls_before = int(stock_numeric.isna().sum())
        df_clean[stock_col] = stock_numeric.fillna(0)

        decisiones.append({
            "Acción": "Imputación de stock nulo",
            "Columna": stock_col,
            "Registros afectados": nulls_before,
            "Justificación": (
                "Se imputaron nulos con 0, interpretando ausencia de stock físico. "
                "Los valores negativos se conservaron (interpretación: unidades en lista de espera/backorder)."
            )
        })

    # 3) Costo_unitario: eliminar outliers extremos (fila)
    costo_cols = [c for c in df_clean.columns if c.lower() in ["costo_unitario_usd", "costo_unitario", "costo"]]
    if costo_cols:
        costo_col = costo_cols[0]
        costo_num = pd.to_numeric(df_clean[costo_col], errors="coerce")

        s = costo_num.dropna()
        removed = 0

        if not s.empty:
            Q1 = s.quantile(0.25)
            Q3 = s.quantile(0.75)
            IQR = Q3 - Q1

            if IQR > 0 and not pd.isna(IQR):
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                mask_out = (costo_num < lower) | (costo_num > upper) | (costo_num <= 0)
                removed = int(mask_out.sum())

                df_clean = df_clean.loc[~mask_out].copy()

                decisiones.append({
                    "Acción": "Eliminación de filas por costo outlier",
                    "Columna": costo_col,
                    "Registros afectados": removed,
                    "Justificación": (
                        "Se eliminaron registros con costos extremos (outliers) "
                        "que distorsionan métricas financieras. "
                        f"Umbrales IQR: [{lower:.2f}, {upper:.2f}] y además costo <= 0."
                    )
                })
            else:
                decisiones.append({
                    "Acción": "Revisión de outliers (sin acción)",
                    "Columna": costo_col,
                    "Registros afectados": 0,
                    "Justificación": "No se aplicó IQR porque la dispersión (IQR) fue 0 o inválida."
                })

        df_clean[costo_col] = pd.to_numeric(df_clean[costo_col], errors="coerce")

    # 4) Lead_Time: "25-30 días" -> "25-30", NaN -> "indefinido"
    lead_cols = [c for c in df_clean.columns if c.lower() in ["lead_time_dias", "lead_time", "leadtime"]]
    if lead_cols:
        lead_col = lead_cols[0]

        lead_before = df_clean[lead_col].copy()
        nulls_before = int(lead_before.isna().sum())

        def normalize_lead_time(v):
            if pd.isna(v):
                return "indefinido"

            s = str(v).strip().lower()

            if s in ["nan", "none", "null", ""]:
                return "indefinido"

            s = s.replace("días", "").replace("dias", "").strip()
            s = re.sub(r"\s*-\s*", "-", s)
            s = re.sub(r"[^0-9\-]", "", s)

            return s if s != "" else "indefinido"

        df_clean[lead_col] = df_clean[lead_col].apply(normalize_lead_time)

        decisiones.append({
            "Acción": "Estandarización Lead Time",
            "Columna": lead_col,
            "Registros afectados": int(len(df_clean)),
            "Justificación": (
                "Se estandarizó Lead_Time eliminando texto ('días') y dejando solo números/rangos. "
                f"Los NaN se reemplazaron por 'indefinido' (nulos detectados: {nulls_before})."
            )
        })

    after_rows = len(df_clean)
    removed_rows_total = before_rows - after_rows

    if removed_rows_total > 0:
        decisiones.append({
            "Acción": "Resumen de eliminación de filas",
            "Columna": "(dataset)",
            "Registros afectados": removed_rows_total,
            "Justificación": "Se eliminaron filas únicamente por reglas explícitas (outliers de costo_unitario)."
        })

    return df_clean, pd.DataFrame(decisiones)

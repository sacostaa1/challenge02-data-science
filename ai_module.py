import pandas as pd
from groq import Groq
import streamlit as st

def generate_ai_strategy(df_filtered, api_key, dataset_name):
    """Genera recomendaciones estratégicas usando Llama-3."""
    if not api_key:
        return "⚠️ Por favor, introduce tu API Key de Groq/Llama-3 en el panel lateral."

    try:
        client = Groq(api_key=api_key)
        
        # 1. Preparar un resumen ejecutivo para la IA
        numeric_desc = df_filtered.describe().to_string()
        total_rows = len(df_filtered)
        null_info = df_filtered.isnull().sum().to_string()
        
        prompt = f"""
        Eres un experto Consultor de Estrategia de Datos y Business Intelligence.
        Analiza el siguiente resumen estadístico del dataset '{dataset_name}' que ha sido filtrado por el usuario:
        
        INFORMACIÓN DEL DATASET:
        - Filas totales: {total_rows}
        - Resumen Estadístico:
        {numeric_desc}
        - Estado de Nulos:
        {null_info}
        
        TAREA:
        Escribe exactamente 3 párrafos de recomendaciones estratégicas basadas en estos números. 
        Párrafo 1: Análisis de la situación actual y salud de los datos.
        Párrafo 2: Oportunidades de negocio o riesgos detectados en las métricas (outliers, tendencias, desviaciones).
        Párrafo 3: Una estrategia accionable inmediata para mejorar la rentabilidad o eficiencia operativa.
        
        REGLAS:
        - Sé profesional, directo y basado en datos.
        - No uses listas, solo 3 párrafos de texto fluido.
        - Idioma: Español.
        """

        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        
        return completion.choices[0].message.content

    except Exception as e:
        return f"❌ Error al conectar con Llama-3: {str(e)}"
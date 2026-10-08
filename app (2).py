import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Configuración de página
st.set_page_config(
    page_title="Predicción de Rendimiento Agrícola",
    page_icon="🌾",
    layout="wide"
)

# Rutas de artefactos
ROOT = Path(".")
MODEL_PATH = ROOT / "models" / "modelo_rendimiento.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"

# Funciones de carga con caché para rendimiento
@st.cache_resource
def cargar_modelo():
    if not MODEL_PATH.exists():
        st.error(f"No se encuentra el modelo en '{MODEL_PATH}'. Asegúrate de subir la carpeta 'models/' a tu repositorio de GitHub.")
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def cargar_metadatos():
    if not METADATA_PATH.exists():
        st.error(f"No se encuentra el archivo de metadatos en '{METADATA_PATH}'.")
        return None
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

modelo = cargar_modelo()
metadata = cargar_metadatos()

if modelo is not None and metadata is not None:
    st.title("🌾 Predicción del Rendimiento Agrícola")
    st.markdown("Esta aplicación interactiva predice el rendimiento agrícola estimado en **toneladas por hectárea (t/ha)** utilizando el modelo predictivo **Lasso** optimizado en base a características del suelo y condiciones climáticas.")

    # Layout de columnas: Entrada de datos (Izquierda) | Resultados (Derecha)
    col_inputs, col_results = st.columns([1.2, 1])

    with col_inputs:
        st.subheader("🔬 Parámetros del Suelo y Clima")
        st.write("Introduce las condiciones de la parcela. Puedes marcar casillas para simular datos faltantes (NaN) y comprobar cómo el modelo responde usando la imputación inteligente.")

        inputs = {}
        # 1. Variables numéricas
        rangos = metadata["rangos_entrenamiento"]

        # Agrupar entradas en pestañas para mayor orden
        tab_nutrientes, col_clima, tab_suelo = st.tabs(["🧪 Nutrientes (N, P, K)", "🌤️ Clima", "🌱 Tipo de Suelo"])

        with tab_nutrientes:
            for var in ["N", "P", "K"]:
                info = rangos[var]
                col_s, col_chk = st.columns([3, 1])
                with col_chk:
                    usar = st.checkbox(f"Usar {var}", value=True, key=f"chk_{var}")
                with col_s:
                    val = st.slider(
                        f"{var} (mediana: {info['mediana']:.1f})",
                        float(info["min"]),
                        float(info["max"]),
                        float(info["mediana"]),
                        step=1.0,
                        disabled=not usar
                    )
                inputs[var] = val if usar else np.nan

        with col_clima:
            for var in ["temperature", "humidity", "ph", "rainfall"]:
                info = rangos[var]
                col_s, col_chk = st.columns([3, 1])
                with col_chk:
                    usar = st.checkbox(f"Usar {var}", value=True, key=f"chk_{var}")
                with col_s:
                    label_var = f"{var} (pH)" if var == "ph" else f"{var}"
                    val = st.slider(
                        f"{label_var} (mediana: {info['mediana']:.2f})",
                        float(info["min"]),
                        float(info["max"]),
                        float(info["mediana"]),
                        step=0.01 if var != "ph" else 0.1,
                        disabled=not usar
                    )
                inputs[var] = val if usar else np.nan

        with tab_suelo:
            col_s, col_chk = st.columns([3, 1])
            with col_chk:
                usar_suelo = st.checkbox("Usar Suelo", value=True, key="chk_soil_type")
            with col_s:
                lista_suelos = metadata["tipos_de_suelo"]
                suelo_sel = st.selectbox(
                    "Tipo de Suelo (soil_type)",
                    lista_suelos,
                    index=0,
                    disabled=not usar_suelo
                )
            inputs["soil_type"] = suelo_sel if usar_suelo else np.nan

    with col_results:
        st.subheader("📊 Predicción y Diagnóstico")

        # Crear un DataFrame con un único registro para predecir
        df_input = pd.DataFrame([inputs])

        # Predicción
        try:
            prediccion = modelo.predict(df_input)[0]

            # Obtener intervalos de predicción desde los metadatos (95% de confianza)
            iq = metadata["intervalo_residuos"]
            lim_inf = max(0.0, prediccion + float(iq["q025"]))
            lim_sup = prediccion + float(iq["q975"])

            # Mostrar resultado principal
            st.metric(
                label="Rendimiento Estimado (t/ha)",
                value=f"{prediccion:.3f} t/ha",
                delta=f"{(prediccion - float(metadata['linea_base_prueba']['mean'])):+.3f} vs media global"
            )

            # Mostrar intervalo de predicción (incertidumbre del modelo)
            st.success(f"🎯 **Intervalo de Predicción del 95%:**  \n**{lim_inf:.2f}** t/ha  a  **{lim_sup:.2f}** t/ha")
            st.caption("El intervalo refleja la variabilidad típica no explicada por el modelo utilizando los residuos calculados en validación cruzada.")

        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")

        # Sección informativa: Importancia de Variables en el modelo elegido
        st.markdown("--- ")
        st.subheader("🧠 Información Técnica de Respaldo")

        with st.expander("💡 ¿Qué variables dominan el modelo?", expanded=False):
            st.write("Aumento promedio del error (RMSE) al desordenar aleatoriamente cada variable:")
            imp_data = []
            for k, v in metadata["importancia_permutacion"].items():
                imp_data.append({"Variable": k, "Importancia (t/ha)": v["media"]})
            df_imp = pd.DataFrame(imp_data).sort_values(by="Importancia (t/ha)", ascending=False)
            st.dataframe(df_imp, use_container_width=True, hide_index=True)
            st.caption("Como se observa, el Nitrógeno (N), Potasio (K) y Fósforo (P) concentran prácticamente toda la capacidad de predicción del modelo.")

        with st.expander("📈 Desempeño del Modelo en Entrenamiento (Lasso)", expanded=False):
            st.markdown(
                f"* **R² en Prueba:** {metadata['metricas_prueba']['R2']:.3%}\n"
                f"* **RMSE en Prueba:** {metadata['metricas_prueba']['RMSE']:.3f} t/ha\n"
                f"* **MAE en Prueba:** {metadata['metricas_prueba']['MAE']:.3f} t/ha\n"
                f"* **Error Porcentual Medio (MAPE):** {metadata['metricas_prueba']['MAPE_%']:.2f}%\n"
                f"* **Mejora del error frente a la línea base:** {100 * (1 - metadata['metricas_prueba']['RMSE'] / metadata['linea_base_prueba']['RMSE']):.1f}%"
            )
            st.caption("El modelo se validó rigurosamente mediante validación cruzada de 5 pliegues y se testeo con un 20% de datos independientes nunca antes vistos.")

else:
    st.info("⚠️ Esperando que los archivos del modelo y sus metadatos estén disponibles en el repositorio.")

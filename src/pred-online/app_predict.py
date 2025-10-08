import streamlit as st
import joblib
import pandas as pd
import os 
import sys
sys.path.append(os.path.abspath('..'))  # Ajusta la ruta según la ubicación de 'pred-online'
from preprocessing.functions import expresiones_regulares, tokenizar, lematizar


# Diccionario de mapeo de etiquetas
label_map = {0: "Otros Trastornos",
             1: "T. de adaptación",
             2: "T. de ansiedad",
             3: "T. depresivos",
             4: "T. externalizantes",
             5: "T. personalidad"}

#  modelos disponibles
modelos_disponibles = {
    "RandomForest": "best_model_vectorizer/modelo_RN.pkl"
}

st.title("Clasificador de Grupo De Trastornos - Selección de Modelo")


modelo_seleccionado = st.selectbox(
    "Selecciona el modelo para predecir:", list(modelos_disponibles.keys()))

texto_usuario = st.text_area("Ingrese el texto clínico a clasificar:")

if st.button("Predecir"):
    if texto_usuario.strip() == "":
        st.warning("Por favor, ingrese un texto.")
    else:
        # Cargar modelo y vectorizador según selección
        modelo_path = modelos_disponibles[modelo_seleccionado]
        if not os.path.exists(modelo_path):
            st.error(f'El modelo "{modelo_seleccionado}" no está disponible.')
        else:
            modelo = joblib.load(modelo_path)
            vectorizer = joblib.load("best_model_vectorizer/vectorizer_RN.pkl")

            # Crear DataFrame
            df = pd.DataFrame({"texto": [texto_usuario]})

            # Preprocesamiento igual que en entrenamiento
            df = expresiones_regulares(df, "texto")
            df = tokenizar(df, "texto")
            df = lematizar(df, "texto")
            # Unir tokens para vectorizar
            df["texto_proc"] = df["texto"].apply(lambda x: " ".join(x))

            # Vectorizar
            X_vect = vectorizer.transform(df["texto_proc"])

            # Predecir
            pred = modelo.predict(X_vect)[0]
            st.success(f"Predicción: {label_map.get(pred, 'Desconocido')}")

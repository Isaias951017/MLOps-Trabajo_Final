import streamlit as st
import joblib
import pandas as pd
from src.preprocessing.functions import expresiones_regulares, tokenizar, lematizar
import os

# Diccionario de mapeo de etiquetas
label_map = {
    0: 'Otros Trastornos',
    1: 'T. Depresivos',
    2: 'T. Externalizantes',
    3: 'T. Personalidad',
    4: 'T. de Adaptación',
    5: 'T. de Ansiedad'
}

#  modelos disponibles
modelos_disponibles = {
    'SVC': 'models/modelo_svc.pkl',
    'LogisticRegression': 'models/modelo_LR.pkl',
    'RandomForest': 'models/modelo_RN.pkl',
    'XGBC': 'models/modelo_XGB.pkl'
}

st.title("Clasificador de Grupo De Trastornos - Selección de Modelo")


modelo_seleccionado = st.selectbox("Selecciona el modelo para predecir:", list(modelos_disponibles.keys()))

texto_usuario = st.text_area("Ingrese el texto clínico a clasificar:")

if st.button("Predecir"):
    if texto_usuario.strip() == "":
        st.warning("Por favor, ingrese un texto.")
    else:
        # Cargar modelo y vectorizador según selección
        modelo_path = modelos_disponibles[modelo_seleccionado]
        if not os.path.exists(modelo_path):
            st.error(f"El modelo '{modelo_seleccionado}' no está disponible.")
        else:
            modelo = joblib.load(modelo_path)
            vectorizer = joblib.load('models/vectorizer.pkl')

            # Crear DataFrame
            df = pd.DataFrame({'texto': [texto_usuario]})

            # Preprocesamiento igual que en entrenamiento
            df = expresiones_regulares(df, 'texto')
            df = tokenizar(df, 'texto')
            df = lematizar(df, 'texto')
            # Unir tokens para vectorizar
            df['texto_proc'] = df['texto'].apply(lambda x: " ".join(x))

            # Vectorizar
            X_vect = vectorizer.transform(df['texto_proc'])

            # Predecir
            pred = modelo.predict(X_vect)[0]
            st.success(f"Predicción: {label_map.get(pred, 'Desconocido')}")
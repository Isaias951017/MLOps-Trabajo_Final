# Predicción de Historias Clínicas API

Este módulo implementa una API web interactiva  para la predicción automática de grupos de trastornos psicológicos a partir de textos clínicos, utilizando modelos de machine learning entrenados previamente. La interfaz está desarrollada con Streamlit y permite al usuario ingresar el texto clínico y obtener la predicción del grupo de trastorno correspondiente con el modelo Random Forest que fue el seleccionado para las predicciones.

## Estructura de Carpetas y Archivos

- **src/pred-online/**
  - `app_predict.py`: Script principal de la API de predicción. Carga el modelo y el vectorizador, realiza el preprocesamiento del texto y muestra la predicción en la interfaz web.
- **src/preprocessing/**
  - `functions.py`: Contiene las funciones de preprocesamiento utilizadas para limpiar, tokenizar y lematizar el texto antes de la predicción.
- **best_model_vectorizer/**
  - `modelo_RN.pkl`: Modelo RandomForest entrenado.
  - `vectorizer_RN.pkl`: Vectorizador Tfidf, utilizado para transformar el texto en vectores numéricos.

## Explicación del Código

- El usuario ingresa un texto clínico en la interfaz web.
- El script carga el modelo y el vectorizador.
- El texto se preprocesa (expresiones regulares, tokenización, lematización) igual que en el entrenamiento.
- El texto procesado se vectoriza y se pasa al modelo para obtener la predicción.
- El resultado se muestra en pantalla,indicando el grupo de trastorno identificado.

## Comandos para Ejecutar la API

Desde la terminal, navega a la carpeta `src/pred-online` y ejecuta:

```powershell
.venv/Script/activate.ps1
```
```powershell
cd src\pred-online
```
```powershell
streamlit run app_predict.py
```

Esto abrirá la interfaz web en tu navegador para realizar predicciones.

## Rutas y Localización de Carpetas

- El script `app_predict.py` se encuentra en: `src/pred-online/app_predict.py`
- Los modelos y vectorizadores deben estar en la carpeta: `best_model_vectorizer/` (ubicada en el mismo nivel que `src`)
- Las funciones de preprocesamiento están en: `src/preprocessing/functions.py`

Asegúrate de que las rutas relativas sean correctas y que los archivos `.pkl` estén disponibles para que la API funcione correctamente.
# Implementación de Prefect con Optuna & MLflow

Este módulo orquesta el flujo completo de entrenamiento de modelos de machine learning para clasificación de historias clínicas, integrando Prefect para la gestión de tareas, Optuna para la optimización de hiperparámetros y MLflow para el seguimiento de experimentos y modelos.

## Estructura de Carpetas y Archivos

- **src/training/**
  - `task_train_prefect.py`: Script principal que define los tasks, flows y reportes del proceso de entrenamiento.
- **src/etl.py**: Módulo para la generación y limpieza inicial de datos.
- **src/feature_engineer.py**: Módulo para el procesamiento y creación de nuevas características.
- **src/utils/conexion.py**: Módulo para la conexión y extracción de datos desde la base de datos.
- **src/train_with_mlflow_optuna.py**: Clase que encapsula el entrenamiento y optimización con Optuna y MLflow.

## Explicación General del Código

El script implementa un flujo de entrenamiento que incluye:
- Extracción y limpieza de datos desde la base de datos.
- Ingeniería de características sobre los datos extraídos.
- Entrenamiento de modelos (RandomForest, LogisticRegression) con optimización de hiperparámetros usando Optuna.
- Registro y seguimiento de experimentos y modelos con MLflow.
- Generación de reportes y tablas resumen como artifacts en Prefect.

## Componentes Principales

### Tasks
- **task_generate_data**: Extrae y limpia los datos desde la base de datos, genera un resumen del dataset como artifact.
- **task_feature_engineering**: Aplica procesamiento de texto y codificación de variables, genera un resumen de ingeniería de características como artifact.
- **task_train_with_optuna**: Entrena el modelo seleccionado, realiza la optimización de hiperparámetros con Optuna y MLflow, y genera tablas resumen de los trials y mejores parámetros.
- **task_create_model_report**: Genera un reporte en formato markdown con los resultados del entrenamiento, métricas y mejores hiperparámetros.

### Flows
- **train_model_flow**: Orquesta el flujo completo de entrenamiento, ejecutando los tasks en orden y generando los artifacts.
- **compare_models_flow**: Permite comparar múltiples modelos y métricas en un solo flujo, generando una tabla comparativa.

### Artifacts Created
- **Table Artifacts**: Tablas resumen del dataset, ingeniería de características, resultados de Optuna y comparación de modelos.
- **Model Training Reports**: Reportes en markdown con detalles del entrenamiento, métricas y mejores hiperparámetros.

## Comandos para Configurar MLflow y Prefect

- **Configurar la base de datos de MLflow (local):**

```powershell
cd src\training
```

```powershell
mlflow ui --backend-store-uri sqlite:///backend db
```

- **Montar el servidor de Prefect:**

```powershell
prefect server start
```

## Comandos para Ejecutar el Script

Desde la terminal, navega a la carpeta `src/training` y ejecuta:
```powershell
cd src\training
```

```powershell
python task_train_prefect.py
```

## Rutas y Localización de Carpetas

- El script principal está en: `src/training/task_train_prefect.py`
- Los módulos auxiliares están en: `src/etl.py`, `src/feature_engineer.py`, `src/utils/conexion.py`, `src/train_with_mlflow_optuna.py`
- Los artifacts y reportes se generan y almacenan automáticamente por Prefect y MLflow en las rutas configuradas.

---

Esta documentación cubre la estructura, componentes y comandos esenciales para ejecutar y monitorear el flujo de entrenamiento automatizado con Prefect, Optuna y MLflow.

# Análisis Exploratorio de Datos (EDA)

## Descripción del Problema
El objetivo de este análisis es explorar y comprender el comportamiento de los pacientes a partir de un conjunto de historias clínicas. Se busca identificar patrones, distribuciones y posibles desbalances en las variables relevantes, con el fin de sentar las bases para futuros modelos predictivos o de clasificación.

## Descripción del Dataset
El dataset utilizado, `historias_clinicas`, contiene información clínica de pacientes, incluyendo variables como:
- **edad**: Edad del paciente.
- **sexo**: Género del paciente (masculino/femenino).
- **grupo**: Grupo al que pertenece el paciente (variable objetivo).
- **especialidad_medica**: Especialidad médica asociada(Psicologia).
- **concatenada**: Variable compuesta relevante para el análisis.

## Proceso de Análisis Exploratorio
El EDA realizado en el notebook `eda.ipynb` incluyó los siguientes pasos:
1. **Carga y visualización inicial del dataset**
2. **Revisión de tipos de datos y dimensiones**
3. **Análisis de la variable objetivo y distribución de clases**
4. **Revisión y tratamiento de valores nulos**
5. **Detección de filas duplicadas**
6. **Estadísticas descriptivas**
7. **Visualización de variables categóricas y numéricas**

## Resultados Obtenidos
### Distribución de la variable objetivo
Se observó un desbalance considerable en la variable `grupo`, lo que puede afectar futuros modelos de clasificación.

### Valores nulos y duplicados
- Se identificaron y eliminaron filas con valores nulos en la columna `concatenada`.
- No se encontraron filas completamente duplicadas, solo repeticiones en la columna `especialidad_medica`.

### Estadísticas descriptivas
- La variable `edad` presenta una distribución con posibles valores atípicos, visualizados mediante boxplot.
- Las variables categóricas muestran una distribución desigual entre los grupos y sexos.

### Visualizaciones
A continuación se presentan algunas de las visualizaciones generadas:

### Boxplot Edad
![Boxplot Edades](notebooks\imagenes_eda\boxplot_edad.png)

#### Distribución por Sexo
![Distribución Sexo](notebooks\imagenes_eda\distribucion_sexo.png)

#### Distribución por Grupo
![Distribución Grupo](notebooks\imagenes_eda\distribucion_grupo.png)

#### Distribución de Edad
![Distribución Edad](notebooks\imagenes_eda\histograma_edad.png)

#### Distribución Edad por Sexo
![Distribución Edad Sexo](notebooks\imagenes_eda\histograma_edad_sexo.png)

#### Distribución Grupo por Sexo
![Distribución Grupo Sexo](notebooks\imagenes_eda\distribucion_grupo_sexo.png)

## Conclusiones
- El dataset presenta un desbalance en la variable objetivo `grupo`, lo que sugiere la posible necesidad de técnicas de balanceo para futuros modelos.
- No existen filas duplicadas completas, por lo que no se requiere limpieza adicional en ese aspecto.
- La variable `edad` muestra una distribución amplia, con posibles valores atípicos que deben considerarse en el preprocesamiento.
- Las visualizaciones permiten identificar patrones y relaciones entre variables, útiles para la selección de características y el diseño de modelos predictivos.

Este análisis exploratorio proporciona una base sólida para el desarrollo de modelos de machine learning y la toma de decisiones informadas sobre el preprocesamiento y selección de variables.

## Resúmenes de scripts y notebooks principales

### src/training/etl.py
Contiene la clase `EDAdataset`, que realiza transformaciones básicas de limpieza y preprocesamiento sobre el DataFrame de historias clínicas. Incluye funciones para:
- Convertir nombres de columnas a minúsculas.
- Eliminar filas nulas en columnas clave.
- Capitalizar la columna de grupo.
- Aplicar un pipeline de EDA para dejar el dataset listo para análisis y modelado.

### src/preprocessing/feature_engineer.py (feature_engineer)
Contiene la clase `PreprocesadorTexto`, que centraliza el procesamiento de texto y la codificación de variables categóricas. Sus métodos principales son:

- **__init__**: Inicializa el objeto con el DataFrame y el conjunto de stopwords personalizadas, cargando el modelo de spaCy para español.
- **concatenar_columnas**: Une dos columnas de texto (por defecto 'subjetivo' y 'objetivo') en una nueva columna llamada 'concatenada'.
- **expresiones_regulares**: Limpia el texto de una columna, convirtiendo a minúsculas y eliminando caracteres no alfabéticos.
- **tokenizar**: Tokeniza el texto de una columna, eliminando stopwords, signos de puntuación y espacios, usando spaCy.
- **lematizar**: Aplica lematización sobre los tokens de una columna, obteniendo la raíz de cada palabra.
- **label_encodering**: Codifica variables categóricas (como sexo o grupo) en valores numéricos usando LabelEncoder, y devuelve el mapeo de clases.
- **procesar**: Ejecuta el pipeline completo de procesamiento de texto (limpieza, tokenización, lematización) y codificación de variables categóricas, devolviendo el DataFrame procesado y los mapeos.

Esta clase permite transformar los textos clínicos y variables categóricas en representaciones numéricas útiles para modelos de machine learning.

### src/training/train_with_mlflow.py
Define la clase `TrainMlflow`, que automatiza el entrenamiento de modelos de machine learning con seguimiento en MLflow. Sus principales tareas son:
- Separar el dataset en entrenamiento y prueba.
- `vectorizer`: Vectorizar textos con TF-IDF.
- Entrenar modelos (por ejemplo, Random Forest, Logistic Regression) dentro de un pipeline.
- Registrar parámetros, métricas y modelos en MLflow para facilitar la trazabilidad y comparación de experimentos.

### src/training/train_with_mlflow_optuna.py
Implementa la clase `TrainMlflowOptuna`, que extiende el entrenamiento con MLflow añadiendo optimización de hiperparámetros con Optuna. Permite:
- Definir espacios de búsqueda para hiperparámetros.
- Ejecutar múltiples pruebas (trials) para encontrar la mejor configuración.
- Registrar automáticamente los resultados y el mejor modelo en MLflow.
- Soporta varios algoritmos y métricas de optimización (accuracy, f1, etc.).

### models/task_train.ipynb (Orchestator.py)
Notebook que integra todo el flujo de procesamiento y modelado:
- Carga y limpieza de datos clínicos.
- Ingeniería de características y codificación.
- Vectorización de texto y entrenamiento de modelos (Logistic Regression, Random Forest, XGBoost, SVC).
- Evaluación de métricas y visualización de resultados.
- Ejemplos de uso de MLflow y Optuna para experimentación y optimización.

Sirve como guía práctica y reproducible para el pipeline completo de NLP y clasificación.

### src/utils/conexion.py
Contiene la clase `SQLConnection`, encargada de gestionar la conexión y extracción de datos desde una base de datos SQL Server. Sus principales funciones son:
- Leer parámetros de conexión desde variables de entorno y argumentos.
- Construir el string de conexión compatible con SQLAlchemy y pyodbc.
- Crear el motor de conexión (`engine`) para ejecutar consultas.
- Leer archivos SQL y ejecutar consultas parametrizadas.
- Devolver los resultados como un DataFrame de pandas listo para su análisis o procesamiento.

## Haz click el el icono para cononcer más y preguntar a la IA.
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Isaias951017/MLOps-Trabajo_Final)


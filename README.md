
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

## Haz click el el icono para cononcer más y preguntar a la IA.
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Isaias951017/MLOps-Trabajo_Final)

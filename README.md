# Optimización Aeroportuaria: Predicción de Retrasos de Vuelos

Este repositorio contiene un análisis y modelo de Machine Learning para predecir los retrasos en vuelos, utilizando un dataset de operaciones aeroportuarias. El objetivo es identificar los factores que más influyen en los retrasos y construir un modelo predictivo robusto.

## Contenido del Cuaderno

### 1. Datos Iniciales

Se realiza la carga de los datos de vuelos (`flights.csv`) utilizando `pandas`. Se explora la forma y las primeras filas del dataset para una visión inicial de la estructura.

### 2. Exploración y Resumen de Datos

Esta sección se enfoca en entender las características del dataset:

- **`datos.info()`**: Muestra un resumen de las columnas, tipos de datos y valores no nulos.
- **`datos.describe()`**: Proporciona estadísticas descriptivas para las columnas numéricas y categóricas.
- **Manejo de Tiempos**: Se clarifica la interpretación de las columnas de tiempo (`delay`, `arrival_time`, `departure_time`).

### 3. Exploración de Datos con Gráficas

Se utilizan visualizaciones (`seaborn` y `matplotlib`) para entender la distribución de las variables y sus relaciones:

- **Distribución de Retrasos**: Boxplot e histograma de la variable `delay`, mostrando la media y mediana.
- **Impacto de Aerolíneas**: Gráficos de barras para el retraso promedio por aerolínea y el número de vuelos por aerolínea.
- **Vuelos Schengen/No-Schengen**: Análisis del retraso promedio y conteo de vuelos por tipo de espacio aéreo.
- **Días Feriados**: Comparación del retraso promedio entre días feriados y no feriados.
- **Tipo de Aeronave**: Distribución del número de vuelos por tipo de aeronave.
- **Distribución de Tiempos**: Histogramas de `arrival_time` y `departure_time` utilizando la regla de Freedman-Diaconis para el ancho de los bins.

### 4. Preparación y Transformación de Datos

- **Creación de Variables Temporales**: Se genera una columna `date` a partir de `year` y `day`, y luego se extraen `is_weekend` y `day_name`.
- **Codificación de Variables Categóricas**: Las variables categóricas (`schengen`, `is_holiday`, `is_weekend`) se convierten a formato numérico (0/1). Las variables `airline`, `aircraft_type`, `origin`, y `day_name` se transforman utilizando `OneHotEncoder` para convertirlas en variables dummy.

### 5. Selección de Variables y Análisis de Correlación

- **Correlación de Tiempos**: Se analiza la correlación entre `arrival_time` y `departure_time`.
- **Eliminación de Columnas Redundantes**: Se eliminan columnas como `flight_id`, `departure_time`, `day`, `year` y `date` del dataframe para el modelado.

### 6. Baseline del Modelo

Se establece un modelo baseline utilizando `DummyRegressor` con diferentes estrategias (`mean`, `median`, `quantile`, `constant`) para comparar el rendimiento de modelos más complejos. Se definen métricas de regresión como RMSE, MAE y R2.

### 7. Modelo de Regresión y Evaluación Inicial

- **RandomForestRegressor**: Se entrena un `RandomForestRegressor` con `max_depth=5` y `random_state=520`.
- **Métricas de Rendimiento**: Se evalúa el modelo utilizando las métricas de regresión (RMSE, MAE, R2).
- **Visualizaciones de Diagnóstico**: Se utilizan `yellowbrick` para generar gráficos de error de predicción y gráficos de residuos.
- **Validación Cruzada**: Se realiza una validación cruzada (KFold con 5 splits) para evaluar la estabilidad del modelo y obtener una estimación más robusta de sus métricas.

### 8. Optimización de Hiperparámetros y Selección de Características

- **Importancia de Características**: Se calcula la importancia de las características del modelo `RandomForestRegressor`.
- **Selección de Características**: Se experimenta con diferentes subconjuntos de características más importantes para ver cómo afectan las métricas del modelo, buscando un balance entre complejidad y rendimiento.
- **Optimización con GridSearchCV**: Se utiliza `GridSearchCV` para encontrar la mejor combinación de hiperparámetros (`max_depth`, `min_samples_leaf`, `min_samples_split`, `n_estimators`) para el `RandomForestRegressor` sobre las características seleccionadas, usando `KFold` para la validación cruzada.
- **Evaluación del Modelo Optimizado**: Se evalúan las métricas del mejor modelo encontrado por `GridSearchCV`.

### 9. Serialización del Modelo

- **Guardar Modelo**: El mejor estimador obtenido del `GridSearchCV` se serializa usando `pickle` y se guarda como `champion.pkl`.
- **Cargar Modelo**: Se demuestra cómo cargar el modelo serializado y realizar una predicción con una nueva muestra de datos.

# Pump it Up: Data Mining the Water Table

Clasificación multiclase del estado operativo de bombas de agua en Tanzania.
Competición de [DrivenData](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/).

Práctica final del módulo de Machine Learning del Máster en Data Science
y Business Analytics.

## El problema

Predecir si una bomba está `functional`, `functional needs repair` o
`non functional` a partir de 39 variables: ubicación, tipo de extracción,
gestión, financiación y calidad del agua.

- 59.400 bombas de entrenamiento, 14.850 de test
- Clases desbalanceadas: `functional needs repair` es menos del 8% del total
- Métrica de evaluación: accuracy

## Qué hay en el repo

| Archivo | Contenido |
|---|---|
| `pump_it_up_final.ipynb` | Notebook completo: EDA, limpieza, modelado y conclusiones |
| `training_values.csv` | Características de las 59.400 bombas de entrenamiento |
| `training_labels.csv` | Estado real de esas bombas |
| `test_value.csv` | Las 14.850 bombas a predecir |

El CSV de predicciones (`entrega_final.csv`) no está incluido: se genera al
ejecutar el notebook y es el archivo que se sube a DrivenData.

## Enfoque

**Ceros que en realidad son faltantes.** Un tercio de las bombas tenía
`construction_year = 0` y `population = 0`. Se convierten a `NaN` antes de
imputar, con banderas que registran dónde faltaba el dato.

**Imputación por zona en tres niveles.** Mediana de su región+distrito, luego
de su región, y global como último recurso. Rellenar con la mediana global
metería el mismo valor en zonas geográficamente muy distintas.

**Frequency encoding en variables de alta cardinalidad.** `funder` e
`installer` tienen ~1.900 categorías cada una. Se sustituye cada categoría
por su número de apariciones: ~1.900 categorías pasan a una columna numérica
sin inventar un orden falso ni usar la variable objetivo.

**Split antes de la limpieza.** Todas las medianas, modas y frecuencias se
calculan sobre el train y se aplican al resto, para que las filas de
validación no contribuyan a los valores con los que se rellenan a sí mismas.

## Modelos comparados

Random Forest (baseline y ajustado), LightGBM, CatBoost, XGBoost, y Random
Forest con oversampling SMOTE. Búsqueda de hiperparámetros con
`RandomizedSearchCV`.

Todos los resultados se generan automáticamente en la tabla comparativa del
notebook.

| Métrica | Valor |
|---|---|
| Modelo final | Random Forest |
| Accuracy en validación | _(rellenar)_ |
| Score en el leaderboard | _(rellenar)_ |

## Conclusiones

**La representación de los datos importó más que el algoritmo.** El frequency
encoding fue el único cambio que movió el accuracy de forma clara. Ninguno de
los tres modelos de boosting superó al Random Forest, y la búsqueda automática
de hiperparámetros aportó muy poco.

**Refinar la imputación apenas cambió nada.** Random Forest es robusto a cómo
se rellenen los huecos: al partir por umbrales, un valor imputado razonable
cae del mismo lado del corte que el real la mayoría de las veces.

**El modelo se apoya sobre todo en la geografía.** `longitude`, `latitude` y
`gps_height` ocupan tres de los cuatro primeros puestos por importancia, junto
a `quantity_dry`. Es la línea con más margen de mejora: variables como la
distancia a la ciudad más cercana o clusters de coordenadas.

**La clase minoritaria es el límite del modelo.** El recall en
`functional needs repair` es bajo y no se arregla con más árboles. SMOTE lo
mejora a costa del accuracy global: son objetivos distintos. Para esta
competición se prioriza accuracy porque es la métrica evaluada, pero en
planificación real de mantenimiento la decisión sería la contraria.

## Cómo ejecutarlo

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
pip install lightgbm catboost xgboost imbalanced-learn
```
## Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · CatBoost · imbalanced-learn

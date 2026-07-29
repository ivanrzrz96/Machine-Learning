# Pump it Up: Data Mining the Water Table

Clasificación multiclase del estado operativo de bombas de agua en Tanzania.
Competición de [DrivenData](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/).

Práctica final del módulo de Machine Learning del Máster en Data Science
y Business Analytics.

---

## El problema

El gobierno de Tanzania mantiene decenas de miles de puntos de agua, y muchos
están averiados. Enviar cuadrillas a revisarlos uno por uno es caro y lento.
Si se pudiera predecir cuáles están fallando a partir de los datos que ya
existen, el mantenimiento se podría priorizar.

La tarea es clasificar cada bomba en una de tres categorías:

- `functional` — funciona
- `functional needs repair` — funciona pero necesita reparación
- `non functional` — no funciona

A partir de 39 variables: ubicación, tipo de extracción, quién la gestiona,
quién la financió, calidad y cantidad de agua, año de construcción.

- 59.400 bombas de entrenamiento, 14.850 de test
- La métrica de evaluación es accuracy

---

## Los tres problemas reales de estos datos

El dataset parece limpio a primera vista. No lo está.

### 1. Miles de ceros que en realidad son datos faltantes

`df.info()` no reporta valores nulos en las columnas numéricas. Pero un tercio
de las bombas tiene `construction_year = 0` y `population = 0`. Ninguna bomba
se construyó en el año 0, y una bomba que no sirve a nadie es un registro
sospechoso.

Lo mismo con `longitude = 0` y `gps_height = 0`: Tanzania está entre los 29 y
41 grados de longitud, y su altitud no es cero.

Si se dejan como ceros, el modelo los trata como valores reales y aprende un
patrón falso: que hay un grupo de bombas construidas en el año 0.

`latitude` tiene un matiz añadido — el valor centinela no es `0` sino
`-2e-08`, así que buscarlo con `== 0` no lo encuentra.

### 2. Variables categóricas con casi 2.000 valores distintos

`funder` (quién financió la bomba) e `installer` (quién la instaló) tienen
alrededor de 1.900 categorías cada una. Con One-Hot Encoding generarían miles
de columnas casi vacías, y el modelo se ahogaría en dimensionalidad.

### 3. Clases desbalanceadas

`functional needs repair` es menos del 8% del total. Un modelo que ignore por
completo esa clase puede tener buen accuracy global, y eso es exactamente lo
que tiende a pasar. Pero es la clase operativamente más interesante: son las
bombas que se pueden salvar interviniendo a tiempo.

---

## Cómo los resolví

### Marcar los ceros y rellenar por zona geográfica

Convierto los ceros imposibles a `NaN` y los relleno con la **mediana de su
zona**, en tres niveles: región+distrito, región, y global como último
recurso.

Uso mediana y no media porque hay valores extremos que la arrastran: en
`population` la mediana está sobre 25 pero la media pasa de 180, porque hay
zonas con 30.000 habitantes. Rellenar con la media metería un valor irreal en
miles de filas.

Y uso la mediana **de la zona** y no la global porque una bomba de una región
montañosa y otra de la costa no tienen la misma altitud. Rellenar con el mismo
número en zonas geográficamente distintas destruye información.

Antes de rellenar, creo banderas (`construction_year_missing`,
`missing_population`) que registran dónde faltaba el dato. La hipótesis era
que la ausencia de datos podía ser informativa: quizá las bombas peor
documentadas están peor mantenidas.

### Frequency encoding en las variables de alta cardinalidad

En lugar de One-Hot, sustituyo cada categoría de `funder` e `installer` por
**su número de apariciones**. `government of tanzania` pasa a ser un número
alto, un financiador que aparece una sola vez pasa a ser 1.

Esto resuelve dos cosas a la vez:

- ~1.900 categorías se convierten en **una sola columna numérica**
- La frecuencia es informativa en sí misma: los financiadores que aparecen
  mucho son organizaciones grandes y consolidadas; los que aparecen una vez
  son proyectos pequeños o errores de transcripción

Descarté las alternativas por motivos concretos. **Label encoding** inventa un
orden que no existe: que `unicef` reciba un número mayor que `danida` no
significa nada. **Target encoding** usa directamente la variable objetivo, lo
que da fuga de información: si una categoría aparece dos veces, el modelo
memoriza esos dos casos en lugar de aprender un patrón.

### Split antes de la limpieza

Este fue el error que corregí a mitad del proyecto y merece explicación.

Mi limpieza calcula medianas, modas y frecuencias — números que *aprendo
mirando los datos*. Si los calculo sobre el dataset completo y después
separo train y validación, las filas de validación han contribuido a calcular
los valores con los que se rellenan a sí mismas. Eso es fuga de información,
y el accuracy de validación sale mejor de lo que realmente es.

El orden correcto es: separar primero, aprender los parámetros mirando **solo
el train**, y aplicar esos mismos parámetros al train, a la validación y al
test. El test se imputa con las medianas del train, nunca con las suyas.

### Eliminar redundancia con criterio

El dataset trae grupos de columnas que dicen lo mismo con distinto nivel de
detalle: `extraction_type` / `extraction_type_group` / `extraction_type_class`,
`payment` / `payment_type`, y varios más.

Me quedo con la versión más detallada de cada grupo, porque las agrupadas se
pueden derivar de ella pero no al contrario. Antes de decidir, verifico la
redundancia real con **Cramér's V** (la correlación de Pearson no sirve para
categóricas). El par más asociado resultó ser `scheme_management` con
`management` — quién gestiona el proyecto frente a quién gestiona la bomba
concreta. Pero no llega a 1: hay casos donde son entidades distintas, y esa
diferencia es información útil, así que mantengo las dos.

---

## Resultados

Comparé seis configuraciones sobre el mismo conjunto de validación:
Random Forest baseline, Random Forest ajustado, LightGBM, CatBoost, XGBoost y
Random Forest con oversampling SMOTE. Más búsqueda de hiperparámetros con
`RandomizedSearchCV` sobre los dos mejores.

**El modelo final es un Random Forest.** Ninguno de los tres modelos de
boosting lo superó, y la búsqueda automática de hiperparámetros no encontró
combinaciones apreciablemente mejores que las elegidas a mano.

Los accuracies exactos de cada configuración se generan automáticamente en la
tabla comparativa del notebook, a partir de los resultados reales de la
ejecución. No hay ningún número escrito a mano en el análisis.

### Qué variables acabó usando el modelo

Lo más revelador del proyecto está en el gráfico de importancia:

- **`quantity_dry` es la variable más importante.** Confirma la hipótesis
  principal del EDA: si la bomba está seca, es muy improbable que funcione.
- **`longitude`, `latitude` y `gps_height` ocupan tres de los cuatro primeros
  puestos.** El modelo se apoya masivamente en *dónde* está la bomba.
- **`funder` e `installer` entran en el top 12**, lo que valida el frequency
  encoding: reducir 1.900 categorías a una columna numérica conservó señal
  real.
- **Las banderas de faltante no aparecen entre las 20 primeras.** La hipótesis
  de que la ausencia de datos sería predictiva **no se confirmó**.

---

## Conclusiones

**La representación de los datos importó más que el algoritmo.** Es lo
contrario de lo que esperaba al empezar. El frequency encoding fue el único
cambio que movió el accuracy de forma clara; cambiar de familia de algoritmo
no aportó nada.

**Refinar la imputación apenas cambió el resultado.** Probé imputación por
zona en tres niveles, banderas de faltante y relleno de categóricas por moda.
Ninguna produjo un salto apreciable. Random Forest es robusto a cómo se
rellenen los huecos: al partir por umbrales, un valor imputado razonable cae
del mismo lado del corte que el real la mayoría de las veces.

**Las banderas de faltante no funcionaron, y merece decir por qué.** El
cálculo de importancia de scikit-learn favorece a las variables continuas
frente a las binarias — una continua ofrece miles de puntos de corte, una
bandera solo uno. Y además, al imputar por mediana de zona, todas las bombas
sin dato de una misma zona quedan con el mismo valor, que el árbol puede
aislar por sí solo. La bandera le decía algo que ya podía deducir.

**La clase minoritaria es el límite real del modelo, no el algoritmo.** El
recall en `functional needs repair` es bajo y no se arregla con más árboles.
SMOTE lo mejora a costa del accuracy global.

Esto no es un fallo, es un conflicto de objetivos. Para esta competición
prioricé accuracy porque es la métrica evaluada. Pero en planificación real de
mantenimiento la decisión sería la contraria: encontrar una bomba averiada de
verdad vale mucho más que equivocarse con una que funcionaba. **La métrica
correcta depende de para qué se use el modelo**, y ese es probablemente el
aprendizaje más transferible del proyecto.

### Qué probaría con más tiempo

Dado que la geografía es donde está la señal, ahí está el margen: distancia a
la ciudad más cercana, densidad de bombas en el entorno, clusters de
coordenadas. También un ensemble por votación entre Random Forest, XGBoost y
CatBoost, que fallan en casos distintos. Y ajustar el umbral de probabilidad
de la clase minoritaria en lugar de usar SMOTE, que da un control más fino
del intercambio entre accuracy y recall.

---

## Qué hay en el repo

| Archivo | Contenido |
|---|---|
| `pump_it_up_final.ipynb` | Notebook completo: EDA, limpieza, modelado y conclusiones |
| `training_values.csv` | Características de las 59.400 bombas de entrenamiento |
| `training_labels.csv` | Estado real de esas bombas |
| `test_value.csv` | Las 14.850 bombas a predecir |

El CSV de predicciones (`entrega_final.csv`) no está incluido: se genera al
ejecutar el notebook y es el archivo que se sube a DrivenData.

## Cómo ejecutarlo

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
pip install lightgbm catboost xgboost imbalanced-learn
```

Los datos ya están en el repo. Abre el notebook y ejecuta
`Kernel → Restart & Run All`.

## Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · CatBoost · imbalanced-learn

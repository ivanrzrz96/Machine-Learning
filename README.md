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

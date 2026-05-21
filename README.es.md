# Clustering K-Means — Segmentos de Vivienda en California

> Segmentación no supervisada de censos de California usando K-Means (k=6) sobre ingreso medio, latitud y longitud — produciendo clústeres geográficamente coherentes que un Árbol de Decisión supervisado puede reproducir con el 100% de fidelidad, validando la consistencia de los clústeres.

---

## Problema

Agrupar distritos de vivienda de California en segmentos significativos basándose en ubicación e ingresos — sin etiquetas predefinidas. Analistas inmobiliarios, urbanistas y prestamistas usan la segmentación geográfica-económica para entender los mercados regionales de vivienda, identificar áreas desatendidas y establecer precios ajustados al riesgo. Este es un problema de aprendizaje no supervisado: no hay una "respuesta correcta", solo agrupaciones significativas frente a no significativas.

## Dataset

- **Fuente:** Dataset California Housing (scikit-learn / GitHub)
- **Tamaño:** 20.640 registros de censos
- **Características completas disponibles:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, MedHouseVal
- **Características usadas:** Solo 3 — `MedInc`, `Latitude`, `Longitude`

La restricción a 3 características es intencional: la ubicación (lat/lon) captura la dinámica del mercado inmobiliario regional, y los ingresos capturan el nivel socioeconómico. Juntas definen segmentos que son tanto geográficamente interpretables como económicamente significativos.

## Pipeline

| Paso | Acción |
|---|---|
| Selección de características | `MedInc`, `Latitude`, `Longitude` (3 características, según especificación del proyecto) |
| División train/test | 80/20 (16.512 entrenamiento / 4.128 prueba) |
| K-Means | `KMeans(n_clusters=6, n_init="auto", random_state=42)` |
| Asignación de clústeres | Etiquetas del ajuste en entrenamiento aplicadas a ambos conjuntos |
| Visualización | 3 scatter plots por conjunto: Lat vs Lon, Lat vs Ingresos, Lon vs Ingresos |
| Validación | Árbol de Decisión entrenado sobre etiquetas de clústeres → 100% de precisión en el conjunto de prueba |

## Resultados

**6 clústeres identificados** — cada uno correspondiente a una región de California geográfica y económicamente distinta:

Los scatter plots (Latitud vs Longitud coloreados por clúster) revelan segmentos geográficos coherentes: zonas costeras de altos ingresos, valles del interior, regiones metropolitanas del sur de California y zonas rurales/agrícolas — todo emergiendo de los datos sin que se proporcione ninguna etiqueta geográfica.

**Validación:** Un `DecisionTreeClassifier` entrenado para predecir las etiquetas de clústeres K-Means a partir de las mismas 3 características alcanza **100% de precisión** en el conjunto de prueba. Esto confirma que los clústeres tienen fronteras limpias y consistentes — no son ruido, son estructura.

## Conclusiones Clave

- **No supervisado ≠ sin validación:** Sin una etiqueta de verdad absoluta, la calidad de los clústeres se evalúa de forma diferente — ¿tienen sentido los segmentos visual/geográficamente? ¿Puede un modelo supervisado reproducirlos perfectamente? Ambas comprobaciones pasan aquí.
- **La elección de características da forma completamente a los clústeres:** Usar las 9 características produciría clústeres impulsados por la antigüedad de las viviendas, el número de habitaciones y la densidad de población. Restringir a ingresos + ubicación produce segmentos que responden a la pregunta práctica: *¿dónde están las áreas de altos vs. bajos ingresos, y cómo se agrupan geográficamente?*
- **K=6 es una decisión de modelado, no un descubrimiento:** El número de clústeres se elige, no se encuentra. Un gráfico de codo o un análisis de silueta proporcionarían una forma fundamentada de seleccionar k en lugar de especificarlo de antemano.

## Stack Tecnológico

`Python` · `scikit-learn` · `pandas` · `Matplotlib` · `Seaborn`

## Ejecutar Localmente

```bash
git clone https://github.com/matthewkane-ml/ML_KMeans_MTK.git
cd ML_KMeans_MTK
pip install -r requirements.txt
jupyter notebook src/KMEANS.ipynb
```

Tanto el modelo K-Means como el Árbol de Decisión de validación se guardan en `models/` mediante `pickle`.

## Próximos Pasos

- Construir un **gráfico de codo** (inercia vs k) y un **gráfico de puntuación de silueta** para encontrar empíricamente el número óptimo de clústeres en lugar de fijar k=6
- Añadir `HouseAge` o `MedHouseVal` como 4ª característica y comparar cómo cambia la estructura de los clústeres
- Visualizar los clústeres en un mapa real usando **Folium** o **Plotly** con los límites de los condados de California para una presentación más interpretable

---

**Autor:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [Portafolio GitHub](https://github.com/matthewkane-ml)

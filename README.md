# Predicción de precios de viviendas en Ames, Iowa

Evaluación Parcial 2, MLY1101 Machine Learning (DuocUC). Caso B: predicción de precios de viviendas.

Integrantes: Nicolas Osses y Javier Sagredo.

## Problema y objetivo

La tasación de una vivienda suele depender del juicio de un perito: es lenta, dos peritos pueden llegar a valores distintos para la misma casa y el criterio rara vez queda documentado. El proyecto busca estimar el precio de venta de una vivienda a partir de sus características físicas, de calidad y de ubicación, como apoyo a la tasación profesional.

Es un problema de regresión supervisada: la variable objetivo es `SalePrice`, el precio efectivamente transado en dólares. El modelo estima el valor de mercado, entendido como el precio en una venta normal entre partes independientes.

En la EP1 se fijó el criterio de salida que el modelo debe cumplir sobre el conjunto de test: R² ≥ 0,85, RMSE ≤ 30.000 USD, MAE ≤ 20.000 USD, MAPE ≤ 12 %, brecha de R² entre train y test ≤ 5 puntos y brecha de MAPE entre el barrio mejor y peor predicho ≤ 8 puntos.

## Dataset

Se usa el Ames, Iowa Housing Dataset, compilado por Dean De Cock (Truman State University, 2011) a partir de los registros de la oficina de tasación de la ciudad de Ames.

| | |
|---|---|
| Registros | 2.930 ventas de viviendas |
| Columnas | 82: 20 continuas, 14 discretas, 23 ordinales, 23 nominales y 2 identificadores (`Order`, `PID`) |
| Período de venta | 2006 a 2010 (`Yr Sold`) |
| Año de construcción | 1872 a 2010 |
| Barrios | 28 |
| `SalePrice` | mediana de 160.000 USD, media de 180.796 USD, mínimo de 12.789 USD y máximo de 755.000 USD |

La EP1 dejó el archivo `data/processed/ames_clean.csv` (2.925 filas y 80 columnas), que es el punto de partida de esta entrega. Sobre los datos originales se hizo lo siguiente:

- Se eliminaron 5 casas con `Gr Liv Area` mayor a 4.000 pies², exclusión que recomienda el propio autor del dataset (tres de ellas eran ventas de casas en construcción con precios muy bajos para su tamaño).
- Los nulos que significan ausencia (sin piscina, sin garaje, sin sótano, etc.) pasaron a la categoría `None`. `Lot Frontage` se imputó con la mediana de su barrio. El resultado tiene 0 nulos.
- Se crearon 4 variables binarias: `Tiene_Piscina`, `Tiene_Sotano`, `Tiene_Garage` y `Tiene_Chimenea`.
- Se eliminaron los identificadores `Order` y `PID` y cuatro variables redundantes: `Garage Area`, `Garage Yr Blt`, `TotRms AbvGrd` y `1st Flr SF`.

## Estructura del repositorio

```
├── data/
│   ├── raw/Ames_Iowa_Housing_Dataset.csv     datos originales
│   └── processed/
│       ├── ames_clean.csv                    salida de la EP1
│       ├── df_ridge_lasso_*.csv              conjuntos preparados para Ridge y Lasso
│       ├── df_random_forest_*.csv            conjuntos preparados para Random Forest
│       ├── df_xgboost_*.csv                  conjuntos preparados para XGBoost
│       └── segmentos.csv                     zona, barrio y condición de venta de cada casa
├── notebooks/
│   └── Build_Model_Housing.ipynb             preparación de datos por modelo
├── requirements.txt
└── README.md
```

Cada modelo tiene tres archivos: `train`, `test` y `test_venta_normal` (el test con la condición de venta normal impuesta; ver más abajo). `segmentos.csv` no entra a los modelos: sirve para reportar el error por barrio o por condición de venta.

## Cómo ejecutar

En Google Colab: abrir `notebooks/Build_Model_Housing.ipynb` desde GitHub y ejecutar todas las celdas. La primera celda después de las importaciones clona el repositorio en `/content` (la rama indicada en la variable `RAMA`) para que las rutas a `data/` funcionen.

En local:

```bash
git clone https://github.com/javisagredo-dev/Housing_Prices_ML_P2.git
cd Housing_Prices_ML_P2
pip install -r requirements.txt
```

y ejecutar el notebook desde la carpeta `notebooks/`. Al terminar, el notebook vuelve a escribir los CSV de `data/processed/`.

El notebook imprime las versiones de Python y de las librerías con que se ejecutó. Con versiones distintas, los resultados de Random Forest y XGBoost cambian en algunos cientos de dólares de RMSE; las decisiones de preparación están fijas en el código para que esas diferencias no cambien los conjuntos de datos.

## Decisiones de preparación

Todas las comparaciones se hicieron con validación cruzada de 5 folds solo sobre train, con el preprocesamiento dentro de un `Pipeline` para que se ajuste de nuevo en cada fold. El test (20 % de los datos) no participa en ningún ajuste. Los modelos se entrenan sobre `log1p(SalePrice)` y las métricas se reportan en dólares.

| | Ridge / Lasso | Random Forest y XGBoost |
|---|---|---|
| Filas eliminadas por IQR o z-score | Ninguna | Ninguna |
| Zonas no residenciales | Excluidas | Excluidas |
| `log1p` en `Lot Area`, `Gr Liv Area`, `Mas Vnr Area`, `Wood Deck SF` y `Open Porch SF` | Sí | No |
| Escalado | `RobustScaler` y `StandardScaler` | No |
| Variables ordinales | Mapeo manual y `StandardScaler` | Mapeo manual |
| Nominales de baja cardinalidad | One-hot (`drop='first'`), categorías con menos de 10 casas agrupadas | One-hot, categorías con menos de 10 casas agrupadas |
| `Neighborhood`, `Exterior 1st`, `Exterior 2nd` | Target encoding y `StandardScaler` | Target encoding |
| `Utilities`, `Street`, `Pool QC`, `Pool Area`, `BsmtFin SF 2` | Eliminadas | Eliminadas |
| `Sale Condition`, `Sale Type` | Se mantienen | Se mantienen |
| Columnas finales | 138 | 158 |

El target encoding usa el `TargetEncoder` de scikit-learn, que en train calcula la codificación de cada fila con los datos de los otros folds (*cross-fitting*), así que no hay fuga del target.

## Hallazgos

Outliers. El IQR y el z-score coinciden en marcar al 7,9 % de las casas como outlier en al menos una variable, pero son propiedades reales. En `Mas Vnr Area` y `Open Porch SF` el IQR exagera porque muchas casas tienen 0; al calcularlo solo sobre los valores distintos de cero, los casos marcados bajan de 200 a 63 y de 156 a 86. Por eso no se eliminan filas y la asimetría se trata con `log1p` en los modelos lineales.

Los árboles no necesitan transformar X. Random Forest y XGBoost dan prácticamente el mismo resultado con y sin `log1p` (diferencias de 0 a 4 USD de RMSE). En Random Forest la codificación de las categóricas tampoco importa: las cuatro opciones probadas quedan a menos de 100 USD entre sí. En XGBoost, one-hot con target encoding baja el RMSE en unos 670 USD respecto de la codificación nativa de categorías.

Zonas no residenciales. Hay 29 casas en zona comercial, industrial o agrícola, 25 de ellas en el barrio `IDOTRR`. Entrenar con ellas empeora las predicciones de las casas residenciales en 3 de los 4 modelos (entre 70 y 190 USD de RMSE), y el modelo tampoco las predice bien: su MAPE queda entre 34 % y 44 %, de 4 a 5 veces el de las residenciales. Se excluyen como criterio de alcance: el modelo tasa viviendas en zonas residenciales, y el resto va a tasación manual. La zonificación se conoce antes de tasar, así que el criterio se puede aplicar a una casa nueva.

Variables casi constantes y redundantes. Quitar `Utilities`, `Street`, `Pool QC`, `Pool Area` y `BsmtFin SF 2` y agrupar las categorías raras baja el RMSE de Ridge en 66 USD y el de Lasso en 221 USD, y no cambia el de los árboles. Los conjuntos pasan de 164 a 138 columnas (Ridge y Lasso) y de 185 a 158 (árboles).

Variables de la venta. `Sale Condition` y `Sale Type` solo se conocen al vender, pero quitarlas empeora a los cuatro modelos (entre 34 y 609 USD de RMSE). Se mantienen como variables de control y, al tasar, se asume una venta normal (`Normal` / `WD`) en lo que no se conoce de antemano. Para medir ese escenario se preparó un segundo test en el que se impone la venta normal; cambian 64 de las 582 casas.

Dónde se equivoca más el modelo. Las ventas `Abnorml`, `Alloca` y `Family` tienen un MAPE de 19,1 %, 17,2 % y 16,5 %, frente a 7,2 % en las ventas `Normal` (Ridge, validación cruzada). Se reportan como limitación: la condición de venta no se conoce antes de vender, así que esas casas no se pueden separar de antemano.

Sesgo por barrio. Con las variables de la venta, Ridge y Lasso sobreestiman en un 2 % las ventas normales de `Edwards`, `IDOTRR` y `SWISU`, los barrios con más ventas anómalas. Queda como insumo para el análisis de sesgos.

Fecha de tasación. Los datos cubren de 2006 a 2010. Para que el modelo no extrapole, al tasar una casa nueva se fija `Yr Sold = 2010`: el modelo tasa a precios de ese período.

## Resultados preliminares

Validación cruzada sobre train (2.314 casas residenciales), con la preparación final e hiperparámetros por defecto:

| Modelo | RMSE (USD) | Desv. entre folds | MAE (USD) | MAPE | R² |
|---|---|---|---|---|---|
| Ridge | 20.159 | 996 | 13.744 | 8,06 % | 0,930 |
| Lasso | 20.199 | 1.032 | 13.761 | 8,08 % | 0,930 |
| XGBoost | 22.351 | 1.350 | 14.310 | 8,39 % | 0,914 |
| Random Forest | 25.149 | 2.483 | 15.792 | 9,21 % | 0,891 |

Con hiperparámetros por defecto, los modelos lineales superan a los de árboles, y los cuatro quedan dentro de los umbrales de R², RMSE, MAE y MAPE fijados en la EP1. Son resultados de validación cruzada, no de test: el criterio de salida se evalúa en el test, después de la búsqueda de hiperparámetros.

## Pendiente

- Búsqueda de hiperparámetros con los pipelines completos sobre los datos sin codificar, para evitar fuga en la validación cruzada.
- Evaluación en test, con el test real y con el de venta normal impuesta, y elección del modelo final según el criterio de salida.
- Error por segmento (barrio, condición de venta) con `segmentos.csv`, brecha de equidad entre barrios e importancia de variables por permutación.
- Definir la ubicación y el formato de exportación de los CSV.

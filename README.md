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
│   └── Housing_Prices_ML_Final.ipynb         preparación de datos, modelado, evaluación en test y elección
├── requirements.txt
└── README.md
```

Cada modelo tiene tres archivos: `train`, `test` y `test_venta_normal` (el test con la condición de venta normal impuesta; ver más abajo). `segmentos.csv` no entra a los modelos: sirve para reportar el error por barrio o por condición de venta.

## Cómo ejecutar

Todo el trabajo está en un solo notebook, `Housing_Prices_ML_Final.ipynb`, en dos partes. La parte 1 (secciones 1 a 18) prepara los datos y escribe los CSV de `data/processed/`. La parte 2 (secciones 19 a 27) lee esos CSV, entrena los modelos, los evalúa en test y elige el final.

En Google Colab: abrir el notebook desde GitHub y ejecutar todas las celdas. La primera celda después de las importaciones clona el repositorio en `/content` (la rama indicada en la variable `RAMA`) para que las rutas a `data/` funcionen.

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

## Modelos y resultados

Se comparan cuatro modelos de tres familias: Ridge y Lasso (regresión lineal regularizada), Random Forest y XGBoost. No hay búsqueda de hiperparámetros; los parámetros iniciales están justificados en la sección 21 del notebook. El criterio de salida y la regla de elección se fijaron en el notebook antes de ver los resultados.

Validación cruzada de 5 folds sobre train (2.314 casas):

| Modelo | RMSE val. (USD) | Desv. entre folds | MAPE val. | R² train | R² val. | Brecha |
|---|---|---|---|---|---|---|
| Ridge | 20.152 | 1.018 | 8,07 % | 0,940 | 0,930 | 1,0 pts |
| Lasso | 20.206 | 1.077 | 8,08 % | 0,939 | 0,930 | 0,9 pts |
| XGBoost | 22.529 | 1.554 | 8,40 % | 0,998 | 0,913 | 8,5 pts |
| Random Forest | 25.136 | 2.518 | 9,25 % | 0,985 | 0,891 | 9,3 pts |

Test (582 casas), comparado con el criterio de salida de la EP1:

| Modelo | R² | RMSE (USD) | MAE (USD) | MAPE | Brecha R² train-test | Brecha MAPE barrios | Cumple los seis |
|---|---|---|---|---|---|---|---|
| Ridge | 0,948 | 19.126 | 12.373 | 6,75 % | −0,9 | 6,1 | Sí |
| Lasso | 0,948 | 19.241 | 12.407 | 6,77 % | −0,9 | 6,1 | Sí |
| XGBoost | 0,934 | 21.660 | 13.579 | 7,39 % | 6,3 | 7,7 | No |
| Random Forest | 0,924 | 23.219 | 14.359 | 7,94 % | 6,1 | 10,0 | No |

Random Forest y XGBoost cumplen los KPI de error, pero no el de sobreajuste: con árboles sin límite de profundidad casi memorizan train. Random Forest tampoco cumple el de equidad. Con la venta normal impuesta, Ridge y Lasso siguen cumpliendo los seis KPI.

El modelo elegido es Lasso. Ridge tiene un RMSE de validación 54 USD menor, pero la diferencia de R² entre los dos es de 0,04 puntos, por debajo de la tolerancia de 1 punto que fijó la EP1, y en ese caso se prefiere el modelo más interpretable. Lasso usa 105 de las 138 columnas.

En lenguaje de negocio: el modelo entrega al perito un valor de partida para una vivienda en zona residencial. En la mitad de las casas de test el error fue menor a 8.659 USD (5,3 % del precio), y en el 77,8 % fue menor a 10 %. Como el modelo es lineal en escala logarítmica, cada estimación se descompone en el aporte de cada variable (por ejemplo, la superficie habitable suma un 8,3 % al valor de una casa), así que el valor se puede justificar ante un cliente o un banco. Las variables que más pesan son la superficie habitable, la calidad general y la superficie del sótano.

Limitaciones del modelo elegido:

- Subestima las casas caras: 8 de las 9 casas de test de más de 450.000 USD quedan bajo su precio real.
- Sobreestima algo las casas del cuartil más barato (+3,4 % en promedio), que es el tramo con mayor error (MAPE de 8,8 %).
- En las ventas anómalas el MAPE es de 11,2 % y en las ventas entre familiares de 15,5 %, frente a 6,3 % en las normales. La condición de venta no se conoce antes de vender.

## Pendiente

- Búsqueda de hiperparámetros con los pipelines completos sobre los datos sin codificar, para evitar fuga en la validación cruzada. Ahí se puede ver si Random Forest y XGBoost cumplen la brecha de R² al limitar la profundidad de los árboles.
- Definir la ubicación y el formato de exportación de los CSV, y si se guarda el modelo entrenado en `models/`.

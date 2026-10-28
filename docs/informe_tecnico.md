# Predicción de precios de viviendas en Ames, Iowa

Informe técnico. Evaluación Parcial 2, MLY1101 Machine Learning (DuocUC). Caso B: predicción de precios de viviendas.

Integrantes: Nicolas Osses y Javier Sagredo.

Repositorio: `https://github.com/javisagredo-dev/Housing_Prices_ML_P2`

## Contenido

1. Descripción del problema de negocio
2. Objetivos del proyecto
3. Fuentes de datos
4. Análisis exploratorio de los datos
5. Preparación y transformación de los datos
6. Selección de modelos
7. Variables finales por modelo
8. Implementación
9. Evaluación y selección del modelo
10. Sesgos y consideraciones éticas
11. Conclusiones, limitaciones y próximos pasos
12. Reproducibilidad
13. Referencias

---

## 1. Descripción del problema de negocio

La tasación de una vivienda depende del juicio de un perito, y ese proceso tiene tres problemas. Es lento, porque cada tasación requiere una visita y un informe manual. Es inconsistente, porque dos peritos pueden llegar a valores distintos para la misma casa. Y deja poca trazabilidad, porque el criterio detrás del valor rara vez queda documentado.

Para corredoras, bancos hipotecarios y plataformas de compraventa, esto tiene costos concretos: casas sobrevaloradas que pasan meses sin venderse, casas subvaloradas que significan una pérdida para el vendedor y créditos hipotecarios respaldados por garantías mal estimadas.

El proyecto estima el precio de venta de una vivienda a partir de sus características físicas, de calidad y de ubicación, como apoyo a la tasación profesional. Es un problema de regresión supervisada: la variable objetivo es `SalePrice`, el precio transado en dólares, y se dispone de ventas históricas con el precio conocido. El modelo estima el valor de mercado, entendido como el precio en una venta normal entre partes independientes.

## 2. Objetivos del proyecto

Objetivo general: construir un modelo que estime el precio de una vivienda residencial con un error acotado y parejo entre barrios, y que permita justificar cada estimación.

Objetivos específicos:

1. Preparar los datos de forma distinta para cada familia de modelos, probando cada decisión con validación cruzada en lugar de asumirla.
2. Entrenar y comparar al menos tres modelos de regresión de familias distintas.
3. Verificar que el modelo elegido cumpla el criterio de salida definido en la EP1, evaluado sobre datos que no participaron en ninguna decisión.
4. Medir dónde se equivoca el modelo (por barrio, condición de venta y tramo de precio) y traducir los resultados a recomendaciones de uso.

Criterio de salida, fijado en la EP1 antes de entrenar. Un modelo es apto solo si cumple los seis indicadores a la vez sobre el conjunto de test:

| KPI | Umbral | Qué controla |
|---|---|---|
| R² | ≥ 0,85 | Qué parte de la variación de los precios explica el modelo |
| RMSE | ≤ 30.000 USD | Error en dólares, castigando más los errores grandes |
| MAE | ≤ 20.000 USD | Error promedio en dólares |
| MAPE | ≤ 12 % | Error relativo al precio de cada casa |
| Brecha de R² entre train y test | ≤ 5 puntos | Sobreajuste |
| Brecha de MAPE entre el barrio mejor y el peor predicho | ≤ 8 puntos | Equidad territorial |

Son indicadores técnicos del modelo. Los indicadores de negocio (tiempo de tasación, cobertura del catálogo, tasa de derivación a tasación manual) requieren un modelo desplegado y quedan fuera del alcance.

Alcance: la búsqueda de hiperparámetros y el despliegue no forman parte de esta entrega.

Metodología. El proyecto sigue CRISP-DM:

| Fase | Qué se hizo | Dónde |
|---|---|---|
| Comprensión del negocio | Problema, objetivos, KPI y criterio de salida | EP1; secciones 1 y 2 |
| Comprensión de los datos | Distribuciones, correlaciones, nulos, outliers, multicolinealidad | EP1; sección 4 |
| Preparación de los datos | Limpieza, transformaciones y codificación por familia de modelos | Notebook, parte 1 (secciones 1 a 18); sección 5 |
| Modelado | Cuatro modelos con parámetros iniciales justificados | Notebook, parte 2 (secciones 19 a 22); secciones 6 a 8 |
| Evaluación | KPI en test, error por segmento, elección e interpretación | Notebook, parte 2 (secciones 23 a 27); sección 9 |
| Despliegue | Fuera del alcance | |

El trabajo volvió varias veces a fases anteriores. El análisis de sesgos de la EP1 llevó a entrenar sobre el logaritmo del precio. En la parte 1 del notebook, el análisis de las casas peor predichas (sección 5 del notebook y 4.5 de este informe) llevó a las pruebas sobre zonas no residenciales y variables de la venta.

## 3. Fuentes de datos

Se usa el Ames, Iowa Housing Dataset, compilado por Dean De Cock (Truman State University, 2011) a partir de los registros de la oficina de tasación de la ciudad de Ames. Es un dataset público de uso educativo.

| | |
|---|---|
| Registros | 2.930 ventas de viviendas |
| Columnas | 82: 20 continuas, 14 discretas, 23 ordinales, 23 nominales y 2 identificadores (`Order`, `PID`) |
| Período de venta | 2006 a 2010 |
| Año de construcción | 1872 a 2010 |
| Barrios | 28 |
| `SalePrice` | mediana de 160.000 USD, media de 180.796 USD, mínimo de 12.789 USD y máximo de 755.000 USD |

Archivos del repositorio:

| Archivo | Contenido |
|---|---|
| `data/raw/Ames_Iowa_Housing_Dataset.csv` | Datos originales |
| `data/processed/ames_clean.csv` | Salida de la EP1: 2.925 filas, 80 columnas, sin nulos |
| `data/processed/df_ridge_lasso_*.csv` | Conjuntos preparados para Ridge y Lasso |
| `data/processed/df_random_forest_*.csv` | Conjuntos preparados para Random Forest |
| `data/processed/df_xgboost_*.csv` | Conjuntos preparados para XGBoost |
| `data/processed/segmentos.csv` | Zona, barrio y condición de venta de cada casa, para reportar el error por segmento |

Cada modelo tiene tres archivos: `train`, `test` y `test_venta_normal` (sección 5.7).

## 4. Análisis exploratorio de los datos

El análisis exploratorio completo está en el informe de la EP1. Esta sección resume los hallazgos que condicionaron las decisiones de esta entrega y agrega los de la parte 1 del notebook.

### 4.1 Variable objetivo

Después de la limpieza de la EP1 (2.925 casas), `SalePrice` tiene una media de 180.412 USD, una mediana de 160.000 USD, una desviación estándar de 78.555 USD y una asimetría de 1,59. La media supera a la mediana porque unas pocas casas caras alargan la cola derecha. Por eso los modelos se entrenan sobre `log1p(SalePrice)`: la escala logarítmica corrige la asimetría y, como mostró la EP1, también reparte el error de forma más pareja entre barrios (sección 10).

### 4.2 Variables más relacionadas con el precio

| Variable | Correlación con `SalePrice` (EP1) |
|---|---|
| `Overall Qual` | 0,799 |
| `Gr Liv Area` | 0,707 |
| `Garage Cars` | 0,648 |
| `Total Bsmt SF` | 0,632 |
| `Year Built` | 0,558 |

El predictor más fuerte es una nota de calidad asignada por el tasador municipal, no una medida física. Es decir, el modelo hereda en parte el criterio humano que busca apoyar.

### 4.3 Calidad de los datos

- 27 columnas tenían nulos, pero casi todos eran estructurales: significan que la casa no tiene la característica (sin piscina, sin garaje, sin sótano). Se validó cruzando cada columna con su compañera numérica; por ejemplo, cuando `Pool QC` es nulo, `Pool Area` vale 0 en todos los casos. Solo `Lot Frontage` tenía datos realmente faltantes (490 casas, 16,7 %).
- No hay filas duplicadas.
- Había cuatro pares de predictores con correlación mayor a 0,8 entre sí (por ejemplo, `Garage Cars` y `Garage Area`, 0,890). La EP1 eliminó uno de cada par.

### 4.4 Outliers

El IQR (fuera de Q1 − 1,5·IQR o Q3 + 1,5·IQR) y el z-score (más de 3 desviaciones de la media) coinciden en marcar 232 casas (7,9 %) en al menos una variable continua. El IQR marca bastantes más casos que el z-score: 202 frente a 28 en `Lot Frontage`.

Antes de decidir, se revisó de qué tipo era cada outlier:

- Artefacto del método. Cuando muchas casas tienen 0 en una variable, Q1 queda en 0 y el IQR mezcla las casas sin la característica con las que sí la tienen. Calculado solo sobre los valores distintos de cero, el conteo baja de 200 a 63 en `Mas Vnr Area` (60,5 % de ceros) y de 156 a 86 en `Open Porch SF`.
- Cola larga natural. `Lot Area` tiene una asimetría de 13,2.
- Casos reales. Los lotes más grandes son ventas normales en terrenos bajos o con pendiente, no errores de registro.

### 4.5 Dónde se concentra el error

Con predicciones fuera de muestra de un Ridge de diagnóstico:

| Grupo | Casas | MAPE |
|---|---|---|
| Ventas `Normal` | 1.927 | 7,2 % |
| Ventas `Partial` (casas nuevas) | 187 | 8,9 % |
| Ventas `Family` | 41 | 16,5 % |
| Ventas `Alloca` | 20 | 17,2 % |
| Ventas `Abnorml` (remates, ventas cortas) | 155 | 19,1 % |
| Zonas residenciales | 2.314 | 8,1 % |
| Zonas no residenciales | 26 | 33,6 % |

Hay 29 casas en zona comercial, industrial o agrícola, con medianas de precio entre 47.300 y 80.313 USD, y 25 de ellas están en el barrio `IDOTRR`. Estos dos hallazgos llevaron a las pruebas de las secciones 5.5 y 5.7.

## 5. Preparación y transformación de los datos

La preparación está en la parte 1 del notebook (secciones 1 a 18). Cada decisión importante se probó con validación cruzada de 5 folds solo sobre train, con un criterio fijado antes de ver el resultado. El test no participó en ninguna decisión.

### 5.1 Limpieza heredada de la EP1

- Se eliminaron 5 casas con más de 4.000 pies² de superficie habitable, como recomienda De Cock: tres eran ventas de casas en construcción con precios muy bajos para su tamaño.
- Los nulos estructurales pasaron a la categoría `None`. `Lot Frontage` se imputó con la mediana de su barrio.
- Se crearon `Tiene_Piscina`, `Tiene_Sotano`, `Tiene_Garage` y `Tiene_Chimenea`.
- Se eliminaron `Order` y `PID` (identificadores) y `Garage Area`, `Garage Yr Blt`, `TotRms AbvGrd` y `1st Flr SF` (redundantes).

### 5.2 Split

`train_test_split` con 20 % para test y semilla 42: 2.340 casas en train y 585 en test. Después de excluir las zonas no residenciales (sección 5.5) quedan 2.314 y 582.

### 5.3 Outliers y `log1p`

No se eliminó ninguna fila: las casas marcadas son reales, y en varias variables el IQR exageraba por los ceros.

La asimetría se trató con `log1p` (log de 1 + x, que admite ceros). Criterio: aplicarlo solo si reduce la asimetría al menos a la mitad sin que la correlación con el precio caiga más de 0,10. Cumplen 5 variables: `Lot Area` (asimetría de 13,2 a −0,5), `Mas Vnr Area`, `Open Porch SF`, `Wood Deck SF` y `Gr Liv Area`. En `Total Bsmt SF` y `Bsmt Unf SF` la asimetría empeoraba; en `2nd Flr SF` y `BsmtFin SF 1` se perdía correlación.

En Ridge y Lasso se compararon cinco variantes:

| Variante | Tratamiento | Resultado |
|---|---|---|
| A | Escalado de la EP1 (`RobustScaler` y `StandardScaler`) | Base |
| B | A + `log1p` en las 5 variables | MAPE unos 0,1 puntos menor; se adopta |
| C | B + recorte de `Lot Frontage` al percentil 99 | RMSE unos 270 USD mayor; se descarta |
| D | C sin zonas no residenciales | No comparable con las demás; se decide aparte (sección 5.5) |
| E | B + `StandardScaler` en las ordinales | Igual que B; se adopta para que la penalización trate a todas las variables en la misma escala |

### 5.4 Qué necesitan los modelos de árboles

- Un árbol solo usa el orden de los valores para cortar, así que `log1p` y el escalado no deberían cambiarlo. Se comprobó: 4 USD de diferencia de RMSE en Random Forest y resultados idénticos en XGBoost. Por la misma razón, los valores extremos de X no los afectan.
- En Random Forest, las cuatro codificaciones de las categóricas que se probaron quedan a menos de 100 USD entre sí, frente a una variación entre folds de unos 2.500 USD.
- En XGBoost, one-hot con target encoding baja el RMSE en unos 670 USD (2,9 %) respecto de la codificación nativa de categorías y es más estable entre folds.

Se usa la misma codificación en los dos: one-hot para las nominales de baja cardinalidad y target encoding para `Neighborhood`, `Exterior 1st` y `Exterior 2nd`.

### 5.5 Zonas no residenciales

Comparar el error con y sin estas casas no sirve, porque al sacarlas también se dejan de evaluar. La prueba fue otra: en cada fold se entrenaron dos modelos, con y sin las no residenciales, y los dos se evaluaron sobre las mismas casas residenciales de validación.

Criterio fijado: las no residenciales empeoran a un modelo si el RMSE sobre residenciales sube en promedio y en al menos 4 de 5 folds; se excluyen si eso pasa en la mayoría de los modelos.

Resultado: empeoran a Ridge, Lasso y XGBoost (entre 70 y 190 USD de RMSE), y el modelo tampoco las predice bien (MAPE entre 34 % y 44 %). Se excluyen como criterio de alcance: el modelo tasa viviendas en zonas residenciales y el resto va a tasación manual. Es un criterio aplicable a una casa nueva, porque la zonificación se conoce antes de tasar.

### 5.6 Variables casi constantes y redundantes

Se probó un paquete de limpieza: quitar `Utilities` y `Street` (casi todas las casas tienen el mismo valor), `Pool QC` y `Pool Area` (solo 11 piscinas; queda `Tiene_Piscina`) y `BsmtFin SF 2` (correlación de 0,008 con el precio y 88 % de ceros), y agrupar en una categoría de infrecuentes las categorías con menos de 10 casas en train.

Criterio fijado: se adopta salvo que empeore el RMSE en promedio y en al menos 4 de 5 folds. Resultado: Ridge mejora 66 USD, Lasso 221 USD y los árboles quedan igual. Se adopta en los tres conjuntos.

### 5.7 Variables que solo se conocen al vender

`Sale Condition` y `Sale Type` se conocen al vender, no al tasar. Se mantuvieron como variables de control: quitarlas empeora a los cuatro modelos (el RMSE sube 505 USD en Ridge, 481 en Lasso, 609 en XGBoost y 34 en Random Forest). Además, sin ellas el descuento de los remates se traslada al barrio.

Al tasar una casa nueva se asume una venta normal (`Normal` / `WD`) en lo que no se conoce de antemano. Las casas nuevas (`Partial` / `New`) sí se conocen antes y conservan su valor. Para medir ese escenario se preparó un segundo test con la venta normal impuesta: cambian 64 de las 582 casas.

También se fija `Yr Sold = 2010` al tasar, para que el modelo no extrapole fuera del período de los datos.

### 5.8 Resumen

| | Ridge / Lasso | Random Forest / XGBoost |
|---|---|---|
| Filas eliminadas por outlier | Ninguna | Ninguna |
| Zonas no residenciales | Excluidas | Excluidas |
| `log1p` en 5 variables de X | Sí | No |
| Escalado | `RobustScaler` y `StandardScaler` | No |
| Ordinales | Mapeo manual + `StandardScaler` | Mapeo manual |
| Nominales de baja cardinalidad | One-hot (`drop='first'`, `min_frequency=10`) | One-hot (`min_frequency=10`) |
| `Neighborhood`, `Exterior 1st`, `Exterior 2nd` | Target encoding + `StandardScaler` | Target encoding |
| Columnas eliminadas | `Utilities`, `Street`, `Pool QC`, `Pool Area`, `BsmtFin SF 2` | Igual |
| Variables de la venta | Se mantienen | Se mantienen |
| Target | `log1p(SalePrice)` al entrenar | Igual |

Todo lo que aprende de los datos (escalado y target encoding) se ajustó solo con train. El target encoding usa el `TargetEncoder` de scikit-learn, que calcula la codificación de cada casa de train sin su propio precio (cross-fitting), así que no hay fuga del target.

## 6. Selección de modelos

Se comparan cuatro modelos de las tres familias que preseleccionó la EP1, de menor a mayor complejidad.

| Familia | Modelo | Por qué se incluye | Limitación |
|---|---|---|---|
| Regresión lineal regularizada | Ridge | Línea base interpretable: cada estimación se descompone en el aporte de cada variable. La penalización L2 controla los coeficientes con variables correlacionadas | Supone relaciones lineales (en escala log) |
| Regresión lineal regularizada | Lasso | Igual que Ridge, pero la penalización L1 puede dejar coeficientes en 0: selecciona variables y deja un modelo más simple | Entre variables muy correlacionadas tiende a quedarse con una sola |
| Bosques aleatorios | Random Forest | Capta relaciones no lineales e interacciones sin especificarlas; no le afecta la escala ni los valores extremos de X | Menos interpretable; con árboles profundos puede sobreajustar |
| Gradient boosting | XGBoost | El algoritmo con mejor desempeño documentado en datos tabulares; cada árbol corrige el error de los anteriores | Muchos hiperparámetros y menos transparencia |

Si los modelos lineales rinden parecido a los de árboles, se prefieren, porque el problema de negocio pide poder justificar el valor ante un cliente o un banco.

## 7. Variables finales por modelo

Los dos conjuntos parten de las mismas 74 variables originales: las 79 predictoras de `ames_clean.csv` menos las 5 eliminadas en la sección 5.6. Cambia la forma de codificarlas.

| Grupo | Variables | Ridge / Lasso | Random Forest / XGBoost |
|---|---|---|---|
| Superficies con `log1p` | `Lot Area`, `Mas Vnr Area`, `Open Porch SF`, `Wood Deck SF`, `Gr Liv Area` | 5 columnas, `log1p` + `StandardScaler` | 5 columnas, sin transformar |
| Otras superficies | `Lot Frontage`, `BsmtFin SF 1`, `Bsmt Unf SF`, `Total Bsmt SF`, `2nd Flr SF` | 5 columnas, `RobustScaler` | 5 columnas, sin transformar |
| Otras numéricas | Años (`Year Built`, `Year Remod/Add`, `Yr Sold`, `Mo Sold`), notas generales (`Overall Qual`, `Overall Cond`), baños, dormitorios, cocinas, chimeneas, `Garage Cars`, porches, `Low Qual Fin SF`, `Misc Val` | 19 columnas, `StandardScaler` | 19 columnas, sin transformar |
| Ordinales | 18 variables de calidad, condición, exposición, terminación, funcionalidad, forma y pendiente del lote | 18 columnas, mapeo manual + `StandardScaler` | 18 columnas, mapeo manual |
| Nominales de alta cardinalidad | `Neighborhood`, `Exterior 1st`, `Exterior 2nd` | 3 columnas, target encoding + `StandardScaler` | 3 columnas, target encoding |
| Nominales de baja cardinalidad | 20 variables, entre ellas `MS SubClass`, `MS Zoning`, `House Style`, `Foundation`, `Garage Type`, `Sale Type` y `Sale Condition` | 84 dummies (sin la primera categoría de cada variable) | 104 dummies |
| Binarias | `Tiene_Piscina`, `Tiene_Sotano`, `Tiene_Garage`, `Tiene_Chimenea` | 4 columnas | 4 columnas |
| Total | 74 variables | 138 columnas | 158 columnas |

En los lineales se elimina la primera dummy de cada variable (`drop='first'`) porque, con intercepto, las dummies de una misma variable suman siempre 1 y generan colinealidad perfecta. Los árboles no tienen ese problema. Random Forest y XGBoost usan el mismo conjunto, guardado en dos archivos para poder cambiar la codificación de uno sin tocar el otro.

Del ajuste final, Lasso dejó en 0 los coeficientes de 33 columnas y usa 105 de las 138. Casi todas son dummies de variables nominales, además de `Tiene_Piscina` y `BsmtFin Type 2`.

## 8. Implementación

### 8.1 Herramientas

Python con pandas y numpy para los datos, scikit-learn para la preparación, los modelos lineales, Random Forest y la validación, y XGBoost para el boosting. Los resultados citados corresponden a una ejecución en Google Colab con Python 3.13, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1 y xgboost 3.4.1.

### 8.2 Estructura

Todo el trabajo está en un solo notebook, `notebooks/Housing_Prices_ML_Final.ipynb`, dividido en dos partes:

- La parte 1 (secciones 1 a 18) arma la preparación como un `Pipeline` de scikit-learn con un `ColumnTransformer`, que aplica a cada grupo de columnas su transformación. Así, en la validación cruzada la preparación se ajusta de nuevo en cada fold. Al final guarda los conjuntos ya transformados en `data/processed/`.
- La parte 2 (secciones 19 a 27) lee esos conjuntos desde los CSV, entrena los cuatro modelos, los evalúa y elige el final.

Los cuatro modelos se envuelven en `TransformedTargetRegressor`, que entrena sobre `log1p(SalePrice)` y devuelve las predicciones en dólares con `expm1`. Así todas las métricas quedan en dólares.

### 8.3 Parámetros iniciales

No hay búsqueda de hiperparámetros. Se usan los mismos parámetros con que la parte 1 tomó sus decisiones de preparación, para que los resultados de las dos partes sean comparables.

| Modelo | Parámetro | Valor | Justificación |
|---|---|---|---|
| Ridge | `alphas` de `RidgeCV` | 30 valores entre 0,01 y 1.000 (escala log) | La penalización es parte del ajuste de un modelo regularizado; `RidgeCV` la elige con leave-one-out usando solo train. Eligió 8,53, dentro del rango |
| Lasso | `alphas` de `LassoCV` | 30 valores entre 0,00001 y 0,1 | Mismo criterio, en la escala de Lasso: con alpha = 0,1 deja solo 5 de 138 coeficientes. Elige con 5 folds dentro de train; eligió 0,000174, dentro del rango |
| Lasso | `max_iter` | 50.000 | Margen para que el algoritmo converja con todo el rango; con estos datos también converge con el valor por defecto |
| Random Forest | `n_estimators` | 300 | Agregar árboles no produce sobreajuste, solo estabiliza la predicción y aumenta el tiempo de cómputo (Breiman, 2001) |
| Random Forest | resto | Por defecto | Árboles sin límite de profundidad, cada uno sobre una muestra con reposición de train, con todas las variables candidatas en cada corte |
| XGBoost | `n_estimators`, `learning_rate` | 500 y 0,05 | Una tasa baja con más árboles suele generalizar mejor que la tasa por defecto (0,3) con pocos árboles (Friedman, 2001) |
| XGBoost | `tree_method` | `hist` | Método por defecto desde XGBoost 2.0 y el más rápido |
| XGBoost | resto | Por defecto | Profundidad máxima 6, sin submuestreo de filas ni columnas |
| Todos | `random_state` | 42 | Reproducibilidad |

### 8.4 Validación

Validación cruzada de 5 folds (`KFold` mezclado, semilla 42) sobre las 2.314 casas de train, con las mismas particiones en todo el proyecto. En cada fold se mide el error en validación y en las casas de entrenamiento, para detectar sobreajuste.

Los conjuntos guardados ya vienen escalados y codificados con todo train, así que en esta validación cruzada cada fold usa una preparación que vio también sus casas de validación. Se midió el efecto: los RMSE quedan a menos de 200 USD de los que se obtienen con los pipelines completos. El test no tiene este problema, porque la preparación se ajustó solo con train.

## 9. Evaluación y selección del modelo

### 9.1 Regla de elección

Fijada antes de entrenar:

1. Solo compiten los modelos que cumplen los seis KPI en el test real.
2. Entre ellos se prefiere el de menor RMSE en validación cruzada (la métrica principal de la EP1). Se elige por validación y no por test, para que el test quede como una medición independiente.
3. Si otro modelo que cumple queda a menos de 1 punto de R² de validación, se elige el más interpretable: primero los lineales y, entre ellos, el que usa menos columnas.
4. Si ningún modelo cumple, no se bajan los umbrales: se vuelve a una fase anterior.

### 9.2 Validación cruzada

| Modelo | RMSE train | RMSE val. | Desv. entre folds | MAPE val. | R² train | R² val. | Brecha |
|---|---|---|---|---|---|---|---|
| Ridge | 18.831 | 20.152 | 1.018 | 8,07 % | 0,940 | 0,930 | 1,0 |
| Lasso | 18.919 | 20.206 | 1.077 | 8,08 % | 0,939 | 0,930 | 0,9 |
| XGBoost | 3.598 | 22.529 | 1.554 | 8,40 % | 0,998 | 0,913 | 8,5 |
| Random Forest | 9.451 | 25.136 | 2.518 | 9,25 % | 0,985 | 0,891 | 9,3 |

Los lineales rinden casi igual en las casas que vieron y en las nuevas. Random Forest y XGBoost casi memorizan train y en validación quedan peor que los lineales. Es la consecuencia de usar árboles sin límite de profundidad; limitarlos sería búsqueda de hiperparámetros.

### 9.3 Test

Cada modelo se entrenó con las 2.314 casas de train y se midió sobre las 582 de test.

| Modelo | R² | RMSE | MAE | MAPE | Brecha train-test | Brecha barrios | Cumple los seis |
|---|---|---|---|---|---|---|---|
| Ridge | 0,948 | 19.126 | 12.373 | 6,75 % | −0,9 | 6,1 | Sí |
| Lasso | 0,948 | 19.241 | 12.407 | 6,77 % | −0,9 | 6,1 | Sí |
| XGBoost | 0,934 | 21.660 | 13.579 | 7,39 % | 6,3 | 7,7 | No |
| Random Forest | 0,924 | 23.219 | 14.359 | 7,94 % | 6,1 | 10,0 | No |

- Los cuatro cumplen los KPI de error. Random Forest y XGBoost no cumplen el de sobreajuste, y Random Forest tampoco el de equidad: su peor barrio es `IDOTRR`, con 14,8 % de MAPE.
- En los lineales la brecha es negativa: rindieron algo mejor en test que en train. Su RMSE de test queda unos 1.000 USD bajo el de validación, del mismo orden que la variación entre folds. Estas 582 casas resultaron algo más fáciles que el promedio.
- La brecha de equidad considera los 17 barrios con al menos 10 casas en test. En Lasso, el barrio peor predicho es `ClearCr` (10,3 %) y el mejor `CollgCr` (4,1 %).
- Con la venta normal impuesta, Ridge y Lasso siguen cumpliendo los seis KPI (Lasso: R² 0,949, MAPE 6,93 %, brecha entre barrios 6,2). XGBoost además falla el de equidad (8,9). En las 64 casas que cambian, Lasso pasa a sobreestimar un 5,8 % en promedio: estima el valor de mercado, y la mayoría de esas casas son remates o ventas entre familiares, que suelen cerrarse bajo ese valor.

### 9.4 Error por segmento (Lasso, test real)

| Segmento | MAPE | Error con signo |
|---|---|---|
| Ventas `Normal` (483 casas) | 6,3 % | |
| Ventas `Abnorml` (33 casas) | 11,2 % | |
| Ventas `Family` (5 casas) | 15,5 % | |
| Precio hasta 130.625 USD | 8,8 % | +3,4 % |
| 130.625 a 160.000 USD | 5,8 % | +0,2 % |
| 160.000 a 215.000 USD | 5,8 % | +1,2 % |
| Más de 215.000 USD | 6,6 % | −2,5 % |

El error con signo positivo indica que el modelo sobreestima. Las estimaciones se cargan hacia el centro: sobreestima las casas baratas y subestima las caras. En Random Forest el efecto es cerca del doble (+6,7 % y −4,9 %).

### 9.5 Selección

1. Cumplen los seis KPI: Ridge y Lasso.
2. Menor RMSE de validación: Ridge, con 20.152 USD.
3. Lasso queda a 0,04 puntos de R² de validación (0,9295 frente a 0,9299), bajo la tolerancia de 1 punto, y usa 105 columnas en vez de 138.

El modelo elegido es Lasso. La diferencia con Ridge (54 USD de RMSE en validación y 115 USD en test) es mucho menor que la variación entre folds, de unos 1.000 USD. Con una precisión equivalente, Lasso se explica con menos variables.

### 9.6 Interpretación para el negocio

El modelo entrega al perito un valor de partida para una vivienda en zona residencial, junto con el detalle de qué variables movieron ese valor.

Error esperado, según el test:

- En la mitad de las casas, el error es menor a 8.659 USD (5,3 % del precio).
- El 77,8 % de las casas queda con un error menor a 10 % y el 96,6 % con uno menor a 20 %.
- Solo 4 de las 582 casas superan el 30 %, y en las cuatro el modelo sobreestimó el precio.

Variables que más pesan, según la importancia por permutación (cuánto sube el RMSE en test si se desordena la variable):

| Variable | Aumento del RMSE |
|---|---|
| `Gr Liv Area` (superficie habitable) | 21.926 USD |
| `Overall Qual` (calidad general) | 13.710 USD |
| `Total Bsmt SF` (superficie del sótano) | 12.753 USD |
| `Year Built` | 5.003 USD |
| `Neighborhood` | 3.854 USD |
| `Lot Area` | 3.744 USD |

Son tamaño, calidad, antigüedad y ubicación, los mismos factores que considera cualquier tasación.

Trazabilidad. En escala logarítmica, Lasso es una suma, así que cada estimación se separa en el aporte de cada variable respecto de una casa con los valores promedio de train (167.304 USD). En una casa de `Mitchel` vendida en 160.000 USD, la superficie habitable suma 8,3 % y la condición general 6,2 %, mientras que la funcionalidad resta 7,9 % y el año de construcción 4,6 %; el modelo estima 177.830 USD. Con este desglose el perito puede justificar el valor y revisar si alguna variable está mal registrada.

Derivación a tasación manual. Las zonas no residenciales quedan fuera de alcance. La EP1 también proponía derivar los barrios con menos de 30 ventas; en este test sus 17 casas tuvieron un MAPE de 4,7 %, menor que el 6,8 % del resto. Son pocas casas para concluir, así que la regla puede mantenerse como precaución.

## 10. Sesgos y consideraciones éticas

### 10.1 Sesgos de los datos (EP1)

| Sesgo | Evidencia | Medida adoptada |
|---|---|---|
| Representación | 7 de 28 barrios con menos de 30 casas; `Landmrk` tiene una | Derivación a tasación manual en barrios sub-representados |
| Histórico o geográfico | Razón de 3,6 a 1 en precio entre barrios; 2,4 a 1 controlando por superficie | Equidad territorial como KPI del criterio de salida |
| Proxy geográfico | Las variables físicas permiten reconstruir el barrio (V de Cramér 0,51) | Conservar `Neighborhood` para poder auditar |
| Medición | `Overall Qual` es una nota subjetiva; solo el 64,8 % se explica por variables físicas | Documentar |
| Temporal | Datos de 2006 a 2010 | El modelo tasa a precios de ese período |
| Selección | 18 % de las ventas no son de mercado libre | Alcance limitado al valor de mercado en venta normal |
| Asimetría del error | Brecha de 16,9 puntos entre barrios con el precio en dólares y 8,4 en escala log | Entrenar sobre `log1p(SalePrice)` |

Quitar `Neighborhood` no elimina el sesgo geográfico, porque las demás variables reconstruyen el barrio, y además impide auditarlo. Entrenar en escala logarítmica funciona como medida de equidad: minimizar el error en log equivale aproximadamente a minimizar el error relativo, y el modelo deja de favorecer a las casas caras.

### 10.2 Resultado de la auditoría en esta entrega

El modelo elegido cumple el umbral de equidad: 6,1 puntos de diferencia de MAPE entre el barrio mejor y el peor predicho, frente a un máximo de 8. Random Forest no lo cumple (10,0 puntos). Quedan dos sesgos medidos: el modelo sobreestima en 3,4 % las casas del cuartil más barato, y los modelos lineales sobreestiman en un 2 % las ventas normales de `Edwards`, `IDOTRR` y `SWISU`, los barrios con más ventas anómalas.

La exclusión de las zonas no residenciales saca 25 de las 93 casas de `IDOTRR`. Las 68 residenciales siguen en el modelo, y `segmentos.csv` permite reportar su error por separado.

### 10.3 Ética y privacidad

- El dataset no tiene nombres ni direcciones. `PID` permitía llegar a la dirección y al propietario a través de registros públicos y se eliminó. Queda un riesgo de reidentificación en barrios con muy pocas casas.
- Quien es tasado por un algoritmo debería poder saber qué factores determinaron el valor. Por eso la regla de elección favorece al modelo interpretable cuando la diferencia de precisión es pequeña.
- El modelo apoya la tasación, no la reemplaza. No debe usarse para decidir solo un crédito hipotecario.
- Los dos errores no cuestan lo mismo: subvalorar perjudica al vendedor y sobrevalorar al comprador y al banco. La sobreestimación del segmento más barato se reporta como limitación por esa razón.

## 11. Conclusiones, limitaciones y próximos pasos

El modelo elegido es Lasso. Cumple los seis KPI del criterio de salida en el test real y con la venta normal impuesta:

| KPI | Umbral | Test real | Venta normal impuesta |
|---|---|---|---|
| R² | ≥ 0,85 | 0,948 | 0,949 |
| RMSE | ≤ 30.000 USD | 19.241 | 19.084 |
| MAE | ≤ 20.000 USD | 12.407 | 12.462 |
| MAPE | ≤ 12 % | 6,77 % | 6,93 % |
| Brecha de R² train-test | ≤ 5 puntos | −0,9 | −1,0 |
| Brecha de MAPE entre barrios | ≤ 8 puntos | 6,1 | 6,2 |

Frente a Random Forest y XGBoost, Lasso se equivoca menos (unos 4.000 y 2.400 USD menos de RMSE en test) y no sobreajusta. Frente a Ridge, tiene la misma precisión con 33 columnas menos. Y cada estimación se puede descomponer y justificar, que es lo que pide el problema de negocio.

Uso recomendado: valor de partida para el perito en viviendas de zonas residenciales, tasadas como venta normal y con `Yr Sold = 2010`.

Limitaciones:

- Subestima las casas caras: 8 de las 9 casas de test de más de 450.000 USD quedan bajo su precio real.
- Sobreestima algo las casas del cuartil más barato (+3,4 %), que es además el tramo con mayor error.
- En las ventas anómalas el MAPE es de 11,2 % y en las ventas entre familiares de 15,5 %. Esas ventas no se pueden identificar antes de vender.
- Las zonas no residenciales quedan fuera del alcance.
- Los datos son de 2006 a 2010: el modelo tasa a precios de ese período.

Próximos pasos:

- Búsqueda de hiperparámetros con los pipelines completos sobre los datos sin codificar, para evitar la fuga de la validación cruzada sobre los conjuntos guardados. Ahí se puede ver si Random Forest y XGBoost cumplen la brecha de sobreajuste al limitar la profundidad de los árboles.
- Definir la política de derivación a tasación manual y el formato de exportación de los datos y del modelo.

## 12. Reproducibilidad

Estructura del repositorio:

```
├── data/
│   ├── raw/Ames_Iowa_Housing_Dataset.csv
│   └── processed/                ames_clean.csv, conjuntos por modelo y segmentos.csv
├── notebooks/
│   └── Housing_Prices_ML_Final.ipynb  preparación de datos, modelado, evaluación y elección
├── requirements.txt
└── README.md
```

Ejecución:

- Google Colab: abrir el notebook desde GitHub y ejecutar todas las celdas. La celda de entorno clona el repositorio en `/content` para que las rutas a `data/` funcionen.
- Local: clonar el repositorio, instalar las dependencias con `pip install -r requirements.txt` y ejecutar el notebook desde `notebooks/`.

La parte 2 lee los conjuntos desde los CSV de `data/processed/` que escribe la sección 17, igual que los leería un proceso que no ejecutó la parte 1. Esos CSV ya están en el repositorio; al ejecutar el notebook completo se vuelven a escribir los mismos archivos.

Todos los procesos aleatorios usan la semilla 42. Las decisiones de preparación quedan fijas en el código; las pruebas se recalculan en cada ejecución y muestran un aviso si en otro entorno dan un resultado distinto. Con otras versiones de las librerías, Random Forest y XGBoost pueden variar en algunos cientos de dólares de RMSE.

## 13. Referencias

- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. Journal of Statistics Education, 19(3).
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. The Annals of Statistics, 29(5), 1189-1232.
- Chapman, P. et al. (2000). CRISP-DM 1.0: Step-by-step data mining guide.
- Informe de la EP1 del proyecto (`Parcial_1/Informe.md`).

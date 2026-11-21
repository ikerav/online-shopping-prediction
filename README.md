# NEXUS · Conversion Intelligence

**Predicción explicable de intención de compra en comercio electrónico**

Proyecto de Machine Learning para analizar el comportamiento de navegación de los usuarios, estimar su probabilidad de compra y priorizar sesiones mediante una aplicación interactiva desarrollada con Streamlit.

## Objetivo

El proyecto busca identificar patrones asociados a la conversión y desarrollar una herramienta que permita:

- Estimar la probabilidad de compra de una sesión.
- Explicar las predicciones mediante SHAP.
- Analizar y priorizar múltiples sesiones a partir de archivos CSV.
- Explorar las posibilidades y limitaciones de la aplicación empresarial de los modelos predictivos.

## Aplicación NEXUS

La aplicación ofrece dos herramientas principales.

**Análisis individual:** permite introducir las 17 características de una sesión, obtener su probabilidad estimada de compra y visualizar las contribuciones SHAP que explican la predicción.

**Priorización de sesiones:** permite cargar un CSV, calcular las probabilidades de todas las sesiones, ordenarlas por puntuación, visualizar su distribución y descargar el ranking completo.

La aplicación utiliza un Random Forest previamente entrenado. No necesita volver a entrenar el modelo al iniciarse.

## Metodología

Se utiliza el conjunto de datos *Online Shoppers Purchasing Intention* del UCI Machine Learning Repository, con 12.330 registros originales.

El desarrollo se organiza en cinco notebooks:

1. **Análisis exploratorio:** limpieza, distribución de variables y relaciones con la compra.
2. **Comparación de modelos:** regresión logística, Random Forest, MLP y CatBoost, con optimización mediante validación cruzada.
3. **Experimentos adicionales:** ingeniería de características, tratamiento del desbalance, análisis de PageValues y ajustes adicionales.
4. **Interpretabilidad:** análisis SHAP, factores asociados a la conversión y explicación de sesiones individuales, principalmente con CatBoost.
5. **Evaluación final:** comparación independiente de Random Forest y CatBoost, selección de umbrales y análisis de priorización.

Tras eliminar duplicados exactos, se reservaron 2.441 sesiones como conjunto de prueba. El entrenamiento y la optimización se realizaron sin utilizar ese conjunto.

## Modelo definitivo

Se seleccionó **Random Forest** por su equilibrio entre rendimiento predictivo, simplicidad e integración.

| Métrica en test | Resultado |
|---|---:|
| Average Precision | 0,7564 |
| ROC AUC | 0,9356 |
| Precision | 0,6626 |
| Recall | 0,7147 |
| F1 | 0,6877 |
| Lift@10 % | 5,16× |

El modelo utiliza las 17 características originales y un umbral de clasificación de **0,365**, seleccionado previamente mediante validación cruzada.

En el conjunto de prueba, el 10 % de las sesiones mejor puntuadas concentra el 51,8 % de los compradores reales.

## Estructura principal

- `app.py`: interfaz de predicción individual y explicaciones SHAP.
- `interfaz_priorizacion.py`: carga, validación y priorización de archivos CSV.
- `assets/styles.css`: identidad visual de NEXUS.
- `.streamlit/config.toml`: configuración del tema.
- `models/random_forest_final.joblib`: pipeline entrenado y umbral definitivo.
- `notebooks/`: análisis, experimentación, interpretabilidad y evaluación.
- `data/`: conjunto de datos utilizado en el proyecto.

## Instalación y ejecución

Se recomienda utilizar Python 3.14 y un entorno virtual.

```bash
git clone https://github.com/ikerav/online-shopping-prediction.git
cd online-shopping-prediction

python -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m streamlit run app.py
```

En Windows, la activación del entorno puede realizarse con `.venv\Scripts\activate`.

La aplicación se abrirá en la dirección local indicada por Streamlit.

Para la priorización masiva, el CSV debe contener las 17 variables de entrada del modelo. La columna `Revenue` no es necesaria.

## Limitaciones

El rendimiento predictivo depende considerablemente de `PageValues`. Antes de utilizar el modelo durante una visita real, es necesario verificar que todas las características estén disponibles en el momento de la predicción.

Las explicaciones SHAP describen el comportamiento del modelo, pero no demuestran relaciones causales.

Por último, identificar sesiones con alta probabilidad de compra no implica necesariamente que una intervención comercial vaya a generar compras adicionales.

**NEXUS es un prototipo académico funcional, no un sistema validado para su despliegue comercial.**
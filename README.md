# NEXUS — Conversion Intelligence

Proyecto de Inteligencia Artificial en las Organizaciones (UC3M).

Predicción de la intención de compra en tiendas online mediante Machine Learning, con especial atención a la interpretabilidad de los modelos y su utilidad empresarial.

## Contenido

- `01`: Análisis exploratorio de los datos.
- `02`: Comparación y optimización de modelos.
- `03`: Ingeniería de características y experimentos.
- `04`: Interpretabilidad mediante SHAP.
- `05`: Evaluación y selección del modelo final.
- `06`: Aproximación a PageValues mediante Google Analytics 4.

## Resultados

El modelo seleccionado es Random Forest, con un Average Precision de 0.7564, un ROC AUC de 0.9356 y un F1-score de 0.6877 en el conjunto de prueba.

## Aplicación

NEXUS permite analizar sesiones individuales, explicar sus predicciones mediante SHAP y priorizar múltiples sesiones a partir de archivos CSV.

## Instalación y ejecución

Recomendado: Python 3.14 y un entorno virtual.

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

El modelo ya está entrenado. No es necesario ejecutar los notebooks para utilizar la aplicación.

## Pruebas

```bash
python -m pytest tests/ -v
```

## Datos

- [Online Shoppers Purchasing Intention — UCI](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- [Google Analytics 4 — BigQuery](https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset)
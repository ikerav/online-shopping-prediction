
# NEXUS — Conversion Intelligence

Proyecto de Inteligencia Artificial en las Organizaciones (UC3M, 2026/2027).

Predicción de la intención de compra en tiendas online mediante Machine Learning. El proyecto compara diferentes modelos, analiza su interpretabilidad y explora su utilidad empresarial mediante una aplicación interactiva de apoyo a la toma de decisiones.

## Contenido

Los notebooks recogen las distintas fases del proyecto:

- `01`: Análisis exploratorio de los datos.
- `02`: Comparación y optimización de modelos.
- `03`: Ingeniería de características y experimentos.
- `04`: Interpretabilidad mediante SHAP.
- `05`: Evaluación y selección del modelo final.
- `06`: Aproximación a PageValues mediante Google Analytics 4.

## Resultados

El modelo seleccionado es Random Forest, con un Average Precision de 0.7564, un ROC AUC de 0.9356 y un F1-score de 0.6877 en el conjunto de prueba.

## Instalación y ejecución

Se recomienda utilizar Python 3.14.

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/ikerav/online-shopping-prediction.git
   cd online-shopping-prediction
   ```

2. Crear un entorno virtual:

   ```bash
   python -m venv .venv
   ```

3. Activar el entorno virtual:

   Linux y macOS:

   ```bash
   source .venv/bin/activate
   ```

   Windows (PowerShell):

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Instalar las dependencias:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Ejecutar la aplicación:

   ```bash
   python -m streamlit run app.py
   ```

La aplicación se abrirá en el navegador (http://localhost:8501).

El modelo entrenado y los conjuntos de datos están incluidos en el repositorio. No es necesario ejecutar los notebooks ni descargar datos adicionales para utilizar la aplicación.

## Aplicación

NEXUS dispone de dos funcionalidades principales:

- **Analizar sesión:** estimar la intención de compra de una sesión individual y explicar la predicción mediante SHAP. Se incluyen ejemplos preconfigurados.
- **Priorizar sesiones:** analizar múltiples sesiones mediante un archivo CSV y generar un ranking según su probabilidad estimada de compra. Puede probarse con `data/raw/online_shoppers_intention.csv`.

## Reproducción de los experimentos

Los notebooks se encuentran en `notebooks/` y pueden ejecutarse en orden utilizando el entorno virtual creado anteriormente como kernel.

El notebook 05 genera el modelo utilizado por la aplicación. El notebook 06 utiliza los datos de GA4 ya extraídos; la consulta SQL original está disponible en `queries/`.

## Pruebas

```bash
python -m pytest tests/ -v
```

## Fuentes de datos

- [Online Shoppers Purchasing Intention — UCI Machine Learning Repository](https://doi.org/10.24432/C5F88Q). Dataset principal del proyecto.
- [Google Analytics 4 — BigQuery Sample Dataset](https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset). Utilizado para estudiar una aproximación alternativa a PageValues.


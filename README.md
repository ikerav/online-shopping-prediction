
# Predicción de compras en tiendas online

Proyecto de la asignatura Inteligencia Artificial en las Organizaciones (UC3M, 2026/2027).

## Objetivo

Predecir la intención de compra de los visitantes de una tienda online a partir de su comportamiento de navegación.

Se compararán tres modelos:
- Regresión logística
- Random Forest
- Red neuronal multicapa (MLP)

También se analizará la importancia de las variables y se utilizará SHAP para interpretar las predicciones.

## Dataset

Online Shoppers Purchasing Intention Dataset (UCI).

- 12.330 sesiones
- 17 variables predictoras
- Variable objetivo: Revenue

Fuente:
https://doi.org/10.24432/C5F88Q

Licencia del dataset: CC BY 4.0.
Autores: Sakar, C. y Kastro, Y. (2018).

## Tecnologías

Python, pandas, scikit-learn, SHAP y Streamlit.

## Instalación

Se recomienda utilizar Python 3.13.

Crear el entorno virtual:

python3.13 -m venv .venv

Activarlo en Linux:

source .venv/bin/activate

Instalar dependencias:

python -m pip install -r requirements.txt

## Ejecución

python -m streamlit run app.py


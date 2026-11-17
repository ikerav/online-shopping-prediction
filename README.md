# Predicción de compras en tiendas online

Proyecto de la asignatura Inteligencia Artificial en las Organizaciones, de la UC3M, curso 2026/2027.

## Objetivo del proyecto

Queremos predecir si una visita a una tienda online terminará en una compra a partir de los datos de navegación.

Compararemos tres modelos:

- Regresión logística.
- Random Forest.
- Red neuronal multicapa (MLP).

Después utilizaremos la importancia de las variables y SHAP para explicar las predicciones.

También desarrollaremos una aplicación con Streamlit para consultar la probabilidad de compra y ordenar varias sesiones según esa probabilidad.

## Qué está hecho

Por ahora, tenemos una aplicación básica que:

- Lee el archivo de datos.
- Muestra el número de sesiones.
- Muestra las primeras cinco filas.

Hemos comprobado que esta aplicación funciona en macOS con Python 3.12.7. También hemos instalado las bibliotecas y comprobado sus dependencias con `pip check`, sin conflictos detectados.

## Qué falta

- Revisar y preparar los datos.
- Entrenar los tres modelos.
- Comparar sus resultados.
- Explicar las predicciones.
- Añadir las predicciones y la ordenación de sesiones a la aplicación.
- Añadir pruebas para comprobar el funcionamiento del código.

## Datos utilizados

Utilizamos el conjunto de datos Online Shoppers Purchasing Intention, publicado en UCI.

Contiene 12.330 sesiones y 17 variables de entrada. La variable `Revenue` indica si una sesión terminó en compra.

El archivo está en:

```text
data/raw/online_shoppers_intention.csv
```

Fuente: https://doi.org/10.24432/C5F88Q

Autores: Sakar, C. y Kastro, Y. (2018).

Licencia: CC BY 4.0.

## Preparar el entorno en macOS

Estos pasos utilizan Python 3.12.

Desde la carpeta principal del proyecto, donde está `app.py`, ejecutar:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-macos-py312.txt
```

En VS Code, abrir `Python: Select Interpreter` y seleccionar el intérprete de `.venv`.

La carpeta `.venv` contiene el entorno virtual y está excluida de Git.

Tenemos dos archivos de dependencias:

- `requirements.txt`: lista general de las bibliotecas del proyecto.
- `requirements-macos-py312.txt`: versiones exactas instaladas en el entorno que hemos comprobado en macOS con Python 3.12.7.

Si se utiliza otro sistema o versión de Python, habrá que comprobar también su instalación y funcionamiento.

## Comprobar las dependencias

Con el entorno virtual activo, ejecutar:

```bash
python -m pip check
```

El resultado esperado es:

```text
No broken requirements found.
```

Esto comprueba que las dependencias declaradas de las bibliotecas instaladas no tienen conflictos.

## Ejecutar la aplicación

Desde la carpeta principal del proyecto y con el entorno virtual activo, ejecutar:

```bash
python -m streamlit run app.py
```

Abrir en el navegador la dirección que aparece como `Local URL` en la terminal.

La aplicación debe mostrar el título «Predicción de compras online», un total de 12.330 sesiones y las primeras cinco filas de los datos.

Para detenerla, pulsar `Control + C` en la terminal.

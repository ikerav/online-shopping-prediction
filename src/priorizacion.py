import joblib
import pandas as pd

from src.datos import CARPETA_PROYECTO, cargar_datos
from src.entrenar_final import RUTA_MODELO
from src.preparacion import dividir_datos


def priorizar_sesiones(modelo, sesiones: pd.DataFrame) -> pd.DataFrame:
    """Predice varias sesiones y las ordena por probabilidad de compra."""

    # Rechazamos una tabla vacía o con nombres de columnas repetidos.
    if sesiones.empty:
        raise ValueError("El archivo no contiene sesiones.")

    if not sesiones.columns.is_unique:
        raise ValueError("El archivo contiene nombres de columnas repetidos.")

    # Comprobamos que están todas las características utilizadas al entrenar.
    columnas = list(modelo.feature_names_in_)
    faltantes = [
        columna
        for columna in columnas
        if columna not in sesiones.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas necesarias: " + ", ".join(faltantes)
        )

    # Seleccionamos únicamente las entradas del modelo y en el mismo orden.
    # Si el archivo contiene Revenue, esa columna no se utiliza.
    entradas = sesiones.loc[:, columnas].copy()

    if entradas.isna().any().any():
        raise ValueError("Hay valores vacíos en las características de las sesiones.")

    # Calculamos las probabilidades de todas las sesiones de una vez.
    indice_compra = list(modelo.classes_).index(True)
    probabilidades = modelo.predict_proba(entradas)[:, indice_compra]
    predicciones = modelo.predict(entradas)

    # Conservamos los datos de cada sesión junto con sus resultados.
    resultado = entradas.reset_index(drop=True).copy()

    # Este número identifica la posición original dentro de la tabla recibida.
    # Se mantiene unido a su sesión cuando ordenamos los resultados.
    resultado.insert(0, "Sesion", range(1, len(resultado) + 1))
    resultado["Probabilidad_compra"] = probabilidades
    resultado["Prediccion_compra"] = predicciones

    # Ordenamos de mayor a menor probabilidad.
    # stable conserva el orden original cuando dos probabilidades son iguales.
    resultado = resultado.sort_values(
        "Probabilidad_compra",
        ascending=False,
        kind="stable",
    )

    return resultado.reset_index(drop=True)


if __name__ == "__main__":
    modelo = joblib.load(RUTA_MODELO)
    datos = cargar_datos()

    # Utilizamos diez sesiones del entrenamiento como ejemplo.
    # No estamos calculando nuevas métricas de evaluación.
    X_entrenamiento, _, _, _ = dividir_datos(datos)
    muestra = X_entrenamiento.head(10).copy()

    resultado = priorizar_sesiones(modelo, muestra)

    # Convertimos a porcentaje solo para mostrarlo.
    # La función conserva las probabilidades originales entre 0 y 1.
    vista = resultado[
        ["Sesion", "Probabilidad_compra", "Prediccion_compra"]
    ].copy()

    vista["Probabilidad_compra"] = (
        vista["Probabilidad_compra"] * 100
    ).round(2)

    vista = vista.rename(
        columns={"Probabilidad_compra": "Probabilidad_compra_pct"}
    )

    print("SESIONES ORDENADAS POR PROBABILIDAD DE COMPRA")
    print(vista.to_string(index=False))

    # Guardamos un CSV de ejemplo para probar después la carga en la aplicación.
    ruta_ejemplo = CARPETA_PROYECTO / "data" / "sesiones_ejemplo.csv"
    muestra.to_csv(ruta_ejemplo, index=False)
    print(f"\nArchivo de ejemplo guardado en: {ruta_ejemplo}")

    # Comprobamos dos errores habituales al recibir un archivo.
    casos_incorrectos = [
        ("tabla vacía", muestra.head(0)),
        ("falta la columna Month", muestra.drop(columns=["Month"])),
    ]

    for nombre, caso in casos_incorrectos:
        try:
            priorizar_sesiones(modelo, caso)
        except ValueError as error:
            print(f"\nComprobación correcta ({nombre}): {error}")
        else:
            raise AssertionError(
                f"No se ha detectado el error del caso: {nombre}"
            )
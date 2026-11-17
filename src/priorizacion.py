import joblib
import pandas as pd

from src.datos import CARPETA_PROYECTO, cargar_datos
from src.entrenar_final import RUTA_MODELO
from src.preparacion import dividir_datos
from src.validacion import validar_sesiones


def priorizar_sesiones(modelo, sesiones: pd.DataFrame) -> pd.DataFrame:
    """Valida las sesiones y las ordena por probabilidad de compra."""

    # Aplicamos las comprobaciones antes de enviar los datos al modelo.
    # La función devuelve una copia con las columnas en el orden correcto.
    entradas = validar_sesiones(modelo, sesiones)

    indice_compra = list(modelo.classes_).index(True)
    probabilidades = modelo.predict_proba(entradas)[:, indice_compra]
    predicciones = modelo.predict(entradas)

    # Conservamos las características junto con sus resultados.
    resultado = entradas.reset_index(drop=True).copy()

    # Este identificador representa la posición original de cada sesión.
    # Se mantiene unido a sus datos después de ordenar.
    resultado.insert(0, "Sesion", range(1, len(resultado) + 1))
    resultado["Probabilidad_compra"] = probabilidades
    resultado["Prediccion_compra"] = predicciones

    # Los empates conservan el orden de entrada.
    resultado = resultado.sort_values(
        "Probabilidad_compra",
        ascending=False,
        kind="stable",
    )

    return resultado.reset_index(drop=True)


if __name__ == "__main__":
    modelo = joblib.load(RUTA_MODELO)
    datos = cargar_datos()

    # Utilizamos los mismos diez ejemplos que en la comprobación anterior.
    X_entrenamiento, _, _, _ = dividir_datos(datos)
    muestra = X_entrenamiento.head(10).copy()

    resultado = priorizar_sesiones(modelo, muestra)

    # Convertimos las probabilidades a porcentaje solo para mostrarlas.
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

    # Creamos una copia incorrecta para comprobar la nueva validación.
    # Usamos el índice real de la primera fila para modificarla,
    # evitando añadir accidentalmente una fila nueva.
    paginas_negativas = muestra.copy()
    primera_fila = paginas_negativas.index[0]
    paginas_negativas.loc[primera_fila, "ProductRelated"] = -1

    # Guardamos un ejemplo válido y otro con el error preparado.
    # El segundo servirá para probar el mensaje de error en la aplicación.
    archivos_ejemplo = [
        ("sesiones_ejemplo.csv", muestra),
        ("sesiones_error_paginas.csv", paginas_negativas),
    ]

    for nombre, contenido in archivos_ejemplo:
        ruta = CARPETA_PROYECTO / "data" / nombre
        contenido.to_csv(ruta, index=False)
        print(f"\nArchivo de ejemplo guardado en: {ruta}")

    casos_incorrectos = [
        ("tabla vacía", muestra.head(0)),
        ("falta la columna Month", muestra.drop(columns=["Month"])),
        ("páginas negativas", paginas_negativas),
    ]

    # Estas comprobaciones verifican que la función de priorización
    # utiliza realmente la validación antes de predecir.
    for nombre, caso in casos_incorrectos:
        try:
            priorizar_sesiones(modelo, caso)
        except ValueError as error:
            print(f"\nComprobación correcta ({nombre}): {error}")
        else:
            raise AssertionError(
                f"No se ha detectado el error del caso: {nombre}"
            )
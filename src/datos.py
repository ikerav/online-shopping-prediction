from pathlib import Path

import pandas as pd


# Localizamos los datos a partir de este archivo para evitar depender
# de la carpeta desde la que se ejecute el programa.
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS = CARPETA_PROYECTO / "data" / "raw" / "online_shoppers_intention.csv"


def cargar_datos(ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    """Lee el CSV y comprueba que contiene datos y la columna objetivo."""

    # Si falta el archivo, mostramos dónde lo estábamos buscando.
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encuentra el archivo: {ruta}")

    datos = pd.read_csv(ruta)

    # Estas comprobaciones ayudan a detectar un archivo incorrecto
    # antes de intentar analizarlo o entrenar un modelo.
    if datos.empty:
        raise ValueError("El archivo no contiene ninguna sesión.")

    if "Revenue" not in datos.columns:
        raise ValueError("Falta la columna Revenue, que indica si hubo compra.")

    return datos


def mostrar_resumen(datos: pd.DataFrame) -> None:
    """Muestra información básica para conocer el conjunto de datos."""

    print(f"Número de sesiones: {len(datos)}")
    print(f"Número de columnas: {len(datos.columns)}")

    # Los tipos nos ayudarán a distinguir números y categorías.
    print("\nTIPOS DE DATOS")
    print(datos.dtypes)

    # Revisamos los valores vacíos y los duplicados.
    # Por ahora solo los contamos, sin modificar ni eliminar filas.
    print("\nVALORES VACÍOS POR COLUMNA")
    print(datos.isna().sum())

    print(f"\nFilas duplicadas: {datos.duplicated().sum()}")

    # Revenue es la variable que queremos predecir:
    # True significa que hubo compra y False que no la hubo.
    recuento = datos["Revenue"].value_counts(dropna=False)
    porcentajes = (recuento / len(datos) * 100).round(2)

    resumen_compras = pd.DataFrame(
        {
            "Sesiones": recuento,
            "Porcentaje": porcentajes,
        }
    )

    print("\nSESIONES CON Y SIN COMPRA")
    print(resumen_compras)


# Este bloque se ejecuta al lanzar el módulo desde la terminal.
# Al importar cargar_datos desde otro archivo, no se muestra el resumen.
if __name__ == "__main__":
    datos = cargar_datos()
    mostrar_resumen(datos)
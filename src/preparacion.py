import pandas as pd
from sklearn.model_selection import train_test_split

from src.datos import cargar_datos


def dividir_datos(datos: pd.DataFrame):
    """Separa las filas únicas en entrenamiento y prueba final."""

    # Para este primer experimento usamos una sola copia de cada fila.
    # Esta operación crea otro DataFrame y no modifica el CSV original.
    datos_unicos = datos.drop_duplicates()

    # X contiene la información que conocerá el modelo.
    # y contiene la respuesta que queremos predecir: compra o no compra.
    X = datos_unicos.drop(columns=["Revenue"])
    y = datos_unicos["Revenue"]

    # Reservamos el 20 % para evaluar el modelo al terminar.
    # stratify mantiene una proporción parecida de compras en cada grupo.
    # random_state fija el reparto para poder repetir el experimento.
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba


if __name__ == "__main__":
    datos = cargar_datos()

    # Mostramos cuántas filas utilizamos después de quitar las repetidas.
    print(f"Sesiones originales: {len(datos)}")
    print(f"Filas repetidas descartadas: {datos.duplicated().sum()}")
    print(f"Sesiones utilizadas: {len(datos.drop_duplicates())}")

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = dividir_datos(datos)

    print(f"\nSesiones de entrenamiento: {len(X_entrenamiento)}")
    print(f"Sesiones de prueba final: {len(X_prueba)}")

    # Comprobamos la proporción de compras del grupo de entrenamiento.
    # La media de una columna booleana es la proporción de valores True.
    porcentaje_compras = y_entrenamiento.mean() * 100
    print(f"Compras en entrenamiento: {porcentaje_compras:.2f} %")
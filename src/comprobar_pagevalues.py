from sklearn.ensemble import RandomForestClassifier

from src.datos import cargar_datos
from src.modelo_base import crear_modelo, evaluar_modelo
from src.preparacion import dividir_datos


if __name__ == "__main__":
    datos = cargar_datos()

    # Primero repetimos el reparto que usamos en los experimentos anteriores.
    # Así mantenemos exactamente las mismas sesiones de entrenamiento.
    X_entrenamiento, _, y_entrenamiento, _ = dividir_datos(datos)

    # Quitamos únicamente PageValues de las entradas del modelo.
    # No modificamos el CSV ni cambiamos las filas utilizadas.
    X_sin_pagevalues = X_entrenamiento.drop(columns=["PageValues"])

    # Conservamos la configuración del bosque anterior.
    # De esta manera, el único cambio del experimento es quitar la variable.
    clasificador = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    # Reutilizamos la preparación y las mismas cinco particiones.
    modelo = crear_modelo(X_sin_pagevalues.columns, clasificador)

    evaluar_modelo(
        "Random Forest sin PageValues",
        modelo,
        X_sin_pagevalues,
        y_entrenamiento,
    )
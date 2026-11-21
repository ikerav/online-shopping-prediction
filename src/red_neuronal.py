from sklearn.neural_network import MLPClassifier

from src.datos import cargar_datos
from src.modelo_base import crear_modelo, evaluar_modelo
from src.preparacion import dividir_datos


def crear_red_neuronal():
    """Configura una red neuronal sencilla para clasificar las sesiones."""

    # Utilizamos dos capas ocultas: una de 32 neuronas y otra de 16.
    # Es una configuración inicial que después podremos comparar con otras.
    #
    # El entrenamiento tendrá un máximo de 500 pasadas por los datos.
    # También podrá detenerse antes si deja de mejorar durante 20 pasadas
    # en el grupo de validación que la propia red reserva internamente.
    return MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=20,
        random_state=42,
    )


if __name__ == "__main__":
    datos = cargar_datos()

    # Usamos el mismo grupo de entrenamiento que los otros modelos.
    # La prueba final sigue reservada.
    X_entrenamiento, _, y_entrenamiento, _ = dividir_datos(datos)

    # Reutilizamos el escalado y la conversión de categorías.
    # El escalado es especialmente importante para la red neuronal.
    clasificador = crear_red_neuronal()
    modelo = crear_modelo(X_entrenamiento.columns, clasificador)

    # La función utiliza las mismas cinco particiones y las mismas métricas
    # que hemos utilizado para evaluar los otros dos modelos.
    evaluar_modelo(
        "Red neuronal MLP",
        modelo,
        X_entrenamiento,
        y_entrenamiento,
    )
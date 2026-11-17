import joblib
from sklearn.ensemble import RandomForestClassifier

from src.datos import CARPETA_PROYECTO, cargar_datos
from src.modelo_base import crear_modelo
from src.preparacion import dividir_datos


RUTA_MODELO = CARPETA_PROYECTO / "models" / "random_forest.joblib"


def entrenar_y_guardar_modelo():
    """Entrena el modelo elegido y guarda toda la preparación necesaria."""

    datos = cargar_datos()

    # Repetimos el reparto anterior y utilizamos solo el entrenamiento.
    # El grupo de prueba final sigue reservado.
    X_entrenamiento, _, y_entrenamiento, _ = dividir_datos(datos)

    # Conservamos la configuración que hemos evaluado.
    clasificador = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    # El Pipeline incluye la preparación y el clasificador.
    # Así podremos aplicar exactamente las mismas transformaciones
    # cuando la aplicación reciba una sesión nueva.
    modelo = crear_modelo(X_entrenamiento.columns, clasificador)

    print(f"Entrenando con {len(X_entrenamiento)} sesiones...")
    modelo.fit(X_entrenamiento, y_entrenamiento)

    # Creamos la carpeta si falta y guardamos el Pipeline completo.
    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, RUTA_MODELO)

    # Volvemos a cargarlo para comprobar que se ha guardado correctamente.
    # Usamos cinco sesiones de entrenamiento para comparar las predicciones.
    # Esta comprobación revisa el guardado, no la calidad del modelo.
    modelo_guardado = joblib.load(RUTA_MODELO)
    ejemplos = X_entrenamiento.head(5)

    predicciones_originales = modelo.predict(ejemplos)
    predicciones_guardadas = modelo_guardado.predict(ejemplos)

    if not (predicciones_originales == predicciones_guardadas).all():
        raise ValueError("Las predicciones han cambiado al guardar el modelo.")

    print(f"Modelo guardado en: {RUTA_MODELO}")
    print("Comprobación de guardado y carga: correcta.")


if __name__ == "__main__":
    entrenar_y_guardar_modelo()
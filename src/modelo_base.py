from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.datos import cargar_datos
from src.preparacion import dividir_datos


def crear_modelo(columnas, clasificador):
    """Une la preparación de datos con el clasificador elegido."""

    # Estos valores representan categorías, aunque algunos sean números.
    # Por ejemplo, los códigos de navegador no indican cantidades.
    columnas_categoricas = [
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend",
    ]

    # El resto de las columnas son cantidades, tiempos o tasas.
    columnas_numericas = [
        columna
        for columna in columnas
        if columna not in columnas_categoricas
    ]

    # Ajustamos la escala de los números y convertimos las categorías.
    # Random Forest no necesita escalado, pero mantenemos una preparación
    # común que también sirve para la regresión logística.
    preparacion = ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), columnas_numericas),
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                columnas_categoricas,
            ),
        ]
    )

    # El clasificador se recibe como argumento para reutilizar este código.
    # Cada entrenamiento aprende también su propia preparación de datos.
    modelo = Pipeline(
        steps=[
            ("preparacion", preparacion),
            ("clasificador", clasificador),
        ]
    )

    return modelo


def evaluar_modelo(nombre_modelo, modelo, X, y):
    """Evalúa un modelo con cinco particiones y muestra sus resultados."""

    # Usamos la misma semilla para repetir exactamente las particiones
    # al evaluar cada modelo y poder compararlos.
    particiones = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # La compra es la clase positiva para precisión, recall y F1.
    metricas = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC_AUC": "roc_auc",
        "AP": "average_precision",
    }

    print(f"\n{nombre_modelo.upper()}: VALIDACIÓN CRUZADA")
    print("Entrenando y evaluando las cinco particiones...")

    # Se entrena una copia del Pipeline en cada partición.
    # Mostramos cualquier error para poder corregirlo.
    resultados = cross_validate(
        modelo,
        X,
        y,
        cv=particiones,
        scoring=metricas,
        error_score="raise",
    )

    # test_ identifica la evaluación de cada partición.
    # El conjunto de prueba final sigue reservado.
    for nombre in metricas:
        valores = resultados[f"test_{nombre}"]
        print(
            f"{nombre}: {valores.mean():.3f} "
            f"(desviación: {valores.std():.3f})"
        )


if __name__ == "__main__":
    datos = cargar_datos()

    # Solo utilizamos el grupo de entrenamiento.
    X_entrenamiento, _, y_entrenamiento, _ = dividir_datos(datos)

    referencia = 1 - y_entrenamiento.mean()
    print(f"Accuracy de predecir siempre 'no compra': {referencia:.3f}")

    # Guardamos los clasificadores en un diccionario para evaluarlos
    # con el mismo código, evitando repetir toda la preparación.
    clasificadores = {
        "Regresión logística": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    # Creamos y evaluamos un Pipeline para cada clasificador.
    # n_jobs=-1 permite al bosque aprovechar los procesadores disponibles.
    for nombre, clasificador in clasificadores.items():
        modelo = crear_modelo(X_entrenamiento.columns, clasificador)
        evaluar_modelo(nombre, modelo, X_entrenamiento, y_entrenamiento)
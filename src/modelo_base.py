from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.datos import cargar_datos
from src.preparacion import dividir_datos


def crear_modelo(columnas):
    """Crea la preparación de datos y la regresión logística."""

    # Algunas columnas contienen números que representan categorías.
    # Por ejemplo, un código de navegador no representa una cantidad.
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

    # StandardScaler ajusta la escala de las variables numéricas.
    # OneHotEncoder crea columnas de 0 y 1 para las categorías.
    # handle_unknown permite recibir categorías nuevas sin dar un error.
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

    # Unimos la preparación y el modelo en una misma secuencia.
    # Así, la preparación se aprende solo con los datos de cada entrenamiento.
    modelo = Pipeline(
        steps=[
            ("preparacion", preparacion),
            ("clasificador", LogisticRegression(max_iter=1000)),
        ]
    )

    return modelo


def evaluar_modelo(modelo, X, y):
    """Evalúa el modelo mediante validación cruzada estratificada."""

    # Dividimos el grupo de entrenamiento en cinco partes.
    # En cada vuelta entrenamos con cuatro y evaluamos con la restante.
    particiones = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # Las métricas de precisión, recall y F1 consideran la compra
    # como el resultado positivo que queremos detectar.
    metricas = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC_AUC": "roc_auc",
        "AP": "average_precision",
    }

    # cross_validate entrena una copia del Pipeline en cada partición.
    # Si hay un fallo, error_score="raise" permite ver el error completo.
    resultados = cross_validate(
        modelo,
        X,
        y,
        cv=particiones,
        scoring=metricas,
        error_score="raise",
    )

    print("\nREGRESIÓN LOGÍSTICA: VALIDACIÓN CRUZADA")

    # Mostramos la media y la variación entre las cinco evaluaciones.
    # El prefijo test_ se refiere a la evaluación de cada partición,
    # no al conjunto de prueba final que tenemos reservado.
    for nombre in metricas:
        valores = resultados[f"test_{nombre}"]
        print(
            f"{nombre}: {valores.mean():.3f} "
            f"(desviación: {valores.std():.3f})"
        )


if __name__ == "__main__":
    datos = cargar_datos()

    # Utilizamos únicamente el grupo de entrenamiento.
    # Los guiones bajos descartan los dos grupos de prueba final.
    X_entrenamiento, _, y_entrenamiento, _ = dividir_datos(datos)

    # Esta referencia muestra el porcentaje de aciertos que conseguiríamos
    # diciendo siempre que no habrá compra, sin aprender ningún patrón.
    referencia = 1 - y_entrenamiento.mean()
    print(f"Accuracy de predecir siempre 'no compra': {referencia:.3f}")

    modelo = crear_modelo(X_entrenamiento.columns)
    evaluar_modelo(modelo, X_entrenamiento, y_entrenamiento)
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.datos import CARPETA_PROYECTO, cargar_datos
from src.entrenar_final import RUTA_MODELO
from src.preparacion import dividir_datos


def calcular_metricas(y_real, predicciones, probabilidades):
    """Calcula las métricas considerando la compra como clase positiva."""

    # Calculamos los puntos de la curva precisión-recall.
    # Su área mediante trapecios y AP son cálculos distintos.
    precision_curva, recall_curva, _ = precision_recall_curve(
        y_real,
        probabilidades,
        pos_label=True,
    )

    # zero_division=0 evita una división por cero si el modelo
    # no predice ninguna compra.
    return {
        "Accuracy": accuracy_score(y_real, predicciones),
        "Precision": precision_score(y_real, predicciones, zero_division=0),
        "Recall": recall_score(y_real, predicciones, zero_division=0),
        "F1": f1_score(y_real, predicciones, zero_division=0),
        "ROC_AUC": roc_auc_score(y_real, probabilidades),
        "AP": average_precision_score(y_real, probabilidades),
        "PR_AUC": auc(recall_curva, precision_curva),
    }


if __name__ == "__main__":
    # Cargamos el modelo ya entrenado. Aquí no volvemos a entrenarlo.
    if not RUTA_MODELO.is_file():
        raise FileNotFoundError(
            "Falta el modelo. Ejecuta primero: python -m src.entrenar_final"
        )

    modelo = joblib.load(RUTA_MODELO)

    # Recuperamos exactamente el grupo reservado en el reparto original.
    datos = cargar_datos()
    _, X_prueba, _, y_prueba = dividir_datos(datos)

    # predict devuelve la clasificación y predict_proba las probabilidades.
    # Buscamos la columna de True para obtener la probabilidad de compra.
    predicciones = modelo.predict(X_prueba)
    indice_compra = list(modelo.classes_).index(True)
    probabilidades = modelo.predict_proba(X_prueba)[:, indice_compra]

    metricas = calcular_metricas(y_prueba, predicciones, probabilidades)

    print(f"Sesiones de prueba final: {len(X_prueba)}")
    print("\nRANDOM FOREST: RESULTADOS DE PRUEBA FINAL")

    for nombre, valor in metricas.items():
        print(f"{nombre}: {valor:.3f}")

    # Las filas representan lo que ocurrió realmente.
    # Las columnas representan lo que predijo el modelo.
    matriz = pd.DataFrame(
        confusion_matrix(y_prueba, predicciones, labels=[False, True]),
        index=["Real: no compra", "Real: compra"],
        columns=["Predicho: no compra", "Predicho: compra"],
    )

    print("\nMATRIZ DE CONFUSIÓN")
    print(matriz.to_string())

    # Guardamos los resultados completos para utilizarlos en la memoria.
    carpeta_resultados = CARPETA_PROYECTO / "reports"
    carpeta_resultados.mkdir(parents=True, exist_ok=True)

    informe = {
        "modelo": "Random Forest",
        "sesiones_prueba": len(X_prueba),
        "umbral_clasificacion": 0.5,
        "metricas": metricas,
    }

    ruta_metricas = carpeta_resultados / "metricas_prueba_final.json"
    with ruta_metricas.open("w", encoding="utf-8") as archivo:
        json.dump(informe, archivo, indent=4, ensure_ascii=False)

    ruta_matriz = carpeta_resultados / "matriz_confusion_prueba_final.csv"
    matriz.to_csv(ruta_matriz)

    print(f"\nMétricas guardadas en: {ruta_metricas}")
    print(f"Matriz guardada en: {ruta_matriz}")
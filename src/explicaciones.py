import joblib
import numpy as np
import pandas as pd
import shap

from src.datos import cargar_datos
from src.entrenar_final import RUTA_MODELO
from src.preparacion import dividir_datos


def crear_explicador(modelo):
    """Prepara SHAP utilizando la información de los árboles entrenados."""

    bosque = modelo.named_steps["clasificador"]

    # Esta configuración utiliza los recuentos de entrenamiento
    # almacenados en los caminos de los árboles como referencia.
    #
    # En este Random Forest, la salida "raw" ya son probabilidades
    # para las dos clases: no compra y compra.
    explicador = shap.TreeExplainer(
        bosque,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )

    return explicador


def explicar_sesion(modelo, explicador, sesion):
    """Explica la probabilidad de compra de una única sesión."""

    if len(sesion) != 1:
        raise ValueError(
            "La explicación individual necesita exactamente una sesión."
        )

    # Conservamos el orden de las columnas y aplicamos
    # la misma preparación utilizada al entrenar.
    sesion = sesion.loc[:, modelo.feature_names_in_]
    preparacion = modelo.named_steps["preparacion"]
    sesion_transformada = preparacion.transform(sesion).astype(np.float32)

    # Seleccionamos la salida correspondiente a una compra.
    indice_compra = list(modelo.classes_).index(True)

    valores = explicador.shap_values(sesion_transformada)
    contribuciones = valores[0, :, indice_compra]

    referencia = float(explicador.expected_value[indice_compra])
    probabilidad = float(modelo.predict_proba(sesion)[0, indice_compra])

    # Mantenemos la comprobación con la misma tolerancia anterior.
    probabilidad_explicada = referencia + contribuciones.sum()
    diferencia = abs(probabilidad_explicada - probabilidad)

    if not np.isclose(probabilidad_explicada, probabilidad, atol=0.00001):
        raise ValueError(
            "La explicación SHAP no coincide con la predicción. "
            f"Modelo: {probabilidad:.12f}. "
            f"SHAP: {probabilidad_explicada:.12f}. "
            f"Diferencia: {diferencia:.12f}."
        )

    # Quitamos los prefijos añadidos por ColumnTransformer.
    # Las categorías conservan nombres como Month_Nov.
    nombres = [
        nombre.split("__", 1)[-1]
        for nombre in preparacion.get_feature_names_out()
    ]

    # Las contribuciones se expresan en puntos porcentuales.
    # El signo indica si aumentan o reducen la probabilidad
    # respecto a la referencia.
    tabla = pd.DataFrame(
        {
            "Variable": nombres,
            "Contribucion_pp": contribuciones * 100,
        }
    )

    # Ordenamos por el tamaño de la contribución sin perder su signo.
    tabla["Impacto_absoluto"] = tabla["Contribucion_pp"].abs()
    tabla = tabla.sort_values("Impacto_absoluto", ascending=False)
    tabla = tabla[["Variable", "Contribucion_pp"]]

    return referencia, probabilidad, tabla


if __name__ == "__main__":
    modelo = joblib.load(RUTA_MODELO)
    datos = cargar_datos()

    # Todas las comprobaciones utilizan el grupo de entrenamiento.
    X_entrenamiento, _, y_entrenamiento, _ = dividir_datos(datos)

    print("Preparando la explicación SHAP...")
    explicador = crear_explicador(modelo)

    # Repetimos la explicación de la misma sesión anterior.
    sesion = X_entrenamiento.head(1)
    referencia, probabilidad, tabla = explicar_sesion(
        modelo,
        explicador,
        sesion,
    )

    # La tabla contiene todas las contribuciones, aunque después
    # mostremos solo las diez de mayor tamaño.
    reconstruida = referencia + tabla["Contribucion_pp"].sum() / 100
    diferencia = abs(reconstruida - probabilidad)

    print(f"\nProbabilidad del modelo: {probabilidad:.12f}")
    print(f"Probabilidad reconstruida por SHAP: {reconstruida:.12f}")
    print(f"Diferencia absoluta: {diferencia:.12f}")
    print(f"Probabilidad de referencia: {referencia:.2%}")

    print("\nLAS DIEZ CONTRIBUCIONES DE MAYOR TAMAÑO")
    print(tabla.head(10).round(3).to_string(index=False))

    # Comprobamos también tres sesiones de cada clase.
    # Si alguna explicación no coincide, la función mostrará el error.
    for clase in [False, True]:
        ejemplos = X_entrenamiento.loc[y_entrenamiento == clase].sample(
            n=3,
            random_state=42,
        )

        for indice in ejemplos.index:
            explicar_sesion(modelo, explicador, ejemplos.loc[[indice]])

    print("\nComprobación adicional en seis sesiones: correcta.")
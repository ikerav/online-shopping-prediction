import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datos import CARPETA_PROYECTO, cargar_datos
from src.entrenar_final import RUTA_MODELO
from src.explicaciones import crear_explicador
from src.preparacion import dividir_datos


def calcular_importancia_global(modelo, sesiones):
    """Resume la magnitud de las contribuciones SHAP de varias sesiones."""

    if sesiones.empty:
        raise ValueError("Se necesita al menos una sesión.")

    # Aplicamos la misma preparación y el mismo orden de columnas
    # que utilizamos al entrenar.
    sesiones = sesiones.loc[:, modelo.feature_names_in_]
    preparacion = modelo.named_steps["preparacion"]
    transformadas = preparacion.transform(sesiones).astype(np.float32)

    explicador = crear_explicador(modelo)
    indice_compra = list(modelo.classes_).index(True)

    # Ahora explicamos todas las sesiones de la muestra de una vez.
    valores = explicador.shap_values(transformadas)
    contribuciones = valores[:, :, indice_compra]
    referencia = float(explicador.expected_value[indice_compra])

    # Verificamos que cada explicación reconstruye su predicción.
    probabilidades = modelo.predict_proba(sesiones)[:, indice_compra]
    reconstruidas = referencia + contribuciones.sum(axis=1)

    if not np.allclose(reconstruidas, probabilidades, atol=0.00001):
        raise ValueError("Alguna explicación no coincide con su predicción.")

    nombres = [
        nombre.split("__", 1)[-1]
        for nombre in preparacion.get_feature_names_out()
    ]

    tabla = pd.DataFrame(contribuciones, columns=nombres)

    # Recuperamos la lista de campos categóricos del preprocesador.
    columnas_categoricas = preparacion.transformers_[1][2]
    agrupadas = pd.DataFrame(index=tabla.index)

    for columna in modelo.feature_names_in_:
        if columna in columnas_categoricas:
            # Agrupamos, por ejemplo, todas las columnas Month_...
            columnas_grupo = [
                nombre
                for nombre in nombres
                if nombre.startswith(f"{columna}_")
            ]
        else:
            columnas_grupo = [columna]

        # Sumamos las categorías dentro de cada sesión.
        # axis=1 significa sumar las columnas de cada fila.
        agrupadas[columna] = tabla[columnas_grupo].sum(axis=1)

    # La agrupación debe conservar la suma de las contribuciones.
    if not np.allclose(
        agrupadas.sum(axis=1).to_numpy(),
        contribuciones.sum(axis=1),
        atol=0.0000000001,
    ):
        raise ValueError("La agrupación ha cambiado las contribuciones.")

    filas = []

    for columna, valores_campo in agrupadas.items():
        # Tomamos el valor absoluto antes de calcular la media.
        # Así, los aumentos y reducciones de distintas sesiones
        # no se cancelan entre sí.
        importancia = valores_campo.abs().mean() * 100

        filas.append(
            {
                "Variable": columna,
                "Importancia_media_pp": importancia,
            }
        )

    resultado = pd.DataFrame(filas)

    return resultado.sort_values(
        "Importancia_media_pp",
        ascending=False,
    ).reset_index(drop=True)


def guardar_resultados(importancia, numero_sesiones):
    """Guarda la tabla de importancia y un gráfico para la memoria."""

    carpeta_reportes = CARPETA_PROYECTO / "reports"
    carpeta_figuras = carpeta_reportes / "figures"
    carpeta_figuras.mkdir(parents=True, exist_ok=True)

    ruta_csv = carpeta_reportes / "importancia_global.csv"
    importancia.to_csv(ruta_csv, index=False)

    # Ordenamos de menor a mayor para que la barra más importante
    # aparezca arriba en el gráfico horizontal.
    datos_grafico = importancia.sort_values("Importancia_media_pp")

    figura, eje = plt.subplots(figsize=(9, 7))

    eje.barh(
        datos_grafico["Variable"],
        datos_grafico["Importancia_media_pp"],
        color="#2878B5",
    )

    eje.set_xlabel("Contribución absoluta media (puntos porcentuales)")
    eje.set_title(
        f"Importancia global: {numero_sesiones} sesiones de entrenamiento"
    )

    figura.tight_layout()

    ruta_figura = carpeta_figuras / "importancia_global.png"
    figura.savefig(ruta_figura, dpi=180, bbox_inches="tight")
    plt.close(figura)

    return ruta_csv, ruta_figura


if __name__ == "__main__":
    modelo = joblib.load(RUTA_MODELO)
    datos = cargar_datos()
    X_entrenamiento, _, _, _ = dividir_datos(datos)

    # Utilizamos una muestra fija del entrenamiento.
    # Este análisis no modifica ni vuelve a entrenar el modelo.
    muestra = X_entrenamiento.sample(n=100, random_state=42)

    print(f"Calculando SHAP para {len(muestra)} sesiones...")
    importancia = calcular_importancia_global(modelo, muestra)

    print("\nIMPORTANCIA GLOBAL")
    print(importancia.round(3).to_string(index=False))

    ruta_csv, ruta_figura = guardar_resultados(importancia, len(muestra))

    print("\nComprobación de las explicaciones y la agrupación: correcta.")
    print(f"Tabla guardada en: {ruta_csv}")
    print(f"Gráfico guardado en: {ruta_figura}")
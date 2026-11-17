import numpy as np
import pandas as pd


# Estos campos representan recuentos o códigos enteros.
COLUMNAS_ENTERAS = [
    "Administrative",
    "Informational",
    "ProductRelated",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

COLUMNAS_TASAS = ["BounceRates", "ExitRates", "SpecialDay"]
COLUMNAS_TEXTO = ["Month", "VisitorType"]


def validar_sesiones(modelo, sesiones: pd.DataFrame) -> pd.DataFrame:
    """Comprueba las sesiones y devuelve una copia preparada para predecir."""

    if sesiones.empty:
        raise ValueError("El archivo no contiene sesiones.")

    if not sesiones.columns.is_unique:
        raise ValueError("El archivo contiene nombres de columnas repetidos.")

    columnas = list(modelo.feature_names_in_)
    faltantes = [
        columna
        for columna in columnas
        if columna not in sesiones.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas necesarias: " + ", ".join(faltantes)
        )

    # Trabajamos sobre una copia y seleccionamos solo las entradas del modelo.
    # Revenue y otras columnas adicionales no se utilizan.
    entradas = sesiones.loc[:, columnas].copy()

    if entradas.isna().any().any():
        raise ValueError("Hay valores vacíos en las características.")

    columnas_numericas = [
        columna
        for columna in columnas
        if columna not in COLUMNAS_TEXTO + ["Weekend"]
    ]

    for columna in columnas_numericas:
        # Un campo numérico debe contener números, no True o False.
        if pd.api.types.is_bool_dtype(entradas[columna]):
            raise ValueError(f"{columna} debe contener números.")

        # Admitimos números escritos como texto, pero rechazamos
        # contenido que no pueda convertirse a un número.
        try:
            entradas[columna] = pd.to_numeric(
                entradas[columna],
                errors="raise",
            )
        except (ValueError, TypeError):
            raise ValueError(
                f"{columna} contiene un valor que no es numérico."
            ) from None

        valores = entradas[columna]

        if not np.isfinite(valores.to_numpy(dtype=float)).all():
            raise ValueError(f"{columna} contiene valores no finitos.")

        if (valores < 0).any():
            raise ValueError(f"{columna} no admite valores negativos.")

        # El resto de dividir entre 1 debe ser cero en los campos enteros.
        if columna in COLUMNAS_ENTERAS and (valores % 1 != 0).any():
            raise ValueError(f"{columna} solo admite números enteros.")

        if columna in COLUMNAS_TASAS and (valores > 1).any():
            raise ValueError(f"{columna} debe estar entre 0 y 1.")

    # Normalizamos el fin de semana a un valor booleano.
    if not entradas["Weekend"].isin([True, False]).all():
        raise ValueError("Weekend solo admite True, False, 1 o 0.")

    entradas["Weekend"] = entradas["Weekend"].astype(bool)

    # Comprobamos los nombres de meses y tipos de visitante
    # usando las categorías que aprendió el preprocesador.
    preparacion = modelo.named_steps["preparacion"]
    columnas_categoricas = preparacion.transformers_[1][2]
    codificador = preparacion.named_transformers_["categoricas"]

    for columna, categorias in zip(
        columnas_categoricas,
        codificador.categories_,
    ):
        if columna in COLUMNAS_TEXTO:
            if not entradas[columna].isin(categorias).all():
                opciones = ", ".join(str(valor) for valor in categorias)
                raise ValueError(
                    f"{columna} contiene categorías no conocidas. "
                    f"Opciones válidas: {opciones}."
                )

    return entradas
"""Pruebas de validación y priorización de sesiones."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from interfaz_priorizacion import validar_csv, calcular_priorizacion


RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def modelo():
    """Carga una sola vez el modelo definitivo."""
    artefacto = joblib.load(
        RAIZ / "models" / "random_forest_final.joblib"
    )
    return artefacto["modelo"], artefacto["umbral"]


@pytest.fixture
def sesion():
    """Crea una sesión válida con las 17 características."""
    return pd.DataFrame([{
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 10,
        "ProductRelated_Duration": 300.0,
        "BounceRates": 0.01,
        "ExitRates": 0.03,
        "PageValues": 12.0,
        "SpecialDay": 0.0,
        "Month": "Nov",
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 2,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }])


def test_sesion_valida(modelo, sesion):
    bosque, _ = modelo
    X = validar_csv(sesion, bosque)

    assert list(X.columns) == list(bosque.feature_names_in_)
    assert len(X) == 1


def test_columna_obligatoria_ausente(modelo, sesion):
    bosque, _ = modelo
    datos = sesion.drop(columns=["PageValues"])

    with pytest.raises(ValueError, match="Faltan columnas"):
        validar_csv(datos, bosque)


def test_valor_ausente(modelo, sesion):
    bosque, _ = modelo
    sesion.loc[0, "PageValues"] = np.nan

    with pytest.raises(ValueError, match="valores vacíos"):
        validar_csv(sesion, bosque)


@pytest.mark.parametrize("columna,valor", [
    ("ExitRates", 1.5),
    ("BounceRates", -0.1),
    ("Administrative", 2.5),
    ("Month", "Enero"),
    ("VisitorType", "Desconocido"),
    ("Weekend", "quizás"),
    ("PageValues", np.inf),
    ("ProductRelated", "abc"),
])
def test_valores_invalidos(modelo, sesion, columna, valor):
    bosque, _ = modelo
    sesion[columna] = [valor]

    with pytest.raises(ValueError):
        validar_csv(sesion, bosque)


def test_archivo_vacio(modelo, sesion):
    bosque, _ = modelo

    with pytest.raises(ValueError, match="no contiene sesiones"):
        validar_csv(sesion.iloc[0:0], bosque)


def test_limite_sesiones(modelo, sesion):
    bosque, _ = modelo

    datos = pd.concat([sesion] * 50001, ignore_index=True)

    with pytest.raises(ValueError, match="50.000"):
        validar_csv(datos, bosque)


def test_columnas_adicionales(modelo, sesion):
    bosque, _ = modelo

    sesion["Revenue"] = True
    sesion["Columna_extra"] = "ejemplo"

    X = validar_csv(sesion, bosque)

    assert "Revenue" not in X.columns
    assert "Columna_extra" not in X.columns
    assert X.shape[1] == 17


def test_ranking_y_predicciones(modelo, sesion):
    bosque, umbral = modelo

    datos = pd.concat([sesion] * 3, ignore_index=True)
    datos.loc[0, "PageValues"] = 0.0
    datos.loc[1, "PageValues"] = 15.0
    datos.loc[2, "PageValues"] = 80.0
    datos["Revenue"] = [False, True, True]

    resultados = calcular_priorizacion(datos, bosque, umbral)

    # Mantiene todas las sesiones y sus posiciones originales.
    assert len(resultados) == 3
    assert resultados["Prioridad"].tolist() == [1, 2, 3]
    assert set(resultados["Fila_original"]) == {1, 2, 3}

    # Las probabilidades deben estar ordenadas.
    probabilidades = resultados["Probabilidad_compra"].to_numpy()

    assert np.all(np.diff(probabilidades) <= 0)
    assert np.all((probabilidades >= 0) & (probabilidades <= 1))

    # La clasificación debe respetar el umbral definitivo.
    esperadas = np.where(
        probabilidades >= umbral,
        "Compra probable",
        "Compra no probable",
    )

    assert np.array_equal(resultados["Clasificacion"], esperadas)

    # Revenue nunca debe exportarse.
    assert "Revenue" not in resultados.columns

    # Cada probabilidad debe corresponder a su fila original.
    X = validar_csv(datos, bosque)
    predicciones_originales = bosque.predict_proba(X)[:, 1]

    np.testing.assert_allclose(
        probabilidades,
        predicciones_originales[
            resultados["Fila_original"].to_numpy() - 1
        ],
    )
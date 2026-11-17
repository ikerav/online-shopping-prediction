import unittest

import joblib
import pandas as pd

from src.datos import cargar_datos
from src.entrenar_final import RUTA_MODELO
from src.validacion import validar_sesiones


class PruebasValidacion(unittest.TestCase):
    """Comprueba datos válidos y errores habituales de entrada."""

    @classmethod
    def setUpClass(cls):
        # Cargamos una sola vez el modelo y tres sesiones del dataset.
        # Estas pruebas no entrenan ni modifican el modelo.
        cls.modelo = joblib.load(RUTA_MODELO)
        cls.base = cargar_datos().loc[
            :, cls.modelo.feature_names_in_
        ].head(3).copy()

    def setUp(self):
        # Cada prueba recibe una copia independiente.
        self.datos = self.base.copy()

    def test_datos_validos(self):
        # Cambiamos el orden y añadimos Revenue para comprobar que
        # se conservan únicamente las entradas correctas y bien ordenadas.
        self.datos = self.datos[self.datos.columns[::-1]].copy()
        self.datos["Revenue"] = True
        original = self.datos.copy()

        resultado = validar_sesiones(self.modelo, self.datos)

        self.assertEqual(
            list(resultado.columns),
            list(self.modelo.feature_names_in_),
        )
        self.assertNotIn("Revenue", resultado.columns)
        pd.testing.assert_frame_equal(self.datos, original)

    def test_tabla_vacia(self):
        with self.assertRaises(ValueError):
            validar_sesiones(self.modelo, self.datos.head(0))

    def test_columnas_repetidas(self):
        repetidas = pd.concat(
            [self.datos, self.datos[["Month"]]],
            axis=1,
        )

        with self.assertRaises(ValueError):
            validar_sesiones(self.modelo, repetidas)

    def test_columna_ausente(self):
        with self.assertRaises(ValueError):
            validar_sesiones(
                self.modelo,
                self.datos.drop(columns=["Month"]),
            )

    def test_paginas_negativas(self):
        self.datos.loc[0, "ProductRelated"] = -1

        with self.assertRaises(ValueError):
            validar_sesiones(self.modelo, self.datos)

    def test_paginas_con_decimales(self):
        # Convertimos primero la columna para poder introducir
        # deliberadamente un decimal en estos datos de prueba.
        self.datos["ProductRelated"] = self.datos["ProductRelated"].astype(float)
        self.datos.loc[0, "ProductRelated"] = 1.5

        with self.assertRaises(ValueError):
            validar_sesiones(self.modelo, self.datos)

    def test_texto_en_columna_numerica(self):
        self.datos["PageValues"] = self.datos["PageValues"].astype(str)
        self.datos.loc[0, "PageValues"] = "texto"

        with self.assertRaises(ValueError):
            validar_sesiones(self.modelo, self.datos)

    def test_valores_vacios_o_infinitos(self):
        # Probamos ambos casos por separado dentro de esta prueba.
        for valor in [float("nan"), float("inf")]:
            with self.subTest(valor=valor):
                caso = self.datos.copy()
                caso.loc[0, "PageValues"] = valor

                with self.assertRaises(ValueError):
                    validar_sesiones(self.modelo, caso)

    def test_tasa_fuera_de_rango(self):
        self.datos.loc[0, "BounceRates"] = 1.2

        with self.assertRaises(ValueError):
            validar_sesiones(self.modelo, self.datos)

    def test_fin_de_semana_incorrecto(self):
        self.datos["Weekend"] = self.datos["Weekend"].astype(object)
        self.datos.loc[0, "Weekend"] = "quizas"

        with self.assertRaises(ValueError):
            validar_sesiones(self.modelo, self.datos)

    def test_tipo_visitante_desconocido(self):
        self.datos.loc[0, "VisitorType"] = "Inventado"

        with self.assertRaises(ValueError):
            validar_sesiones(self.modelo, self.datos)


if __name__ == "__main__":
    unittest.main()
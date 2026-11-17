import joblib
import pandas as pd
import streamlit as st

from src.datos import cargar_datos
from src.entrenar_final import RUTA_MODELO
from src.explicaciones import crear_explicador, explicar_sesion


# Compartimos estas etiquetas entre el formulario y las explicaciones.
CAMPOS_CONTEO = {
    "Administrative": "Páginas administrativas visitadas",
    "Informational": "Páginas informativas visitadas",
    "ProductRelated": "Páginas de productos visitadas",
}

CAMPOS_DECIMALES = {
    "Administrative_Duration": "Tiempo en páginas administrativas",
    "Informational_Duration": "Tiempo en páginas informativas",
    "ProductRelated_Duration": "Tiempo en páginas de productos",
    "PageValues": "Valor de las páginas (PageValues)",
}

CAMPOS_TASAS = {
    "BounceRates": "Tasa de rebote",
    "ExitRates": "Tasa de salida",
    "SpecialDay": "Cercanía a una fecha especial",
}

CAMPOS_CATEGORICOS = {
    "Month": "Mes",
    "OperatingSystems": "Código del sistema operativo",
    "Browser": "Código del navegador",
    "Region": "Código de la región",
    "TrafficType": "Código del tipo de tráfico",
    "VisitorType": "Tipo de visitante",
}

ETIQUETAS = {}
ETIQUETAS.update(CAMPOS_CONTEO)
ETIQUETAS.update(CAMPOS_DECIMALES)
ETIQUETAS.update(CAMPOS_TASAS)
ETIQUETAS.update(CAMPOS_CATEGORICOS)
ETIQUETAS["Weekend"] = "Fin de semana"


@st.cache_resource
def obtener_modelo():
    """Carga el Pipeline entrenado para reutilizarlo."""

    return joblib.load(RUTA_MODELO)


@st.cache_resource
def obtener_explicador():
    """Prepara SHAP una vez para reutilizarlo en las predicciones."""

    return crear_explicador(obtener_modelo())


@st.cache_data
def obtener_datos():
    """Carga los datos para obtener las opciones de las categorías."""

    return cargar_datos()


def mostrar_formulario(datos):
    """Recoge las características de una sesión."""

    entradas = {}

    with st.form("formulario_sesion"):
        izquierda, derecha = st.columns(2)

        with izquierda:
            st.subheader("Navegación")

            # Los recuentos admiten únicamente enteros no negativos.
            for columna, etiqueta in CAMPOS_CONTEO.items():
                valor_inicial = 1 if columna == "ProductRelated" else 0

                entradas[columna] = st.number_input(
                    etiqueta,
                    min_value=0,
                    value=valor_inicial,
                    step=1,
                )

            # Los tiempos y el valor de las páginas admiten decimales.
            for columna, etiqueta in CAMPOS_DECIMALES.items():
                entradas[columna] = st.number_input(
                    etiqueta,
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                )

        with derecha:
            st.subheader("Otros datos de la sesión")

            # Las tasas y la cercanía a una fecha especial van de 0 a 1.
            for columna, etiqueta in CAMPOS_TASAS.items():
                entradas[columna] = st.number_input(
                    f"{etiqueta} (0 a 1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    format="%.4f",
                )

            # Ofrecemos las categorías del dataset para evitar
            # errores al escribir nombres o códigos.
            for columna, etiqueta in CAMPOS_CATEGORICOS.items():
                opciones = sorted(datos[columna].dropna().unique().tolist())
                entradas[columna] = st.selectbox(etiqueta, options=opciones)

            entradas["Weekend"] = st.checkbox(
                "La visita ocurre en fin de semana"
            )

        enviado = st.form_submit_button("Estimar compra")

    return entradas, enviado


def predecir_sesion(modelo, entradas):
    """Devuelve la probabilidad, la clasificación y la sesión preparada."""

    # Conservamos el orden de las columnas utilizado al entrenar.
    sesion = pd.DataFrame([entradas])
    sesion = sesion.loc[:, modelo.feature_names_in_]

    if sesion.isna().any().any():
        raise ValueError("Completa todos los datos de la sesión.")

    indice_compra = list(modelo.classes_).index(True)
    probabilidad = modelo.predict_proba(sesion)[0, indice_compra]
    prediccion = modelo.predict(sesion)[0]

    # Devolvemos también la sesión para explicar exactamente
    # los mismos datos que acabamos de predecir.
    return float(probabilidad), bool(prediccion), sesion


def agrupar_contribuciones(modelo, tabla):
    """Suma las contribuciones de las categorías de cada campo original."""

    filas = []

    for columna in modelo.feature_names_in_:
        es_categoria = (
            columna in CAMPOS_CATEGORICOS or columna == "Weekend"
        )

        if es_categoria:
            # Por ejemplo, sumamos Month_May, Month_Nov y las demás
            # columnas de meses para mostrar una sola contribución: Mes.
            mascara = tabla["Variable"].str.startswith(f"{columna}_")
        else:
            # Las variables numéricas ya tienen una única columna.
            # Buscamos el nombre exacto para evitar mezclar campos.
            mascara = tabla["Variable"] == columna

        contribucion = tabla.loc[mascara, "Contribucion_pp"].sum()

        filas.append(
            {
                "Variable": ETIQUETAS[columna],
                "Contribucion_pp": contribucion,
            }
        )

    agrupada = pd.DataFrame(filas)

    # La agrupación debe conservar la suma de todas las contribuciones.
    diferencia = abs(
        tabla["Contribucion_pp"].sum()
        - agrupada["Contribucion_pp"].sum()
    )

    if diferencia > 0.000001:
        raise ValueError("La agrupación ha cambiado la suma de contribuciones.")

    agrupada["Impacto_absoluto"] = agrupada["Contribucion_pp"].abs()
    agrupada = agrupada.sort_values("Impacto_absoluto", ascending=False)

    return agrupada[["Variable", "Contribucion_pp"]]


def mostrar_explicacion(referencia, tabla):
    """Muestra las contribuciones agrupadas en un gráfico."""

    st.subheader("Explicación de la predicción")
    st.write(f"Probabilidad de referencia del modelo: **{referencia:.1%}**.")
    st.write(
        "Las contribuciones positivas aumentan la probabilidad respecto "
        "a esa referencia. Las negativas la reducen."
    )

    # Mostramos los diez campos con mayor contribución en valor absoluto.
    principales = tabla.head(10).rename(
        columns={"Contribucion_pp": "Contribución (puntos porcentuales)"}
    )

    grafico = principales.set_index("Variable")
    st.bar_chart(grafico, horizontal=True, sort=False, height=420)

    st.caption(
        "Se muestran los diez campos con mayor contribución. "
        "Los demás también participan en la predicción."
    )

    # La tabla permite consultar todas las contribuciones.
    with st.expander("Ver todas las contribuciones"):
        completa = tabla.rename(
            columns={"Contribucion_pp": "Contribución (puntos porcentuales)"}
        )
        st.dataframe(completa.round(3), hide_index=True)


def main():
    """Muestra el formulario, la predicción y su explicación."""

    st.set_page_config(
        page_title="Predicción de compras online",
        layout="wide",
    )

    st.title("Predicción de compras online")
    st.write(
        "Introduce las características de una sesión para estimar "
        "su probabilidad de terminar en compra."
    )
    st.caption("Prototipo para simular sesiones con las variables del dataset.")

    try:
        modelo = obtener_modelo()
        datos = obtener_datos()
    except FileNotFoundError:
        st.error("Faltan los datos o el modelo entrenado para iniciar la aplicación.")
        st.stop()

    entradas, enviado = mostrar_formulario(datos)

    if enviado:
        try:
            with st.spinner("Calculando la predicción y su explicación..."):
                probabilidad, prediccion, sesion = predecir_sesion(
                    modelo,
                    entradas,
                )

                # SHAP comprueba internamente en nuestra función que
                # las contribuciones reconstruyen la predicción.
                explicador = obtener_explicador()
                referencia, _, tabla = explicar_sesion(
                    modelo,
                    explicador,
                    sesion,
                )

                agrupada = agrupar_contribuciones(modelo, tabla)

        except ValueError as error:
            st.error(str(error))

        else:
            st.subheader("Resultado")
            st.metric(
                "Probabilidad estimada de compra",
                f"{probabilidad * 100:.1f} %",
            )

            if prediccion:
                st.success("El modelo clasifica la sesión como compra.")
            else:
                st.info("El modelo clasifica la sesión como no compra.")

            st.caption("Clasificación con un umbral del 50 %.")
            mostrar_explicacion(referencia, agrupada)

    with st.expander("Ver ejemplos del dataset"):
        st.write(f"Número de sesiones: {len(datos)}")
        st.dataframe(datos.head())


if __name__ == "__main__":
    main()
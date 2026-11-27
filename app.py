import numpy as np
import shap
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from interfaz_priorizacion import mostrar_priorizacion


# CONFIGURACIÓN

RAIZ = Path(__file__).resolve().parent
RUTA_MODELO = RAIZ / "models" / "random_forest_final.joblib"

st.set_page_config(
    page_title="NEXUS | Conversion Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ruta_css = RAIZ / "assets" / "styles.css"

if ruta_css.exists():
    st.markdown(
        f"<style>{ruta_css.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


# CARGA DEL MODELO

@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


if not RUTA_MODELO.exists():
    st.error("No se encuentra el modelo definitivo.")
    st.stop()

try:
    artefacto = cargar_modelo()
    modelo = artefacto["modelo"]
    umbral = float(artefacto["umbral"])
except Exception as error:
    st.error(f"No se ha podido cargar el modelo: {error}")
    st.stop()


# CATEGORÍAS DEL MODELO

def obtener_categorias(pipeline):
    """Recupera las categorías aprendidas por el preprocesador."""

    opciones = {
        "Month": [
            "Feb", "Mar", "May", "June", "Jul",
            "Aug", "Sep", "Oct", "Nov", "Dec"
        ],
        "VisitorType": [
            "Returning_Visitor", "New_Visitor", "Other"
        ],
        "OperatingSystems": list(range(1, 9)),
        "Browser": list(range(1, 14)),
        "Region": list(range(1, 10)),
        "TrafficType": list(range(1, 21)),
    }

    preprocesador = next(
        (
            paso
            for paso in pipeline.named_steps.values()
            if isinstance(paso, ColumnTransformer)
        ),
        None,
    )

    if preprocesador is None:
        return opciones

    for _, transformador, columnas in preprocesador.transformers_:
        if not hasattr(transformador, "categories_"):
            continue

        for columna, categorias in zip(
            columnas, transformador.categories_
        ):
            if columna in opciones:
                opciones[columna] = [
                    valor.item() if hasattr(valor, "item") else valor
                    for valor in categorias
                ]

    return opciones


opciones = obtener_categorias(modelo)

nombres_meses = {
    "Feb": "Febrero",
    "Mar": "Marzo",
    "May": "Mayo",
    "June": "Junio",
    "Jul": "Julio",
    "Aug": "Agosto",
    "Sep": "Septiembre",
    "Oct": "Octubre",
    "Nov": "Noviembre",
    "Dec": "Diciembre",
}

nombres_visitantes = {
    "Returning_Visitor": "Visitante recurrente",
    "New_Visitor": "Visitante nuevo",
    "Other": "Otro",
}



# EJEMPLOS PARA LA DEMOSTRACIÓN


valores_iniciales = {
    "Administrative": 3,
    "Administrative_Duration": 75.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 18,
    "ProductRelated_Duration": 700.0,
    "BounceRates_pct": 3.0,
    "ExitRates_pct": 5.0,
    "PageValues": 0.0,
    "SpecialDay": 0.0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": False,
}

ejemplos = {
    "alto": {
        **valores_iniciales,
        "ProductRelated": 42,
        "ProductRelated_Duration": 1850.0,
        "BounceRates_pct": 1.0,
        "ExitRates_pct": 2.0,
        "PageValues": 35.0,
    },
    "bajo": {
        **valores_iniciales,
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "ProductRelated": 3,
        "ProductRelated_Duration": 65.0,
        "BounceRates_pct": 30.0,
        "ExitRates_pct": 40.0,
        "PageValues": 0.0,
        "Month": "May",
    },
}

# Evitar categorías que no existan en el modelo entrenado.
for ejemplo in [valores_iniciales, *ejemplos.values()]:
    for campo, categorias in opciones.items():
        if ejemplo[campo] not in categorias:
            ejemplo[campo] = categorias[0]

for campo, valor in valores_iniciales.items():
    st.session_state.setdefault(campo, valor)


def cargar_ejemplo(nombre):
    st.session_state.update(ejemplos[nombre])
    st.session_state.pop("resultado", None)


# PREPARACIÓN DE ENTRADAS

def construir_sesion():
    """Convierte los controles del formulario en las 17 entradas."""

    datos = {
        "Administrative": st.session_state["Administrative"],
        "Administrative_Duration": st.session_state[
            "Administrative_Duration"
        ],
        "Informational": st.session_state["Informational"],
        "Informational_Duration": st.session_state[
            "Informational_Duration"
        ],
        "ProductRelated": st.session_state["ProductRelated"],
        "ProductRelated_Duration": st.session_state[
            "ProductRelated_Duration"
        ],
        "BounceRates": st.session_state["BounceRates_pct"] / 100,
        "ExitRates": st.session_state["ExitRates_pct"] / 100,
        "PageValues": st.session_state["PageValues"],
        "SpecialDay": st.session_state["SpecialDay"],
        "Month": st.session_state["Month"],
        "OperatingSystems": st.session_state["OperatingSystems"],
        "Browser": st.session_state["Browser"],
        "Region": st.session_state["Region"],
        "TrafficType": st.session_state["TrafficType"],
        "VisitorType": st.session_state["VisitorType"],
        "Weekend": st.session_state["Weekend"],
    }

    # Mantener exactamente el orden usado durante el entrenamiento.
    columnas = list(modelo.feature_names_in_)

    return pd.DataFrame([datos], columns=columnas)


def crear_indicador(probabilidad):
    """Construye el indicador visual de probabilidad."""

    figura = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probabilidad * 100,
            number={
                "suffix": " %",
                "valueformat": ".1f",
                "font": {
                    "size": 52,
                    "color": "#F0F7FF",
                },
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#8193A7",
                    "tickfont": {"color": "#9BAEC2"},
                },
                "bar": {
                    "color": "#53D8BE",
                    "thickness": 0.25,
                },
                "bgcolor": "#26394D",
                "borderwidth": 0,
                "threshold": {
                    "line": {
                        "color": "#E8AC70",
                        "width": 4,
                    },
                    "thickness": 0.8,
                    "value": umbral * 100,
                },
            },
        )
    )

    figura.update_layout(
        height=275,
        margin=dict(l=20, r=20, t=35, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#D5E0EC"},
    )

    return figura


# EXPLICACIONES SHAP

@st.cache_resource(show_spinner=False)
def cargar_explicador(_bosque):
    """Prepara Tree SHAP sobre el Random Forest entrenado."""

    return shap.TreeExplainer(
        _bosque,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )


def explicar_sesion(sesion, probabilidad):
    """Calcula SHAP y agrupa las contribuciones por variable original."""

    # Separar preprocesamiento y clasificador.
    preprocesador = modelo[:-1]
    bosque = modelo.steps[-1][1]

    # Aplicar exactamente las transformaciones del modelo.
    datos_transformados = preprocesador.transform(sesion)

    if hasattr(datos_transformados, "toarray"):
        datos_transformados = datos_transformados.toarray()

    datos_transformados = np.asarray(
        datos_transformados,
        dtype=np.float64,
    )

    explicador = cargar_explicador(bosque)
    valores = explicador.shap_values(datos_transformados)

    # SHAP puede devolver una lista (versiones antiguas)
    # o un array tridimensional (versiones actuales).
    if isinstance(valores, list):
        contribuciones = np.asarray(valores[1])[0]
    else:
        valores = np.asarray(valores)

        if valores.ndim != 3 or valores.shape[-1] != 2:
            raise ValueError(
                f"Formato SHAP inesperado: {valores.shape}"
            )

        contribuciones = valores[0, :, 1]

    referencia = float(
        np.asarray(explicador.expected_value).reshape(-1)[1]
    )

    # Comprobar que SHAP reconstruye la probabilidad.
    reconstruccion = referencia + contribuciones.sum()

    if not np.isclose(
        reconstruccion,
        probabilidad,
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError(
            "La explicación SHAP no coincide con "
            "la probabilidad predicha."
        )

    # Recuperar los nombres después del One-Hot Encoding.
    nombres_transformados = [
        nombre.split("__", 1)[-1]
        for nombre in preprocesador.get_feature_names_out()
    ]

    variables_categoricas = {
        "Month", "OperatingSystems", "Browser",
        "Region", "TrafficType", "VisitorType", "Weekend"
    }

    resultados = []
    indices_utilizados = []

    for variable in sesion.columns:

        if variable in variables_categoricas:
            indices = [
                i
                for i, nombre in enumerate(nombres_transformados)
                if nombre.startswith(f"{variable}_")
            ]
        else:
            indices = [
                i
                for i, nombre in enumerate(nombres_transformados)
                if nombre == variable
            ]

        if not indices:
            raise ValueError(
                f"No se encontraron columnas para {variable}."
            )

        indices_utilizados.extend(indices)

        resultados.append({
            "Variable": variable,
            "Contribucion": float(
                contribuciones[indices].sum()
            ),
        })

    # Ninguna característica transformada puede quedar sin asignar.
    if sorted(indices_utilizados) != list(
        range(len(nombres_transformados))
    ):
        raise ValueError(
            "Error al agrupar las características transformadas."
        )

    tabla = pd.DataFrame(resultados)

    return referencia, tabla


# NAVEGACIÓN

with st.sidebar:
    st.markdown("## ◈ NEXUS")
    st.caption("CONVERSION INTELLIGENCE")

    st.divider()

    st.markdown("##### ESPACIO DE TRABAJO")

    herramienta = st.radio(
        "Seleccionar herramienta",
        [
            "Analizar sesión",
            "Priorizar sesiones",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        """
        <div class="model-status">
            <span class="status-dot"></span>
            Motor predictivo disponible
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Predicciones generadas con Random Forest."
    )


# Mostrar la herramienta seleccionada.
# Pasamos además SHAP a la otra pantalla sin duplicar su implementación.
if herramienta == "Priorizar sesiones":
    mostrar_priorizacion(modelo, umbral, explicar_sesion)
    st.stop()



# CABECERA


st.markdown(
    """
    <div class="hero hero-compact">
        <div class="eyebrow">ANÁLISIS INTELIGENTE DE SESIONES</div>
        <h1>Anticipa la intención de compra.</h1>
        <p>
            Analiza el comportamiento de un visitante, obtén una
            estimación de compra y descubre qué información
            influye en el resultado.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Análisis individual")
st.caption(
    "Configura una sesión de navegación o utiliza un ejemplo para "
    "explorar cómo responde el sistema."
)


# EJEMPLOS RÁPIDOS

col_ej1, col_ej2, espacio = st.columns([1, 1, 1.7])

with col_ej1:
    st.button(
        "↗ Ejemplo: interés alto",
        on_click=cargar_ejemplo,
        args=("alto",),
        width="stretch",
    )

with col_ej2:
    st.button(
        "↘ Ejemplo: interés bajo",
        on_click=cargar_ejemplo,
        args=("bajo",),
        width="stretch",
    )

st.caption(
    "Los ejemplos son sesiones ilustrativas, no observaciones reales."
)


# FORMULARIO Y RESULTADOS

col_formulario, col_resultado = st.columns(
    [1.55, 1],
    gap="large",
)

with col_formulario:

    with st.form("formulario_prediccion"):

        tab_navegacion, tab_contexto = st.tabs(
            ["◈ Comportamiento", "▦ Contexto de la visita"]
        )

        # COMPORTAMIENTO

        with tab_navegacion:

            st.markdown("#### Actividad de navegación")
            st.caption(
                "Número de páginas visitadas y tiempo dedicado "
                "a cada tipo de contenido."
            )

            c1, c2 = st.columns(2)

            with c1:
                st.number_input(
                    "Páginas de producto",
                    min_value=0,
                    step=1,
                    key="ProductRelated",
                )

                st.number_input(
                    "Páginas administrativas",
                    min_value=0,
                    step=1,
                    key="Administrative",
                )

                st.number_input(
                    "Páginas informativas",
                    min_value=0,
                    step=1,
                    key="Informational",
                )

            with c2:
                st.number_input(
                    "Tiempo en productos (s)",
                    min_value=0.0,
                    step=10.0,
                    key="ProductRelated_Duration",
                )

                st.number_input(
                    "Tiempo administrativo (s)",
                    min_value=0.0,
                    step=10.0,
                    key="Administrative_Duration",
                )

                st.number_input(
                    "Tiempo informativo (s)",
                    min_value=0.0,
                    step=10.0,
                    key="Informational_Duration",
                )

            st.divider()

            st.markdown("#### Indicadores de comportamiento")
            st.caption(
                "Métricas de interacción registradas durante la sesión."
            )

            c3, c4 = st.columns(2)

            with c3:
                st.number_input(
                    "Tasa de rebote (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    key="BounceRates_pct",
                    help="Se transforma a un valor entre 0 y 1.",
                )

                st.number_input(
                    "Valor de página (PageValues)",
                    min_value=0.0,
                    step=1.0,
                    key="PageValues",
                    help=(
                        "Indicador de valor de página del conjunto "
                        "de datos. Su disponibilidad en tiempo real "
                        "debe verificarse."
                    ),
                )

            with c4:
                st.number_input(
                    "Tasa de salida (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    key="ExitRates_pct",
                    help="Se transforma a un valor entre 0 y 1.",
                )

                st.slider(
                    "Proximidad a fecha comercial",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.2,
                    key="SpecialDay",
                    help="Corresponde a la variable SpecialDay.",
                )

        # CONTEXTO

        with tab_contexto:

            st.markdown("#### Perfil del visitante")
            st.caption(
                "Características de la visita y del canal de acceso."
            )

            c1, c2 = st.columns(2)

            with c1:
                st.selectbox(
                    "Mes",
                    options=opciones["Month"],
                    format_func=lambda x: nombres_meses.get(x, str(x)),
                    key="Month",
                )

                st.selectbox(
                    "Tipo de visitante",
                    options=opciones["VisitorType"],
                    format_func=lambda x: nombres_visitantes.get(
                        x, str(x)
                    ),
                    key="VisitorType",
                )

                st.selectbox(
                    "Sistema operativo",
                    options=opciones["OperatingSystems"],
                    key="OperatingSystems",
                )

            with c2:
                st.selectbox(
                    "Navegador",
                    options=opciones["Browser"],
                    key="Browser",
                )

                st.selectbox(
                    "Región",
                    options=opciones["Region"],
                    key="Region",
                )

                st.selectbox(
                    "Tipo de tráfico",
                    options=opciones["TrafficType"],
                    key="TrafficType",
                )

            st.checkbox(
                "La sesión se produce en fin de semana",
                key="Weekend",
            )

            st.info(
                "Los códigos de sistema operativo, navegador, "
                "región y tráfico mantienen la codificación "
                "del conjunto de datos original."
            )

        st.markdown("")

        enviado = st.form_submit_button(
            "Analizar intención de compra",
            type="primary",
            width="stretch",
        )

    if enviado:
        try:
            sesion = construir_sesion()

            probabilidad = float(
                modelo.predict_proba(sesion)[0, 1]
            )

            st.session_state["resultado"] = {
                "probabilidad": probabilidad,
                "clasificacion": probabilidad >= umbral,
                "sesion": sesion,
            }

            # Calcular SHAP para la misma sesión.
            try:
                referencia, contribuciones = explicar_sesion(
                    sesion,
                    probabilidad,
                )

                st.session_state["resultado"].update({
                    "referencia_shap": referencia,
                    "contribuciones_shap": contribuciones,
                })

            except Exception as error_shap:
                st.session_state["resultado"]["error_shap"] = str(
                    error_shap
                )

        except Exception as error:
            st.error(f"Error al analizar la sesión: {error}")


# PANEL DE RESULTADOS

with col_resultado:

    st.markdown("### Resultado del análisis")

    resultado = st.session_state.get("resultado")

    if resultado is None:

        st.markdown(
            """
            <div class="empty-result">
                <div class="empty-icon">◎</div>
                <h3>Listo para analizar</h3>
                <p>
                    Introduce los datos de navegación y pulsa
                    <strong>Analizar intención de compra</strong>
                    para obtener una predicción.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        probabilidad = resultado["probabilidad"]
        clasificacion = resultado["clasificacion"]

        st.markdown(
            '<div class="result-heading">PROBABILIDAD ESTIMADA DE COMPRA</div>',
            unsafe_allow_html=True,
        )

        figura = crear_indicador(probabilidad)

        st.plotly_chart(
            figura,
            width="stretch",
            config={"displayModeBar": False},
        )

        if clasificacion:
            st.markdown(
                """
                <div class="prediction-positive">
                    <strong>↗ Intención de compra detectada</strong>
                    <p>
                        La puntuación supera el umbral de clasificación
                        establecido para el modelo.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="prediction-negative">
                    <strong>↘ Intención de compra no detectada</strong>
                    <p>
                        La puntuación queda por debajo del umbral
                        de clasificación establecido.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Probabilidad estimada",
                f"{probabilidad:.1%}",
            )

        with c2:
            st.metric(
                "Umbral de decisión",
                f"{umbral:.1%}",
            )

        st.divider()

        st.markdown("#### Interpretación de la predicción")
        st.write(
            "Esta puntuación resume la estimación del modelo "
            "para el comportamiento introducido. "
            "No representa una garantía de compra."
        )

        with st.expander("Ver los datos utilizados"):
            st.dataframe(
                resultado["sesion"].T.rename(
                    columns={0: "Valor"}
                ).astype(str),
                width="stretch",
            )



# EXPLICACIÓN SHAP A ANCHO COMPLETO

resultado = st.session_state.get("resultado")

if resultado is not None:

    st.divider()
    st.markdown("### ¿Qué explica esta predicción?")

    if "error_shap" in resultado:
        st.warning("No se ha podido calcular la explicación SHAP.")
        st.code(resultado["error_shap"])

    elif "contribuciones_shap" in resultado:

        referencia = resultado["referencia_shap"]
        probabilidad = resultado["probabilidad"]
        tabla = resultado["contribuciones_shap"].copy()

        etiquetas = {
            "Administrative": "Páginas administrativas",
            "Administrative_Duration": "Tiempo administrativo",
            "Informational": "Páginas informativas",
            "Informational_Duration": "Tiempo informativo",
            "ProductRelated": "Páginas de producto",
            "ProductRelated_Duration": "Tiempo en productos",
            "BounceRates": "Tasa de rebote",
            "ExitRates": "Tasa de salida",
            "PageValues": "Valor de página",
            "SpecialDay": "Fecha comercial",
            "Month": "Mes",
            "OperatingSystems": "Sistema operativo",
            "Browser": "Navegador",
            "Region": "Región",
            "TrafficType": "Canal de tráfico",
            "VisitorType": "Tipo de visitante",
            "Weekend": "Fin de semana",
        }

        tabla["Etiqueta"] = tabla["Variable"].map(etiquetas)
        tabla["Puntos"] = tabla["Contribucion"] * 100

        # Seleccionar las ocho mayores contribuciones.
        indices = tabla["Puntos"].abs().nlargest(8).index

        principales = (
            tabla.loc[indices]
            .sort_values("Puntos")
        )

        colores = [
            "#53D8BE" if valor >= 0 else "#E8AC70"
            for valor in principales["Puntos"]
        ]

        # GRÁFICO


        fig_shap = go.Figure()

        fig_shap.add_trace(
            go.Bar(
                x=principales["Puntos"],
                y=principales["Etiqueta"],
                orientation="h",
                marker_color=colores,
                text=[
                    f"{valor:+.1f} pp"
                    for valor in principales["Puntos"]
                ],
                textposition="outside",
                textfont=dict(size=14),
                cliponaxis=False,
                hovertemplate=(
                    "%{y}<br>"
                    "Contribución: %{x:+.2f} pp"
                    "<extra></extra>"
                ),
            )
        )

        fig_shap.add_vline(
            x=0,
            line_color="#93A9BD",
            line_width=1,
        )

        fig_shap.update_layout(
            height=520,
            margin=dict(l=50, r=110, t=25, b=55),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#CBD5E1",
                size=14,
            ),
            showlegend=False,
            xaxis=dict(
                title="Contribución (puntos porcentuales)",
                gridcolor="#293C50",
                zeroline=False,
            ),
            yaxis=dict(
                title=None,
                automargin=True,
            ),
        )

        st.plotly_chart(
            fig_shap,
            width="stretch",
            config={"displayModeBar": False},
        )

        st.caption(
            "Turquesa: aumenta la probabilidad estimada. "
            "Naranja: la reduce. Las contribuciones se "
            "calculan respecto a la referencia SHAP."
        )

        st.info(
            f"**Probabilidad de referencia:** {referencia:.1%}  "
            f"· **Variación de esta sesión:** "
            f"{(probabilidad - referencia) * 100:+.1f} pp"
        )

        # TABLA COMPLETA

        with st.expander("Ver las 17 contribuciones"):

            tabla_completa = tabla.copy()

            tabla_completa["Contribución"] = (
                tabla_completa["Puntos"]
                .map(lambda valor: f"{valor:+.2f} pp")
            )

            tabla_completa = tabla_completa.reindex(
                tabla_completa["Puntos"].abs()
                .sort_values(ascending=False)
                .index
            )

            st.dataframe(
                tabla_completa[
                    ["Etiqueta", "Contribución"]
                ].rename(
                    columns={"Etiqueta": "Variable"}
                ),
                hide_index=True,
                width="stretch",
            )

# NOTAS

st.divider()

st.caption(
    "NEXUS · Conversion Intelligence | "
    "Herramienta de apoyo al análisis de sesiones. "
    "La disponibilidad de PageValues debe verificarse "
    "antes de utilizar estas predicciones en tiempo real."
)
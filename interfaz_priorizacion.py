import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# VALIDACIÓN DE LOS DATOS

CATEGORICAS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

ENTERAS = [
    "Administrative",
    "Informational",
    "ProductRelated",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

TASAS = ["BounceRates", "ExitRates", "SpecialDay"]

MESES = {
    "Feb", "Mar", "May", "June", "Jul",
    "Aug", "Sep", "Oct", "Nov", "Dec",
}

VISITANTES = {
    "Returning_Visitor", "New_Visitor", "Other",
}


def validar_csv(datos, modelo):
    """Comprueba las entradas y las prepara para el modelo."""

    if datos.empty:
        raise ValueError("El archivo no contiene sesiones.")

    if len(datos) > 50000:
        raise ValueError(
            "El archivo supera el límite de 50.000 sesiones."
        )

    columnas = list(modelo.feature_names_in_)

    faltantes = [
        columna
        for columna in columnas
        if columna not in datos.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas obligatorias: "
            + ", ".join(faltantes)
        )

    if datos.columns.duplicated().any():
        raise ValueError(
            "El archivo contiene nombres de columnas repetidos."
        )

    # Solo estas 17 características se utilizarán al predecir.
    X = datos[columnas].copy()

    if X.isna().any().any():
        raise ValueError(
            "Hay valores vacíos en las columnas necesarias."
        )

    numericas = [
        columna
        for columna in columnas
        if columna not in CATEGORICAS
        or columna in [
            "OperatingSystems",
            "Browser",
            "Region",
            "TrafficType",
        ]
    ]

    for columna in numericas:
        valores = pd.to_numeric(
            X[columna],
            errors="coerce",
        )

        if (
            valores.isna().any()
            or not np.isfinite(valores.to_numpy()).all()
        ):
            raise ValueError(
                f"La columna {columna} contiene valores "
                "no numéricos o no finitos."
            )

        if (valores < 0).any():
            raise ValueError(
                f"La columna {columna} contiene valores negativos."
            )

        if columna in ENTERAS:
            if (valores % 1 != 0).any():
                raise ValueError(
                    f"La columna {columna} requiere números enteros."
                )

            X[columna] = valores.astype(int)
        else:
            X[columna] = valores.astype(float)

    for columna in TASAS:
        if (X[columna] > 1).any():
            raise ValueError(
                f"La columna {columna} debe estar entre 0 y 1."
            )

    # Categorías textuales del dataset.
    X["Month"] = X["Month"].astype(str).str.strip()
    X["VisitorType"] = X["VisitorType"].astype(str).str.strip()

    if not X["Month"].isin(MESES).all():
        raise ValueError(
            "Month contiene meses no reconocidos."
        )

    if not X["VisitorType"].isin(VISITANTES).all():
        raise ValueError(
            "VisitorType contiene categorías no reconocidas."
        )

    # Aceptar True/False y 1/0 en la columna Weekend.
    valores_weekend = (
        X["Weekend"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    conversion = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    if not valores_weekend.isin(conversion).all():
        raise ValueError(
            "Weekend debe contener True/False o 1/0."
        )

    X["Weekend"] = valores_weekend.map(conversion).astype(bool)

    return X[columnas]


# PRIORIZACIÓN

def calcular_priorizacion(datos, modelo, umbral):
    """Calcula probabilidades y ordena las sesiones."""

    X = validar_csv(datos, modelo)

    probabilidades = modelo.predict_proba(X)[:, 1]

    # Conservar los datos originales y su posición.
    # Revenue nunca se utiliza como entrada ni se exporta.
    resultados = (
        datos.drop(columns=["Revenue"], errors="ignore")
        .reset_index(drop=True)
        .copy()
    )

    resultados.insert(
        0,
        "Fila_original",
        np.arange(1, len(resultados) + 1),
    )

    resultados["Probabilidad_compra"] = probabilidades

    resultados["Probabilidad (%)"] = (
        probabilidades * 100
    )

    resultados["Clasificacion"] = np.where(
        probabilidades >= umbral,
        "Compra probable",
        "Compra no probable",
    )

    resultados = (
        resultados
        .sort_values(
            "Probabilidad_compra",
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    resultados.insert(
        0,
        "Prioridad",
        np.arange(1, len(resultados) + 1),
    )

    return resultados


# INTERFAZ

def mostrar_priorizacion(modelo, umbral, explicar_sesion):
    """Muestra la herramienta de priorización masiva."""

    st.markdown(
        """
        <div class="hero hero-compact">
            <div class="eyebrow">INTELIGENCIA COMERCIAL</div>
            <h1>Prioriza las sesiones con mayor intención de compra.</h1>
            <p>
                Analiza múltiples visitas, identifica las sesiones
                con mayor probabilidad de compra y convierte
                los datos en una lista de prioridades.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Importar sesiones")

    st.caption(
        "Sube un archivo CSV que contenga las 17 variables "
        "utilizadas por el modelo."
    )

    archivo = st.file_uploader(
        "Seleccionar archivo CSV",
        type=["csv"],
        help=(
            "El archivo puede incluir columnas adicionales. "
            "Revenue se ignora porque representa el resultado real."
        ),
    )

    if archivo is None:
        st.info(
            "Selecciona un CSV para comenzar. Puedes probar con "
            "el archivo original online_shoppers_intention.csv."
        )

        with st.expander("¿Qué debe contener el CSV?"):
            st.write(
                "El archivo debe incluir estas 17 columnas:"
            )
            st.code(
                ", ".join(modelo.feature_names_in_),
                language=None,
            )

            st.caption(
                "Las tasas deben expresarse entre 0 y 1, "
                "no entre 0 y 100."
            )

        return

    # CARGAR Y PREDECIR

    try:
        with st.spinner("Analizando sesiones..."):
            datos = pd.read_csv(
                archivo,
                encoding="utf-8-sig",
            )

            resultados = calcular_priorizacion(
                datos,
                modelo,
                umbral,
            )

    except Exception as error:
        st.error(f"No se ha podido analizar el archivo: {error}")
        return

    st.success(
        f"Análisis completado: {len(resultados):,} sesiones procesadas."
    )

    # RESUMEN

    st.markdown("### Resumen del análisis")

    total = len(resultados)

    positivas = (
        resultados["Probabilidad_compra"] >= umbral
    ).sum()

    puntuacion_media = (
        resultados["Probabilidad_compra"].mean()
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Sesiones analizadas",
            f"{total:,}",
        )

    with c2:
        st.metric(
            "Por encima del umbral",
            f"{positivas:,}",
        )
        st.caption(f"{positivas / total:.1%} del total")

    with c3:
        st.metric(
            "Probabilidad media estimada",
            f"{puntuacion_media:.1%}",
        )

    st.caption(
        "Las clasificaciones y probabilidades son estimaciones "
        "del modelo, no compras observadas."
    )

    st.divider()

    # RANKING

    st.markdown("### Ranking de sesiones")

    st.caption(
        "Las sesiones están ordenadas por probabilidad "
        "estimada de compra, de mayor a menor."
    )

    # Filtros del ranking
    c1, c2 = st.columns(2)

    with c1:
        filtro_clasificacion = st.selectbox(
            "Clasificación",
            ["Todas", "Compra probable", "Compra no probable"],
        )

    with c2:
        probabilidad_minima = st.slider(
            "Probabilidad mínima (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
        )

    filtradas = resultados.copy()

    if filtro_clasificacion != "Todas":
        filtradas = filtradas[
            filtradas["Clasificacion"] == filtro_clasificacion
        ]

    filtradas = filtradas[
        filtradas["Probabilidad (%)"] >= probabilidad_minima
    ]

    porcentaje = st.slider(
        "Mostrar el porcentaje superior de las sesiones filtradas",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        format="%d%%",
    )

    k = int(np.ceil(len(filtradas) * porcentaje / 100))
    seleccion = filtradas.head(k)

    st.markdown(
        f"**Mostrando {len(seleccion):,} de "
        f"{len(filtradas):,} sesiones que cumplen los filtros.**"
    )

    tabla = seleccion[
        [
            "Prioridad",
            "Fila_original",
            "Probabilidad (%)",
            "Clasificacion",
        ]
    ]

    st.dataframe(
        tabla,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Prioridad": "Prioridad",
            "Fila_original": "Fila original",
            "Probabilidad (%)": st.column_config.ProgressColumn(
                "Probabilidad de compra",
                min_value=0,
                max_value=100,
                format="%.1f%%",
                color="#53D8BE",
            ),
            "Clasificacion": "Clasificación",
        },
    )

    # EXPLICACIÓN INDIVIDUAL DESDE EL RANKING

    st.markdown("#### Explicar una sesión del ranking")

    st.caption(
        "Selecciona una sesión de las mostradas para conocer "
        "qué características influyen en su predicción."
    )

    modo_busqueda = st.radio(
        "Buscar sesión por",
        ["Prioridad", "Fila original"],
        horizontal=True,
    )

    analizar = False
    prioridad = None
    fila_buscada = None

    if modo_busqueda == "Prioridad" and seleccion.empty:
        st.info("No hay sesiones que cumplan los filtros actuales.")

    else:
        with st.form("form_explicacion_ranking"):

            if modo_busqueda == "Prioridad":
                prioridad = st.selectbox(
                    "Sesión",
                    options=seleccion["Prioridad"].tolist(),
                    format_func=lambda p: (
                        f"Prioridad {p} · "
                        f"Fila {resultados.iloc[p - 1]['Fila_original']} · "
                        f"Probabilidad "
                        f"{resultados.iloc[p - 1]['Probabilidad_compra']:.1%}"
                    ),
                )

            else:
                fila_buscada = st.number_input(
                    "Número de fila original del CSV",
                    min_value=1,
                    max_value=int(total),
                    value=1,
                    step=1,
                )

            analizar = st.form_submit_button(
                "Ver explicación SHAP",
                use_container_width=True,
            )

    if analizar:
        try:
            if modo_busqueda == "Fila original":
                fila = resultados.loc[
                    resultados["Fila_original"] == fila_buscada
                ]
            else:
                fila = resultados.iloc[[prioridad - 1]]

            if fila.empty:
                raise ValueError("No se ha encontrado la sesión.")

            st.caption(
                f"Analizando prioridad {int(fila['Prioridad'].iloc[0])} "
                f"· Fila original {int(fila['Fila_original'].iloc[0])}"
            )

            sesion = validar_csv(fila, modelo)
            probabilidad = float(
                fila["Probabilidad_compra"].iloc[0]
            )

            referencia, contribuciones = explicar_sesion(
                sesion, probabilidad
            )

            st.metric(
                "Probabilidad estimada de compra",
                f"{probabilidad:.1%}",
            )

            st.caption(
                f"Probabilidad de referencia: {referencia:.1%} · "
                f"Variación: "
                f"{(probabilidad - referencia) * 100:+.1f} pp"
            )

            tabla_shap = contribuciones.copy()
            tabla_shap["Puntos"] = (
                tabla_shap["Contribucion"] * 100
            )

            principales = tabla_shap.loc[
                tabla_shap["Puntos"].abs().nlargest(8).index
            ].sort_values("Puntos")

            figura_shap = go.Figure(
                go.Bar(
                    x=principales["Puntos"],
                    y=principales["Variable"],
                    orientation="h",
                    marker_color=[
                        "#53D8BE" if valor >= 0 else "#E8AC70"
                        for valor in principales["Puntos"]
                    ],
                    text=[
                        f"{valor:+.1f} pp"
                        for valor in principales["Puntos"]
                    ],
                    textposition="outside",
                    cliponaxis=False,
                )
            )

            figura_shap.add_vline(
                x=0,
                line_color="#93A9BD",
            )

            figura_shap.update_layout(
                height=460,
                margin=dict(l=40, r=90, t=25, b=55),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1"),
                xaxis_title="Contribución (puntos porcentuales)",
                yaxis_title=None,
                showlegend=False,
            )

            st.plotly_chart(
                figura_shap,
                use_container_width=True,
            )

            with st.expander("Ver las 17 contribuciones"):
                st.dataframe(
                    tabla_shap[["Variable", "Puntos"]]
                    .sort_values("Puntos", ascending=False),
                    hide_index=True,
                    use_container_width=True,
                )

        except Exception as error:
            st.error(
                f"No se ha podido explicar la sesión: {error}"
            )

    # DISTRIBUCIÓN DE PUNTUACIONES

    st.markdown("### Distribución de probabilidades")

    figura = go.Figure()

    figura.add_trace(
        go.Histogram(
            x=resultados["Probabilidad (%)"],
            nbinsx=30,
            marker_color="#53D8BE",
            opacity=0.85,
            hovertemplate=(
                "Probabilidad: %{x:.1f}%"
                "<br>Sesiones: %{y}"
                "<extra></extra>"
            ),
        )
    )

    figura.add_vline(
        x=umbral * 100,
        line_color="#E8AC70",
        line_width=2,
        line_dash="dash",
    )

    figura.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=15, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1"),
        xaxis=dict(
            title="Probabilidad estimada de compra (%)",
            range=[0, 100],
            gridcolor="#293C50",
        ),
        yaxis=dict(
            title="Número de sesiones",
            gridcolor="#293C50",
        ),
        bargap=0.08,
    )

    st.plotly_chart(
        figura,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.caption(
        f"La línea naranja indica el umbral de clasificación "
        f"del {umbral:.1%}."
    )

    # DESCARGA

    st.divider()

    st.markdown("### Exportar resultados")

    st.write(
        "Descarga el ranking completo, incluyendo las "
        "características originales, la probabilidad estimada, "
        "la clasificación y la posición de prioridad."
    )

    archivo_salida = (
        resultados
        .to_csv(index=False)
        .encode("utf-8-sig")
    )

    st.download_button(
        label="Descargar sesiones priorizadas",
        data=archivo_salida,
        file_name="nexus_sesiones_priorizadas.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

    st.caption(
        "La prioridad indica probabilidad de compra estimada, "
        "no el efecto que tendría una intervención comercial. "
        "El uso en tiempo real requiere verificar la "
        "disponibilidad de PageValues."
    )
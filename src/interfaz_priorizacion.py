import pandas as pd
import streamlit as st

from src.priorizacion import priorizar_sesiones


def mostrar_priorizacion(modelo):
    """Permite cargar sesiones, ordenarlas y descargar los resultados."""

    st.header("Priorizar varias sesiones")
    st.write(
        "Sube un CSV con las características de las sesiones. "
        "Se mostrarán primero las de mayor probabilidad estimada de compra."
    )

    archivo = st.file_uploader(
        "Selecciona un archivo CSV separado por comas",
        type=["csv"],
    )

    # Mientras no haya un archivo, solo mostramos el control de carga.
    if archivo is None:
        return

    try:
        # Volvemos al principio para poder leer el archivo
        # también cuando Streamlit actualice la página.
        archivo.seek(0)
        sesiones = pd.read_csv(archivo, encoding="utf-8-sig")

        resultado = priorizar_sesiones(modelo, sesiones)

    except pd.errors.EmptyDataError:
        st.error("El archivo está vacío o no contiene una cabecera.")
        return

    except (pd.errors.ParserError, UnicodeDecodeError):
        st.error(
            "No se puede leer el CSV. Comprueba que está separado "
            "por comas y guardado con codificación UTF-8."
        )
        return

    except (ValueError, TypeError) as error:
        st.error(str(error))
        return

    # Preparamos una copia para descargar los resultados en porcentaje.
    # Conservamos también las características originales de cada sesión.
    descarga = resultado.rename(
        columns={"Probabilidad_compra": "Probabilidad_compra_pct"}
    ).copy()

    descarga["Probabilidad_compra_pct"] = (
        descarga["Probabilidad_compra_pct"] * 100
    )
    descarga.insert(0, "Prioridad", range(1, len(descarga) + 1))

    # En pantalla mostramos únicamente las columnas principales.
    vista = descarga[
        ["Prioridad", "Sesion", "Probabilidad_compra_pct", "Prediccion_compra"]
    ].copy()

    vista["Probabilidad_compra_pct"] = (
        vista["Probabilidad_compra_pct"].round(2)
    )
    vista["Prediccion_compra"] = vista["Prediccion_compra"].map(
        {True: "Compra", False: "No compra"}
    )

    vista = vista.rename(
        columns={
            "Sesion": "Sesión",
            "Probabilidad_compra_pct": "Probabilidad de compra (%)",
            "Prediccion_compra": "Clasificación",
        }
    )

    st.write(f"Sesiones procesadas: **{len(resultado)}**")
    st.dataframe(vista, hide_index=True)
    st.caption(
        "Sesión identifica la posición original de la fila en el archivo. "
        "Prioridad indica su posición después de ordenar."
    )

    # La descarga incluye los resultados y las características de las sesiones.
    contenido_csv = descarga.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "Descargar sesiones ordenadas",
        data=contenido_csv,
        file_name="sesiones_priorizadas.csv",
        mime="text/csv",
        on_click="ignore",
    )

import streamlit as st
import pandas as pd

st.title("Predicción de compras online")
st.write("Proyecto de Inteligencia Artificial en las Organizaciones")

# Cargamos el dataset
df = pd.read_csv("data/raw/online_shoppers_intention.csv")

st.subheader("Dataset utilizado")
st.write(f"Número de sesiones: {len(df)}")

st.dataframe(df.head())

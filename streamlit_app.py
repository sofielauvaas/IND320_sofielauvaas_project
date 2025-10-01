import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="IND320 - Sofie Lauvaas Project", layout="wide")

st.title("IND320 — Home Page")
st.markdown("""
Welcome to my IND320 project app.  
This app reads a local CSV file (`open-meteo-subset.csv`) and visualizes it.
Use the sidebar to navigate to different pages.
""")

# Quick preview of CSV if it exists
data_path = "data/open-meteo-subset.csv"

if not os.path.exists(data_path):
    st.error("CSV file not found! Please put `open-meteo-subset.csv` in the project folder.")

else:
    df = pd.read_csv(data_path)
    st.subheader("Preview of data")
    st.dataframe(df.head(10), use_container_width=True)

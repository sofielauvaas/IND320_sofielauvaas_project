import streamlit as st
import pandas as pd

st.title("📊 Data Table")

# Load CSV
@st.cache_data
def load_data():
    return pd.read_csv("data/open-meteo-subset.csv")

df = load_data()

st.subheader("Dataset preview")
st.dataframe(df, use_container_width=True)

# Sparklines (first 30 rows as a proxy for "first month")
st.subheader("Column sparklines (first month)")

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        st.line_chart(df[col].head(30), height=100)

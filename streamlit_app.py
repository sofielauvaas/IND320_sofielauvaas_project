import streamlit as st
import utilities.functions as functions 
import pandas as pd 

# PAGE CONFIGURATION
st.set_page_config(page_title="IND320 - Sofie Lauvaas Project", layout="wide")

# The sidebar is now empty for global controls on the home page.
st.sidebar.title("Data Selection")
st.sidebar.markdown("Use the pages to select analysis parameters.")

# MAIN PAGE CONTENT

st.title("IND320 - Home Page")
st.markdown("""
Welcome to my IND320 project app. This application analyzes **Elhub power production data** and **Open-Meteo weather data** for Norway.

The controls for selecting the **Price Area** and **Production Groups** are set on the **first data page** and are then maintained and accessible via the **sidebar** on all subsequent analysis pages.

Use the sidebar navigation to explore the different analyses.
""")

st.image("data/home_page.png", use_container_width=True)
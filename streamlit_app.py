import streamlit as st
import functions 
import pandas as pd 

# PAGE CONFIGURATION
st.set_page_config(page_title="IND320 - Sofie Lauvaas Project", layout="wide")

# The sidebar is now empty for global controls.
st.sidebar.title("Data Selection")
st.sidebar.markdown("Use the pages to select analysis parameters.")

# MAIN PAGE CONTENT

st.title("IND320 - Home Page")
st.markdown("""
Welcome to my IND320 project app.  
This application analyzes Elhub power production data and Open-Meteo weather data for Norway.

**IMPORTANT:** The weather analysis pages (3, 4, 5) use the **Price Area** selector from **Page 2 (Energy Decomposition)** to determine which city's weather data to download.

Use the sidebar navigation to explore the different analyses.
""")

st.image("data/home_page.png", use_container_width=True)
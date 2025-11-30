import streamlit as st

st.title("IND320 - Home Page")
st.markdown("""
Welcome to my IND320 project app. This application analyzes **Elhub power production data** 
and **Open-Meteo weather data** for Norway.

There is an app state control containing adjustable parameters, which is accessible globally via the **sidebar** on analysis pages.

Use the sidebar navigation to explore the different analyses.
""")

st.image("data/home_page.png", use_container_width=True)

import streamlit as st

st.set_page_config(page_title="IND320 - Sofie Lauvaas Project", layout="wide")

st.title("IND320 — Home Page")
st.markdown("""
Welcome to my IND320 project app.  
This app reads a local CSV file (`open-meteo-subset.csv`) and visualizes it.
Use the sidebar to navigate to different pages.
""")

# Display the image at the bottom
st.image("home_page.png", use_container_width=True)
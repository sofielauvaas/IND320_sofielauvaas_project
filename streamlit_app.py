import streamlit as st

st.set_page_config(page_title="IND320 - Sofie Lauvaas Project", layout="wide")

st.title("IND320 — Home Page")
st.markdown("""
Welcome to my IND320 project app.  
Use the sidebar to navigate to different pages.
""")

# Display the image at the bottom
st.image("data/home_page.png", use_container_width=True)
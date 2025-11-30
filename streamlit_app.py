import streamlit as st

# Page config
st.set_page_config(page_title="IND320 - Sofie Lauvaas Project", layout="wide")

# Hide default multipage sidebar
st.markdown("""
<style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Page groups
PAGE_GROUPS = {
    "🔌 Energy": [
        ("Energy Explorer", "pages/1_Energy_Explorer.py"),
        ("Time Series Analysis", "pages/2_Time_Series_Analysis.py"),
        ("Correlation Analysis", "pages/7_Correlation_Analysis.py"),
        ("Forecasting", "pages/8_Forecasting.py"),
    ],
    "🌦 Weather": [
        ("Weather Info", "pages/3_Weather_Info.py"),
        ("Weather Anomalies", "pages/4_Weather_Anomalies.py"),
    ],
    "❄️ Snow": [
        ("Map", "pages/5_Map.py"),
        ("Snow Drift", "pages/6_Snow_Drift.py"),
    ],
    "🎮 Fun": [
        ("Mini Games", "pages/9_Mini_games.py"),
    ],
}

# Cache pages
if "pages" not in st.session_state:
    cache = {}
    cache["home"] = st.Page("pages/0_Home.py", title="Home")
    for group, items in PAGE_GROUPS.items():
        cache[group] = [st.Page(path, title=title) for title, path in items]
    st.session_state["pages"] = cache

cache = st.session_state["pages"]

# Flatten pages for navigation
all_pages = [cache["home"]]
for group in PAGE_GROUPS:
    all_pages.extend(cache[group])

nav = st.navigation(all_pages)

# Sidebar
with st.sidebar:
    st.markdown("## Navigation 🧭")
    if st.button("🏠 Home", key="home_nav"):
        st.switch_page(cache["home"])
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    for group, items in PAGE_GROUPS.items():
        with st.expander(group):
            for i, (title, _) in enumerate(items):
                page_obj = cache[group][i]
                if st.button(title, key=f"{group}_{i}"):
                    st.switch_page(page_obj)

nav.run()

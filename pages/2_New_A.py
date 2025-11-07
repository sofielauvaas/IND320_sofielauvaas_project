import streamlit as st
# Import your custom control function and supporting utilities
from utilities.app_state import render_app_state_controls
from utilities import functions
import pandas as pd
import plotly.express as px
# NOTE: We assume 'functions.plot_spectrogram_elhub' exists and returns a Matplotlib figure,
# based on the original code's st.pyplot() usage and the function names you provided previously.

st.set_page_config(page_title="Energy Time Series Analysis", layout="wide")

# --- 1. RENDER GLOBAL CONTROLS IN SIDEBAR ---
with st.sidebar:
    render_app_state_controls()

# --- 2. ACCESS GLOBAL STATE ---
# Read the globally selected price area and groups from session_state
price_area = st.session_state.get("price_area")
globally_selected_groups = st.session_state.get("production_group")

# --- INITIAL CHECK ---
if not price_area or not globally_selected_groups:
    st.info("Please use the sidebar to select a Price Area and at least one Production Group.")
    st.stop()

st.title("Time Series Decomposition and Frequency Analysis")

# --- DATA LOADING ---
try:
    df_production = functions.load_data_from_mongo()
except Exception as e:
    st.error(f"Failed to load production data from MongoDB. Error: {e}")
    st.stop() 

if df_production.empty:
    st.warning("No production data found in the MongoDB collection.")
    st.stop()

# --- FILTERING LOGIC & CONTEXT DISPLAY ---

# The options for analysis are now determined by the global production group selection
analysis_options = globally_selected_groups

# Display the globally selected groups and area for context (Replacing st.subheader)
groups_text = ', '.join([g.capitalize() for g in analysis_options])
st.info(
    f"""
    **Analysis Scope** (by the sidebar configuartion):
    
    * **Price Area:** **{price_area}**
    * **Available Groups:** {groups_text}
    """
)

# --- LOCAL SELECTOR ---

# Selector for the single group required for STL/Spectrogram
st.markdown("##### Select Production Group for Detailed Analysis:")
selected_group_for_analysis = st.selectbox(
    "Group:", # Simple label
    analysis_options,
    index=0,
    label_visibility="collapsed" # Hide label to keep it clean
)

# --- TABS AND ANALYSIS ---
tab1, tab2 = st.tabs(["STL Decomposition", "Spectrogram"])


# TAB 1: STL Decomposition
with tab1:
    st.markdown(f"#### STL Decomposition: {selected_group_for_analysis.capitalize()} in {price_area}")
    
    col_period, col_seasonal, col_trend = st.columns(3)
    with col_period:
        period = st.number_input("Period (e.g., 168 for weekly)", min_value=1, value=168, step=1, key='stl_period')
    with col_seasonal:
        seasonal = st.number_input("Seasonal Smoothing (Odd)", min_value=3, value=9, step=2, key='stl_seasonal')
    with col_trend:
        trend = st.number_input("Trend Smoothing (Odd)", min_value=3, value=241, step=2, key='stl_trend')
    
    
    try:
        fig_stl = functions.stl_decomposition_elhub(
            df_production, # Pass full data for internal filtering flexibility
            price_area=price_area, 
            production_group=selected_group_for_analysis,
            period=period,
            seasonal=seasonal,
            trend=trend
        )
        st.plotly_chart(fig_stl, use_container_width=True)
    except Exception as e:
        st.error(f"Error during STL Decomposition. Please check parameters or data availability. Error: {e}")


# TAB 2: Spectrogram Analysis
with tab2:
    st.markdown(f"#### Spectrogram Analysis: {selected_group_for_analysis.capitalize()} in {price_area}")
    
    col_window, col_overlap = st.columns(2)
    with col_window:
        window_length = st.slider("Window Length (NPERSEG - hours)", min_value=64, max_value=512, value=256, step=64, key='spec_window')
        st.caption("Length of each segment for analysis.")
    with col_overlap:
        overlap = st.slider("Overlap (NOVERLAP - hours)", min_value=32, max_value=256, value=128, step=32, key='spec_overlap')
        st.caption("Number of overlapping samples between segments.")

    try:
        # Assuming functions.create_spectrogram is the correct name and returns a Matplotlib figure
        fig_spec = functions.create_spectrogram( 
            df_production, 
            price_area=price_area, 
            production_group=selected_group_for_analysis,
            window_length=window_length,
            overlap=overlap
        )
        st.pyplot(fig_spec, use_container_width=True) 
    except Exception as e:
        st.error(f"Error during Spectrogram analysis. Error: {e}")
import streamlit as st
import pandas as pd
import numpy as np
# Import your custom control function
from utilities.app_state import render_app_state_controls
from utilities import functions # This now holds all analysis functions
import requests # Needed for exception handling


st.set_page_config(
    page_title="Anomaly Detection",
    layout="wide"
)

# --- 1. RENDER GLOBAL CONTROLS IN SIDEBAR ---
with st.sidebar:
    render_app_state_controls()

# --- 2. ACCESS GLOBAL STATE AND DATA FETCHING ---

# Use the canonical key for the Price Area
selected_area = st.session_state.get('price_area')
# REMOVED: selected_groups = st.session_state.get('production_group', ['No groups selected']) 

if not selected_area:
    st.info("The global Price Area selector is not yet initialized. Please use the sidebar.")
    st.stop() 

# REMOVED: groups_text = ', '.join([g.capitalize() for g in selected_groups])

st.title("Anomaly and Outlier Detection")

# --- DISPLAY CONTEXT BOX (REVISED TEXT: Only Price Area) ---
st.info(
    f"""
    **Analysis Scope** (by the sidebar configuration):
    
    * **Price Area:** **{selected_area}** 
    """
)
# -----------------------------------------------------------

# Load data using the cached API function (Layer 1 Caching)
try:
    # Ensure functions.download_weather_data uses the correct mechanism to fetch data
    df = functions.download_weather_data(selected_area) 
    
    if df is None or df.empty:
        st.error("Failed to retrieve weather data or dataset is empty. Check API/network.")
        st.stop()
        
except Exception as e:
    st.error(f"Error loading weather data: {e}")
    st.stop()


# Reset index to make 'time' a column for passing to functions
df_ready = df.reset_index()


# --- 3. TABBED INTERFACE ---

tab1, tab2 = st.tabs(["Temperature Outliers (Robust SPC)", "Precipitation/Wind Anomalies (LOF)"])

with tab1:
    st.header("Temperature Outlier Analysis (Robust SPC)")
    st.caption("Robust Statistical Process Control (SPC) based on Seasonal-Adjusted Time Variation (SATV) using DCT filtering.")

    # SLIDERS for SPC Analysis
    col_a, col_b = st.columns(2)
    with col_a:
        freq_cutoff = st.slider(
            "DCT Low-Frequency Index Cutoff", 
            min_value=1, 
            max_value=250, 
            value=100, 
            step=1, 
            help="Controls how many low-frequency DCT indices are kept (smaller index = smoother trend, more variation isolated)."
        )
    with col_b:
        num_std = st.slider("Number of Robust Standard Deviations (k)", min_value=2.0, max_value=5.0, value=3.0, step=0.1, help="Defines the control limits (Center ± k * Robust Std. Dev.).")

    if 'temperature_2m' in df_ready.columns:
        # --- CALL FUNCTION FROM functions.py ---
        fig, summary = functions.temperature_spc_from_satv(
            df_ready["time"].values,
            df_ready["temperature_2m"].values,
            # Use the correct keyword argument 'keep_low_index'
            keep_low_index=freq_cutoff, 
            k=num_std
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Summary Statistics")
        st.write(f"**Total records analyzed:** {summary['n_total']}")
        st.write(f"**Outliers detected:** {summary['n_outliers']} ({summary['percent_outliers']}%)")
        st.write(f"**Robust Standard Deviation (SATV):** {summary['robust_std']:.3f} °C")

    else:
        st.warning("Temperature data (temperature_2m) not available in the dataset.")


with tab2:
    st.header("Anomaly Detection (Local Outlier Factor - LOF)")
    st.caption("LOF detects local density deviations, useful for identifying short-lived, isolated anomalies.")

    # RADIO BUTTON and SLIDERS for LOF Analysis
    selected_variable = st.radio("Select Variable for LOF Analysis", 
                                ["precipitation", "wind_speed_10m", "wind_gusts_10m"])

    col_c, col_d = st.columns(2)
    with col_c:
        contamination = st.slider("Contamination Level (Expected % of Anomalies)", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f", help="The expected proportion of outliers in the data.")
    with col_d:
        n_neighbors = st.slider("Number of Neighbors (n_neighbors)", min_value=5, max_value=50, value=20, step=1, help="Number of neighbors used to calculate local density.")

    if selected_variable in df_ready.columns:
        # --- CALL FUNCTION FROM functions.py ---
        fig, summary = functions.precipitation_lof_plot( 
            df_ready["time"].values,
            df_ready[selected_variable], 
            contamination=contamination,
            n_neighbors=n_neighbors,
            )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Anomaly Summary")
        st.write(f"**Total records analyzed:** {summary['n_total']}")
        st.write(f"**Anomalies detected:** {summary['n_outliers']} ({summary['percent_outliers']}%)")
    else:
        st.warning(f"Selected variable ({selected_variable}) not available in the dataset.")
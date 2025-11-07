import streamlit as st
import pandas as pd
import numpy as np
import functions # This now holds all analysis functions
import requests # Needed for exception handling


st.set_page_config(
    page_title="Anomaly Detection",
    layout="wide"
)

st.title("Anomaly and Outlier Detection")

# --- 1. CONTEXT AND DATA FETCHING ---

if 'weather_source_area' not in st.session_state:
    st.info("Please go back to the 'Elhub Production Data' page to select a Price Area first.")
    st.stop() 

selected_area = st.session_state['weather_source_area']
selected_groups = st.session_state.get('elhub_selected_groups', ['No groups selected']) 
groups_text = ', '.join([g.capitalize() for g in selected_groups])

st.info(
    f"""
    **Chosen parameters from Elhub Production Page:**
    
    * **Weather Location (Price Area):** {selected_area}
    """
)

# Load data using the cached API function (Layer 1 Caching)
try:
    df = functions.download_weather_data(selected_area) 
    
    if df is None or df.empty:
        st.error("Failed to retrieve weather data or dataset is empty. Check API/network.")
        st.stop()
        
except Exception as e:
    st.error(f"Error loading weather data: {e}")
    st.stop()


# Reset index to make 'time' a column for passing to functions
df_ready = df.reset_index()


# --- 2. TABBED INTERFACE ---

tab1, tab2 = st.tabs(["Temperature Outliers (Robust SPC)", "Precipitation/Wind Anomalies (LOF)"])

with tab1:
    st.header("Temperature Outlier Analysis (Robust SPC)")
    st.caption("Robust Statistical Process Control (SPC) based on Seasonal-Adjusted Time Variation (SATV) using DCT filtering.")

    # SLIDERS for SPC Analysis
    col_a, col_b = st.columns(2)
    with col_a:
        freq_cutoff = st.slider("DCT Low-Frequency Index Cutoff", min_value=1, max_value=250, value=100, step=1, help="Controls how much low-frequency trend is removed to isolate variation.")
    with col_b:
        num_std = st.slider("Number of Robust Standard Deviations (k)", min_value=2.0, max_value=5.0, value=3.0, step=0.1, help="Defines the control limits (Center ± k * Robust Std. Dev.).")

    if 'temperature_2m' in df_ready.columns:
        # --- CALL FUNCTION FROM functions.py ---
        fig, summary = functions.temperature_spc_from_satv(
            df_ready["time"].values,
            df_ready["temperature_2m"].values,
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
        variable_title = selected_variable.replace('_', ' ').title()
        
        # --- CALL FUNCTION FROM functions.py ---
        fig, summary = functions.precipitation_lof_plot( 
            df_ready["time"].values,
            df_ready[selected_variable], 
            contamination=contamination,
            n_neighbors=n_neighbors,
            variable_name=variable_title
            )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Anomaly Summary")
        st.write(f"**Total records analyzed:** {summary['n_total']}")
        st.write(f"**Anomalies detected:** {summary['n_outliers']} ({summary['percent_outliers']}%)")
    else:
        st.warning(f"Selected variable ({selected_variable}) not available in the dataset.")
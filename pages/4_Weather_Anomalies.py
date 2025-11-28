import streamlit as st
import pandas as pd
import numpy as np
from utilities.app_state import render_app_state_controls
from utilities import functions # Contains anomaly functions and weather download
import requests
from datetime import datetime, timezone 
import calendar # Needed for date range calculation

st.set_page_config(
    page_title="Anomaly Detection",
    layout="wide"
)

# --- GLOBAL UTILITY: DATE RANGE CALCULATION (Copied for page independence) ---
def get_global_date_range():
    """Calculates the filtering date range based on the global session state."""
    
    period_level = st.session_state.get('period_level')
    tz = timezone.utc
    
    if period_level == "Annual":
        year = st.session_state.get('selected_year', datetime.now().year)
        start_dt = datetime(year, 1, 1, tzinfo=tz)
        end_dt = datetime(year, 12, 31, 23, 59, 59, tzinfo=tz)
    
    elif period_level == "Monthly":
        year = st.session_state.get('selected_year', datetime.now().year)
        month = st.session_state.get('selected_month', 1)
        
        start_dt = datetime(year, month, 1, tzinfo=tz)
        last_day = calendar.monthrange(year, month)[1]
        end_dt = datetime(year, month, last_day, 23, 59, 59, tzinfo=tz)
        
    elif period_level == "Custom Date Range":
        start_date_obj = st.session_state.get('start_date', datetime(2021, 1, 1).date())
        end_date_obj = st.session_state.get('end_date', datetime.now().date())
        
        start_dt = datetime.combine(start_date_obj, datetime.min.time(), tzinfo=tz)
        end_dt = datetime.combine(end_date_obj, datetime.max.time(), tzinfo=tz)
        
    else: # Default (fallback)
        year = datetime.now().year
        start_dt = datetime(year, 1, 1, tzinfo=tz)
        end_dt = datetime(year, 12, 31, 23, 59, 59, tzinfo=tz)

    return start_dt, end_dt


# --- 1. RENDER GLOBAL CONTROLS IN SIDEBAR ---
with st.sidebar:
    render_app_state_controls()

# --- 2. ACCESS GLOBAL STATE AND DATE RANGE ---
selected_area = st.session_state.get('pricearea')
period_level = st.session_state.get('period_level')

# Calculate the effective date range
start_dt, end_dt = get_global_date_range()
start_str = start_dt.strftime('%Y-%m-%d')
end_str = end_dt.strftime('%Y-%m-%d')


if not selected_area:
    st.info("The global Price Area selector is not yet initialized. Please use the sidebar on the Explorer page.")
    st.stop() 


st.title("Anomaly and Outlier Detection")

# --- DISPLAY CONTEXT BOX ---
st.info(
    f"""
    **Current relevant filter settings:**
    
    Inherited from the analysis scope filters configured in the sidebar.
    
    * **Weather Location (Price Area):** {selected_area}
    * **Time Period:** {period_level} ({start_str} to {end_str})
    """
)


# Load data using the cached API function
try:
    with st.spinner(f"Loading weather data for {selected_area}..."):
        # df_full loads all available data (2021-2024)
        df_full = functions.download_weather_data(selected_area) 
    
    if df_full is None or df_full.empty:
        st.warning("Failed to retrieve weather data or dataset is empty. Check API/network.")
        st.stop()
        
except Exception as e:
    st.error(f"Error loading weather data: {e}")
    st.stop()


# --- FILTER DATA BY GLOBAL DATE RANGE ---
# Ensure the index is localized before filtering
if df_full.index.tz is None:
    df_full = df_full.tz_localize('UTC')

df = df_full[(df_full.index >= start_dt) & (df_full.index <= end_dt)].copy()

if df.empty:
    st.warning(f"No data available for the selected time range ({start_str} to {end_str}). Please adjust the sidebar filter.")
    st.stop()
    
# Reset index to make 'time' a column for passing to functions
df_ready = df.reset_index()


# --- 3. TABBED INTERFACE ---

tab1, tab2 = st.tabs(["Temperature Outliers (Robust SPC)", "Precipitation/Wind Anomalies (LOF)"])

with tab1:
    st.header("Temperature Outlier Analysis (Robust SPC)")
    st.caption("Robust Statistical Process Control (SPC) based on Seasonal-Adjusted Time Variation (SATV) using DCT filtering.")

    # Sliders for SPC Analysis
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
        with st.spinner("Running Robust SPC Analysis..."):
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

    # Radio button and sliders for LOF Analysis
    selected_variable = st.radio("Select Variable for LOF Analysis", 
                                ["precipitation", "wind_speed_10m", "wind_gusts_10m"])

    col_c, col_d = st.columns(2)
    with col_c:
        outlier_fraction = st.slider(
            "Outlier Fraction (Expected % of Anomalies)", 
            min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f", 
            key='lof_contamination_slider',
            help="The expected proportion of outliers in the data."
        )
    with col_d:
        n_neighbors = st.slider(
            "Number of Neighbors (n_neighbors)", 
            min_value=5, max_value=50, value=20, step=1, 
            key='lof_neighbors_slider', 
            help="Number of neighbors used to calculate local density."
        )

    if selected_variable in df_ready.columns:
        with st.spinner(f"Running LOF Analysis on {selected_variable}..."):
            fig, summary = functions.precipitation_lof_plot( 
                df_ready["time"].values,
                df_ready[selected_variable], 
                outlier_frac=outlier_fraction,
                n_neighbors=n_neighbors,
                variable_name=selected_variable
                )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Anomaly Summary")
        st.write(f"**Total records analyzed:** {summary['n_total']}")
        st.write(f"**Anomalies detected:** {summary['n_outliers']} ({summary['percent_outliers']}%)")
    else:
        st.warning(f"Selected variable ({selected_variable}) not available in the dataset.")
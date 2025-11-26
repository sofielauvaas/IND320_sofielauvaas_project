import streamlit as st
from utilities.app_state import render_app_state_controls
from utilities import functions
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Energy Time Series Analysis", layout="wide")

# --- 1. RENDER GLOBAL CONTROLS IN SIDEBAR ---
with st.sidebar:
    render_app_state_controls()

# --- 2. ACCESS GLOBAL STATE ---
pricearea = st.session_state.get("pricearea")
globally_selected_groups = st.session_state.get("productiongroup")

# --- INITIAL CHECK ---
if not pricearea or not globally_selected_groups:
    st.info("Please select a Price Area and at least one Production Group on the Production/Consumption Explorer pages before viewing this analysis.")
    st.stop()

st.title("Time Series Decomposition and Frequency Analysis (2021–2024)")

# --- DATA LOADING ---
try:
    with st.spinner("Loading production data..."):
        df_production = functions.load_data_from_mongo()
except Exception as e:
    st.error(f"Failed to load production data from MongoDB. Error: {e}")
    st.stop() 

if df_production.empty:
    st.warning("No production data found in the MongoDB collection.")
    st.stop()

# --- DATA SANITIZATION (From previous fix) ---
df_production.columns = df_production.columns.str.strip().str.lower()
if 'productionGroup' in df_production.columns:
    df_production.rename(columns={'productionGroup': 'productiongroup'}, inplace=True)
if 'productiongroup' not in df_production.columns:
    st.error("FATAL ERROR: The required column 'productiongroup' could not be found.")
    st.stop()
# ---------------------------------------------


# --- FILTERING LOGIC & CONTEXT DISPLAY ---
analysis_options = globally_selected_groups

groups_text = ', '.join([g.capitalize() for g in analysis_options])
st.info(
    f"""
    **Analysis Scope** (by the explorer page configuration):
    
    * **Price Area:** **{pricearea}**
    * **Available Groups:** {groups_text}
    """
)

# --- LOCAL SELECTOR ---
st.markdown("##### Select Production Group for Detailed Analysis:")
selected_group_for_analysis = st.selectbox(
    "Group:",
    analysis_options,
    index=0,
    label_visibility="collapsed"
)

# --- TABS AND ANALYSIS ---
tab1, tab2 = st.tabs(["STL Decomposition", "Spectrogram"])


# TAB 1: STL Decomposition
with tab1:
    st.markdown(f"#### Seasonal-Trend Decomposition (STL): {selected_group_for_analysis.capitalize()} in {pricearea}")
    
    col_period, col_seasonal, col_trend = st.columns(3)
    with col_period:
        period = st.number_input("Period (e.g., 168 for weekly)", min_value=1, value=168, step=1, key='stl_period')
    with col_seasonal:
        seasonal = st.number_input("Seasonal Smoothing (Odd)", min_value=3, value=9, step=2, key='stl_seasonal')
    with col_trend:
        trend = st.number_input("Trend Smoothing (Odd)", min_value=3, value=241, step=2, key='stl_trend')
    
    
    try:
        with st.spinner("Performing STL decomposition..."):
            fig_stl = functions.stl_decomposition_elhub(
                df_production,
                pricearea=pricearea, # Uses pricearea (no underscore)
                productiongroup=selected_group_for_analysis,
                period=period,
                seasonal=seasonal,
                trend=trend
            )
            st.plotly_chart(fig_stl, use_container_width=True)
    except Exception as e:
        st.error(f"Error during STL Decomposition. Please check parameters or data availability. Error: {e}")


# TAB 2: Spectrogram Analysis (Requires functions.py to be updated to return Plotly)
with tab2:
    st.markdown(f"#### Spectrogram Frequency Analysis: {selected_group_for_analysis.capitalize()} in {pricearea}")
    st.caption("Visualizes the power of different frequencies (periods) over time. Requires updating backend to use Plotly.")
    
    col_window, col_overlap = st.columns(2)
    with col_window:
        window_length = st.slider("Window Length (NPERSEG - hours)", min_value=64, max_value=512, value=256, step=64, key='spec_window')
        st.caption("Length of each segment for analysis.")
    with col_overlap:
        overlap = st.slider("Overlap (NOVERLAP - hours)", min_value=32, max_value=256, value=128, step=32, key='spec_overlap')
        st.caption("Number of overlapping samples between segments.")

    try:
        with st.spinner("Calculating Spectrogram..."):
            # ASSUMPTION: functions.create_spectrogram is updated to return a Plotly figure
            fig_spec = functions.create_spectrogram( 
                df_production, 
                pricearea=pricearea, # Uses pricearea (no underscore)
                productiongroup=selected_group_for_analysis,
                window_length=window_length,
                overlap=overlap
            )
            st.plotly_chart(fig_spec, use_container_width=True) 
    except Exception as e:
        st.error(f"Error during Spectrogram analysis. Error: {e}")
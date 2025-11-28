import streamlit as st
from utilities.app_state import render_app_state_controls
from utilities import functions 
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone 
import calendar # Required for date range calculation

st.set_page_config(page_title="Energy Time Series Analysis", layout="wide")

# Define the expected group column name in the loaded data
GROUP_COL = 'group' 

# --- Date Range Calculation (Copied here for completeness, or assumed to be in 1_Energy_Explorer.py) ---
# NOTE: If this function is already defined in 1_Energy_Explorer.py, you should ensure it is globally accessible 
# or copy it here. We assume it is copied here to avoid dependency issues.
def get_elhub_date_range():
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

# --- 2. ACCESS GLOBAL STATE ---
pricearea = st.session_state.get("pricearea")
globally_selected_groups = st.session_state.get("group") 
data_type = st.session_state.get("data_type", "production") 

# --- INITIAL CHECK ---
if not pricearea or not globally_selected_groups:
    st.info("Please select a Price Area and at least one Energy Source/Group on the Explorer pages before viewing this analysis.")
    st.stop()

st.title("Time Series Decomposition and Frequency Analysis")

# --- DATA LOADING ---
try:
    with st.spinner(f"Loading {data_type} data..."):
        # Load the raw data
        df_energy = functions.load_data_from_mongo(data_type=data_type)
except Exception as e:
    st.error(f"Failed to load {data_type} data from MongoDB. Error: {e}")
    st.stop() 

if df_energy.empty:
    st.warning(f"No {data_type} data found in the MongoDB collection.")
    st.stop()

# --- DATA SANITIZATION AND GROUP COLUMN CHECK ---
df_energy.columns = df_energy.columns.str.strip().str.lower()
if GROUP_COL not in df_energy.columns:
    st.error(
        f"""
        **FATAL DATA ERROR: Grouping column missing.**
        
        The analysis requires the `{GROUP_COL}` column for filtering, but it is missing.
        
        **Action Required:** This indicates your MongoDB collection for `{data_type}` contains aggregated data (only totals). 
        Please verify the Spark ingestion logic in your notebook to ensure 
        the raw grouping column (`productiongroup`/`consumptiongroup`) was not accidentally dropped 
        by an aggregation function before writing to MongoDB.
        """
    )
    st.stop()
# --------------------------------------------------------------------------


# --- FILTERING LOGIC & CONTEXT DISPLAY (Using st.info) ---
analysis_options = globally_selected_groups
data_label = "Production Group" if data_type.lower() == 'production' else "Consumption Group"

groups_text = ', '.join([g.capitalize() for g in analysis_options])

# Get date range from global state
start_dt, end_dt = get_elhub_date_range()
start_str = start_dt.strftime('%Y-%m-%d')
end_str = end_dt.strftime('%Y-%m-%d')

# Filter display
st.info(
    f"""
    **Current relevant filter settings:**
    
    * **Data Type:** {data_type.capitalize()}
    * **Price Area:** {pricearea}
    * **Energy Sources:** {groups_text}
    * **Time Period:** {st.session_state.get('period_level')} ({start_str} to {end_str})

    *Inherited from the analysis scope filters configured in the sidebar.
    """
)


# --- LOCAL SELECTOR ---
st.markdown(f"##### Select {data_label} for Detailed Analysis:")
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
    st.markdown(f"#### Seasonal-Trend Decomposition (STL): {selected_group_for_analysis.capitalize()} ({data_type.capitalize()}) in {pricearea}")
    st.caption("Decomposing the time series into trend, seasonal, and residual components.")
    
    col_period, col_seasonal, col_trend = st.columns(3)
    with col_period:
        # 168 hours = 7 days (weekly seasonality)
        period = st.number_input("Period (e.g., 168 for weekly)", min_value=1, value=168, step=1, key='stl_period')
    with col_seasonal:
        seasonal = st.number_input("Seasonal Smoothing (Odd)", min_value=3, value=9, step=2, key='stl_seasonal')
    with col_trend:
        trend = st.number_input("Trend Smoothing (Odd)", min_value=3, value=241, step=2, key='stl_trend')
    
    
    try:
        with st.spinner("Performing STL decomposition..."):
            # CRITICAL FIX: Pass start_dt and end_dt for filtering
            fig_stl = functions.stl_decomposition_elhub(
                df_energy,
                pricearea=pricearea,
                productiongroup=selected_group_for_analysis,
                period=period,
                seasonal=seasonal,
                trend=trend,
                start_dt=start_dt, # NEW PARAM
                end_dt=end_dt # NEW PARAM
            )
            st.plotly_chart(fig_stl, use_container_width=True)
    except Exception as e:
        st.error(f"Error during STL Decomposition. Please check parameters or data availability. Error: {e}")
        st.code(f"Traceback: {e}") 


# TAB 2: Spectrogram Analysis
with tab2:
    st.markdown(f"#### Spectrogram Frequency Analysis: {selected_group_for_analysis.capitalize()} ({data_type.capitalize()}) in {pricearea}")
    st.caption("Visualizes the intensity (power) of different frequencies (periods) over time. This helps identify when cycles like daily or weekly patterns are strongest.")
    
    col_window, col_overlap = st.columns(2)
    with col_window:
        window_length = st.slider("Window Length (NPERSEG - hours)", min_value=64, max_value=512, value=256, step=64, key='spec_window')
        st.caption("Length of each segment for analysis (e.g., 256 hours).")
    with col_overlap:
        overlap = st.slider("Overlap (NOVERLAP - hours)", min_value=32, max_value=256, value=128, step=32, key='spec_overlap')
        st.caption("Number of overlapping samples between segments.")

    try:
        with st.spinner("Calculating Spectrogram..."):
            # CRITICAL FIX: Pass start_dt and end_dt for filtering
            fig_spec = functions.create_spectrogram( 
                df_energy, 
                pricearea=pricearea,
                productiongroup=selected_group_for_analysis,
                window_length=window_length,
                overlap=overlap,
                start_dt=start_dt, # NEW PARAM
                end_dt=end_dt # NEW PARAM
            )
            st.plotly_chart(fig_spec, use_container_width=True) 
    except Exception as e:
        st.error(f"Error during Spectrogram analysis. Error: {e}")
        st.code(f"Traceback: {e}")
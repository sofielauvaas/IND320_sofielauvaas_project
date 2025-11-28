import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
import calendar
from utilities.app_state import render_app_state_controls
from utilities import functions 

st.set_page_config(layout="wide")

# Define the standardized group column name (must match functions.py)
GROUP_COL = 'group' 

# --- UTILITIES ---

def calculate_sliding_correlation(df, weather_col, energy_col, window_size, lag_days):
    """
    Calculates the rolling correlation between two daily series, 
    applying a time lag (shifting the weather series).
    """
    
    # 1. Apply Lag (shift the weather data backward by lag_days)
    if lag_days != 0:
        df[weather_col] = df[weather_col].shift(-lag_days) 
    
    # 2. Calculate the rolling correlation
    correlation_series = df[weather_col].rolling(window=window_size).corr(df[energy_col])
    
    # Shift the weather data back immediately (crucial for accurate plotting/recalculation)
    if lag_days != 0:
        df[weather_col] = df[weather_col].shift(lag_days) 
        
    return correlation_series.dropna()

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


@st.cache_data(ttl=3600, show_spinner="Aggregating and aligning daily data...")
def prepare_daily_data(df_energy_hourly, pricearea, selected_group, weather_col, start_dt, end_dt): # <--- ADDED selected_group
    """
    Filters the hourly data by the selected energy group, aggregates it to daily 
    frequency, and aligns it with daily weather data.
    """
    
    # 1. Energy Aggregation (Filter by Group, Area, and Sum to Daily)
    
    # Filter by Price Area and selected Group
    energy_series = df_energy_hourly[
        (df_energy_hourly['pricearea'] == pricearea) & 
        (df_energy_hourly[GROUP_COL] == selected_group) # --- CRITICAL FILTER ---
    ].copy()

    # Sum hourly quantity to daily total
    energy_daily = (
        energy_series.set_index("starttime")['quantitykwh']
        .resample("D").sum().rename('energy_quantity')
    )
    
    # 2. Weather Aggregation (Load and Aggregate to Daily)
    weather_hourly = functions.download_weather_data(pricearea)
    
    if weather_col == 'precipitation':
        weather_daily = weather_hourly[weather_col].resample("D").sum().rename(weather_col)
    else:
        weather_daily = weather_hourly[weather_col].resample("D").mean().rename(weather_col)
        
    # 3. Merge daily series and filter by global time range
    df_merged = pd.concat([energy_daily, weather_daily], axis=1).dropna()
    
    # Ensure index is localized for filtering consistency
    if df_merged.index.tz is None:
        df_merged = df_merged.tz_localize('UTC')
    
    df_analysis = df_merged[(df_merged.index >= start_dt) & (df_merged.index <= end_dt)].copy()
    
    # Remove Timezone info for final correlation compatibility (required when mixing libraries)
    if df_analysis.index.tz is not None:
        df_analysis.index = df_analysis.index.tz_localize(None)

    return df_analysis


# --- PAGE SETUP AND STATE ACCESS ---

# 1. Load app state controls (Area, Type, Time)
with st.sidebar:
    render_app_state_controls()

# 2. Access global filters
data_type = st.session_state.get('data_type', "Production")
pricearea = st.session_state.get('pricearea', "NO1")
globally_selected_groups = st.session_state.get('group') 
period_level = st.session_state.get('period_level')

# 3. Define time range and coordinates
start_dt, end_dt = get_global_date_range()
start_str = start_dt.strftime('%Y-%m-%d')
end_str = end_dt.strftime('%Y-%m-%d')

coords = st.session_state.get('selected_coords')
lat, lon = coords if coords else (60.0, 10.0) 

# --- INITIAL CHECKS ---
if not globally_selected_groups:
    st.info("Please use the sidebar to select a Price Area and at least one Energy Source/Group.")
    st.stop() 

st.title("Sliding Window Correlation Analysis")
st.subheader("Meteorology vs. Energy Flow")

st.info(
    f"""
    **Analysis Scope:**
    * **Data Type:** {data_type}
    * **Location:** {pricearea}
    * **Time Period:** {period_level} ({start_str} to {end_str})

    *Inherited from the analysis scope filters configured in the sidebar.
    """
)

# --- Data Fetching and Preparation ---

# Get the full, merged dataset (Weather + Aggregated Energy)
with st.spinner(f"Loading hourly {data_type} and weather data for {pricearea}..."):
    # Load the base hourly energy data
    df_energy_hourly = functions.load_elhub_data(data_type) 
    
if df_energy_hourly.empty:
    st.warning("Energy data loading failed or returned empty.")
    st.stop()


# --- UI PARAMETERS ---
st.header("Analysis Parameters")

col1, col2, col3, col4 = st.columns(4) # Added a column for the Group Selector

# Filter options
weather_vars = ['temperature_2m', 'wind_speed_10m', 'precipitation', 'wind_direction_10m', 'wind_gusts_10m']
energy_col = 'energy_quantity' 

with col1:
    # 1. GROUP SELECTION (NEW)
    selected_group = st.selectbox(
        f"Energy Group ({data_type})",
        globally_selected_groups,
        key='energy_group_select'
    )

with col2:
    # 2. WEATHER VARIABLE
    weather_col = st.selectbox("Meteorological Driver:", weather_vars, key='weather_col_select')
with col3:
    # 3. WINDOW SIZE
    window_days = st.slider("Window Size (Days):", 7, 365, 30, key='swc_window_days')
with col4:
    # 4. LAG
    lag_days = st.slider("Time Lag (Energy follows Weather by N days):", -30, 30, 0, key='swc_lag_days')
    st.caption("Positive lag means weather leads energy.")


# --- Daily Aggregation and Time Filtering (CRITICAL STEP) ---

# Aggregate data to Daily resolution and align
with st.spinner(f"Aggregating {selected_group} data to daily resolution..."):
    # FIX: Pass all required arguments including the selected group
    df_analysis = prepare_daily_data(
        df_energy_hourly, 
        pricearea, 
        selected_group, # <--- NEW PARAMETER
        weather_col, 
        start_dt, 
        end_dt
    )

if df_analysis.empty or len(df_analysis) < window_days:
    st.warning("No overlapping daily data found for the selected area and time range.")
    st.stop()

# --- Window Sliders ---
date_min = df_analysis.index.min().date()
date_max = df_analysis.index.max().date()

# Calculate the minimum start date based on window size
latest_start = (date_max - timedelta(days=window_days - 1))
if latest_start < date_min:
    latest_start = date_min

# Start date slider controls the center of the SWC window
start_date_window = st.slider(
    "Move window across time (start date)",
    min_value=date_min,
    max_value=latest_start,
    value=latest_start,
    format="YYYY-MM-DD",
    key='swc_start_date_slider'
)

# Convert slider output to naive datetime objects for filtering the daily data
win_start = datetime.combine(start_date_window, datetime.min.time())
win_end = win_start + timedelta(days=window_days - 1)

# --- Calculation (Auto-Run) ---

with st.spinner("Calculating rolling correlation..."):
    corr_series = calculate_sliding_correlation(
        df_analysis.copy(), # Pass a copy to avoid permanent lag shift
        weather_col, 
        energy_col, 
        window_size=window_days, 
        lag_days=lag_days
    )
    
    # Store result in session state for display
    st.session_state['correlation_result'] = corr_series
    st.session_state['correlation_params'] = {
        'weather_var': weather_col,
        'energy_var': energy_col,
        'window_days': window_days,
        'lag_days': lag_days
    }
        
# --- Visualization ---

if not st.session_state['correlation_result'].empty:
    corr_series = st.session_state['correlation_result']
    params = st.session_state['correlation_params']
    
    st.header("Sliding Window Correlation (SWC)")
    
    # 1. Plot Rolling Correlation (SWC)
    plot_df_corr = pd.DataFrame(corr_series, columns=['Correlation'])
    fig_swc = go.Figure()
    
    fig_swc.add_trace(go.Scatter(x=plot_df_corr.index, y=plot_df_corr['Correlation'].values, mode='lines', name='Rolling Correlation', line=dict(color='#0055A4', width=2)))

    # Zero line for context
    fig_swc.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Correlation")
    
    # Highlight the current SWC window segment
    fig_swc.add_vrect(
        x0=win_start, x1=win_end, 
        fillcolor="red", opacity=0.10, line_width=0, layer="below"
    )

    fig_swc.update_layout(
        title=f"Rolling Correlation ({params['window_days']} Day Window, {params['lag_days']} Day Lag)",
        xaxis_title="Time",
        yaxis_title=f"Correlation Coefficient (r)",
        yaxis=dict(range=[-1.0, 1.0]),
        height=400,
        title_x=0.01
    )

    st.plotly_chart(fig_swc, use_container_width=True)
    
    # 2. Plot Energy and Weather Series (for Window Highlight Visualization)
    st.header("Daily Time Series Visualization")
    
    # Create the necessary data structures based on the calculated parameters
    df_window = df_analysis.copy()
    
    # Apply Lag to weather for visualization purposes (Weather leads)
    if params['lag_days'] != 0:
        df_window[params['weather_var']] = df_window[params['weather_var']].shift(params['lag_days'])
    
    col_chart_1, col_chart_2 = st.columns(2)
    
    # A. Energy Series Plot
    with col_chart_1:
        fig_energy = go.Figure()
        
        # Highlighted Window Segment (Blue)
        fig_energy.add_trace(go.Scatter(x=df_analysis.index, y=df_analysis['energy_quantity'], mode="lines", name="Energy Quantity", line=dict(color='steelblue', width=3)))
        
        # Add window highlight
        fig_energy.add_vrect(x0=win_start, x1=win_end, fillcolor="red", opacity=0.15, line_width=0, layer="below")
        
        fig_energy.update_layout(height=300, xaxis_title="Date", yaxis_title="Energy (kWh/day)", title=f"Daily Energy Quantity for {selected_group.capitalize()}")
        st.plotly_chart(fig_energy, use_container_width=True)

    # B. Weather Series Plot
    with col_chart_2:
        fig_weather = go.Figure()
        
        # Highlighted Window Segment (Red)
        fig_weather.add_trace(go.Scatter(x=df_window.index, y=df_window[params['weather_var']], mode="lines", name="Weather Data", line=dict(color='red', width=3)))

        # Add window highlight
        fig_weather.add_vrect(x0=win_start, x1=win_end, fillcolor="red", opacity=0.15, line_width=0, layer="below")

        fig_weather.update_layout(height=300, xaxis_title="Date", yaxis_title=params['weather_var'].replace('_', ' ').title(), title=f"Daily Weather Series: {params['weather_var'].replace('_', ' ').title()}")
        st.plotly_chart(fig_weather, use_container_width=True)


    # 3. Summary Calculation
    overall_corr = calculate_sliding_correlation(df_analysis.copy(), params['weather_var'], params['energy_var'], window_size=len(df_analysis), lag_days=params['lag_days']).iloc[-1]
    
    st.subheader("Summary")
    col_int_1, col_int_2 = st.columns(2)

    with col_int_1:
        st.metric("Overall Lagged Correlation", f"{overall_corr:.3f}", help="Correlation over the entire period with applied lag.")
    with col_int_2:
        st.metric("Mean SWC", f"{corr_series.mean():.3f}", help="Average correlation coefficient across all sliding windows.")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
import calendar
import math
from utilities import functions 
from utilities.app_state import render_app_state_controls

# --- REQUIRED CONSTANTS (Accessed from functions.py) ---
FENCE_FACTORS = {
    "Wyoming": 8.5, "Slat-and-wire": 7.7, "Solid": 2.9
}
TABLER_T = functions.TABLER_T      
TABLER_F = functions.TABLER_F     
TABLER_THETA = functions.TABLER_THETA   

@st.cache_data(ttl=3600, show_spinner="Fetching localized weather data...")
def get_weather_data_for_drift(area_id):
    """
    Fetches the full 2021-2024 weather dataset from the Open-Meteo API for the given price area.
    """
    return functions.download_weather_data(area_id)

def calculate_monthly_snow_drift(df_season, T, F, theta, current_year):
    """Helper function for the monthly calculation (Bonus Task)."""
    
    if 'temperature_2m' not in df_season.columns or 'precipitation' not in df_season.columns:
        return pd.DataFrame(columns=['period', 'season', 'Qt (kg/m)']) 

    df_season['Swe_hourly'] = np.where(df_season['temperature_2m'] < 1, df_season['precipitation'], 0)
    
    df_monthly_agg = df_season.resample('M').agg({
        'Swe_hourly': 'sum',
        'wind_speed_10m': lambda x: x.tolist()
    }).dropna(subset=['wind_speed_10m'])
    
    monthly = []
    
    for month_start, row in df_monthly_agg.iterrows():
        monthly_wind_speeds = row['wind_speed_10m']
        monthly_Swe = row['Swe_hourly']
        
        monthly_result = functions.compute_snow_transport(T, F, theta, monthly_Swe, monthly_wind_speeds)
        
        monthly.append({
            "period": month_start.strftime("%Y-%m"),
            "season": f"{current_year}-{current_year+1}",
            "Qt (kg/m)": monthly_result["Qt (kg/m)"]
        })
        
    return pd.DataFrame(monthly)


# --- Main Page Execution ---

st.set_page_config(layout='wide')
st.title("Snow Drift Analysis")
st.markdown("Calculates potential snow transport ($$Q_t$$) based on wind speed, temperature, and precipitation for the selected map location (Seasonal Year: July 1st—June 30th).")


# 1. Check for Map Selection
selected_coords = st.session_state.get("last_clicked")
selected_area_id = st.session_state.get("selected_area")

if not selected_coords or selected_area_id is None:
    st.warning("No map location selected yet. Please go to the **Map** page and click a location.")
    st.stop()

# 1.1 Sidebar Control Setup 
with st.sidebar:
    render_app_state_controls() 
    
    if not selected_coords or selected_area_id is None:
        st.warning("Please select a location on the Map page first.")
        st.stop()
        
    lat, lon = selected_coords

# 2. Snow Drift Parameters UI (Simplified and cleaned)

st.header("Drift Calculation Parameters")
st.markdown(f"**Analysis Location:** `{selected_area_id}` (Lat {lat:.4f}, Lon {lon:.4f})")

# --- INPUTS (Year Slider Only) ---
with st.container(border=True):
    
    col_input_1, col_input_2 = st.columns([1, 1])

    with col_input_1:
        # 2.1 Year Range Slider (Seasonal Year: July 1st - June 30th)
        current_year = 2024 # Data runs through 2024
        start_year, end_year = st.slider(
            "Select Seasonal Year Range (July 1st — June 30th)",
            min_value=2021, # Restricted to available data
            max_value=current_year,
            value=(2021, 2023), 
            step=1,
            key='drift_year_range'
        )
        
        # Internal definition of fence type for the required metric
        fence_type = "Wyoming" 

        
    # Pass fixed constants for calculation (read directly from the module)
    T_param = TABLER_T
    F_param = TABLER_F
    theta_param = TABLER_THETA


# 3. Data Fetching and Calculations (Automatic execution)

with st.spinner("Fetching weather data and running seasonal drift calculations..."):
    
    # 3.1 Fetch Weather Data for the entire range
    df_weather_full = get_weather_data_for_drift(selected_area_id)

    # 3.2 Prepare the date range for iteration
    start_date_full = datetime(start_year, 7, 1, tzinfo=timezone.utc)
    end_date_full = datetime(end_year + 1, 6, 30, 23, 59, 59, tzinfo=timezone.utc)
    
    # Filter full weather data by the seasonal range selected
    df_weather_filtered = df_weather_full[(df_weather_full.index >= start_date_full) & (df_weather_full.index <= end_date_full)].copy()
    
    if df_weather_filtered.empty:
        st.error(f"No weather data available in the requested seasonal range ({start_year}-{end_year+1}). Please adjust the year slider.")
        st.stop()
    
    # 3.3 Calculate annual results
    
    annual_results = []
    
    # Iterate through the selected seasonal years
    for y in range(start_year, end_year + 1):
        season_start = datetime(y, 7, 1, tzinfo=timezone.utc)
        season_end = datetime(y + 1, 6, 30, 23, 59, 59, tzinfo=timezone.utc)

        # CRITICAL FIX: The inner loop must only filter the ALREADY FILTERED DataFrame (df_weather_filtered).
        df_season = df_weather_filtered[(df_weather_filtered.index >= season_start) & (df_weather_filtered.index <= season_end)].copy()
        
        if df_season.empty:
            continue
            
        df_season['Swe_hourly'] = np.where(df_season['temperature_2m'] < 1, df_season['precipitation'], 0)
        total_Swe = df_season['Swe_hourly'].sum()
        wind_speeds = df_season["wind_speed_10m"].tolist()

        # Annual Calculation
        annual_result = functions.compute_snow_transport(T_param, F_param, theta_param, total_Swe, wind_speeds)
        
        annual_results.append({
            "season": f"{y}-{y+1}",
            "Qt (kg/m)": annual_result["Qt (kg/m)"],
            "Control": annual_result["Control"]
        })

    yearly_results_df = pd.DataFrame(annual_results)
    
    # --- Monthly Calculation (Retention for completeness) ---
    monthly_results_dfs = [] 
    for y in range(start_year, end_year + 1):
        df_season = df_weather_full[(df_weather_full.index >= datetime(y, 7, 1, tzinfo=timezone.utc)) & (df_weather_full.index <= datetime(y + 1, 6, 30, 23, 59, 59, tzinfo=timezone.utc))].copy()
        if not df_season.empty:
            df_monthly = calculate_monthly_snow_drift(df_season, T_param, F_param, theta_param, y) 
            monthly_results_dfs.append(df_monthly)

    if monthly_results_dfs:
        monthly_results_df = pd.concat(monthly_results_dfs, ignore_index=True)
    else:
        monthly_results_df = pd.DataFrame(columns=['period', 'season', 'Qt (kg/m)']) 
            
# 4. Display Results

if not yearly_results_df.empty:
    
    # Convert units and calculate metrics
    yearly_results_df["Qt (tonnes/m)"] = yearly_results_df["Qt (kg/m)"] / 1000
    overall_avg_qt = yearly_results_df['Qt (kg/m)'].mean()
    overall_avg_tonnes = overall_avg_qt / 1000
    
    # Calculate fence height for the standard fence type (Wyoming)
    avg_fence_height = functions.compute_fence_height(overall_avg_qt, fence_type)

    st.header("Analysis Results")
    st.markdown(f"**Mean Annual Snow Transport ($$Q_t$$):** The average calculated snow transport quantity in tonnes per meter of structure length.")
    
    # --- CONSOLIDATED METRICS & PLOT LAYOUT ---
    
    # Row 1: Annual Bar Plot (Left) and Wind Rose (Right)
    col_annual_plot, col_wind_rose = st.columns([1, 1])

    with col_annual_plot:
        fig_yearly = px.bar(
            yearly_results_df, 
            x='season', 
            y='Qt (tonnes/m)', 
            color='Control',
            title="Annual Snow Drift per Season (July–June)",
        )
        st.plotly_chart(fig_yearly, use_container_width=True)


    with col_wind_rose:
        fig_rose = functions.plot_wind_rose_plotly(df_weather_filtered, overall_avg_qt, start_year, end_year)
        st.plotly_chart(fig_rose, use_container_width=True)

    st.markdown("---")
    
    # Row 2: Summary Metrics (Below Plots)
    st.subheader("Summary Metrics")
    
    col_metric_1, col_metric_2 = st.columns(2)

    with col_metric_1:
        st.metric("Overall Avg. Drift ($$Q_t$$)", f"{overall_avg_tonnes:.1f} tonnes/m")
    with col_metric_2:
        st.metric("Total Seasons Analyzed", len(yearly_results_df))
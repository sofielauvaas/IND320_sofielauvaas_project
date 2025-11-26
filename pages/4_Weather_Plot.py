import streamlit as st
import pandas as pd
import numpy as np 
import requests 
import plotly.graph_objects as go
from utilities.app_state import render_app_state_controls
from utilities import functions

st.set_page_config(
    page_title="Weather Data Analysis",
    layout="wide"
)

# --- GLOBAL CONFIGURATION ---
WIND_DIRECTION_COL = "wind_direction_10m"
WIND_ARROW_COLOR = '#128264' 
COLOR_MAP = {
    'hydro': '#035397', 'wind': '#128264', 'solar': '#f9c80e',
    'thermal': '#546e7a', 'other': '#9dc183'
}
LINE_COLORS = list(COLOR_MAP.values()) 

# --- CACHED DATA PREPARATION & LOADING ---
@st.cache_data(show_spinner=False)
def prepare_data_subset(df, selected_year_months):
    # selected_year_months is a tuple of (start_YYYY-MM, end_YYYY-MM)
    start_str, end_str = selected_year_months
    
    # Create a Year-Month index for filtering
    df['year_month'] = df.index.strftime('%Y-%m')
    
    subset = df[(df['year_month'] >= start_str) & (df['year_month'] <= end_str)].copy()
    
    # Drop the temporary column before returning
    subset = subset.drop(columns=['year_month'], errors='ignore')
    return subset

@st.cache_data(show_spinner="Downloading weather data...")
def load_weather_data_local(area):
    return functions.download_weather_data(area)



# --- 1. RENDER GLOBAL CONTROLS IN SIDEBAR ---
with st.sidebar:
    render_app_state_controls()

# --- 2. ACCESS GLOBAL STATE ---
selected_area = st.session_state.get('pricearea')

if not selected_area:
    st.info("The global Price Area selector is not yet initialized. Please use the sidebar on the Production Explorer page.")
    st.stop() 

st.title("Weather Data Visualisation (2021–2024)")

# --- DISPLAY CONTEXT BOX ---
st.info(
    f"""
    **Analysis Scope** (by the explorer page configuration):
    
    * **Weather Location (Price Area):** **{selected_area}**
    """
)


try:
    df_raw = load_weather_data_local(selected_area) 
    if df_raw.empty:
        st.warning(f"No weather data available for {selected_area} in the 2021-2024 range.")
        st.stop()
except Exception as e:
    st.error(f"Error during data fetching: {e}")
    st.stop() 


# --- WIDGETS AND DYNAMIC DATA PROCESSING ---
columns = list(df_raw.columns.drop(WIND_DIRECTION_COL, errors='ignore'))
selected_col = st.selectbox("Select a column", ["All columns"] + columns)

# Prepare YYYY-MM options for the slider
df_raw['year_month'] = df_raw.index.strftime('%Y-%m')
available_year_months = sorted(df_raw['year_month'].unique())

if not available_year_months:
    st.warning("No time series data available for range selection.")
    st.stop()

# --- FIX START: Set default value back to the first month only ---
default_start_month = available_year_months[0]
# --- FIX END ---


# Use the YYYY-MM strings for the slider
selected_year_months = st.select_slider(
    "Select Month Range (Year-Month)",
    options=available_year_months,
    # The default value is now (first month, first month)
    value=(default_start_month, default_start_month), 
    format_func=lambda x: x # Display YYYY-MM directly
)

normalize_flag = st.checkbox("Normalize numeric columns (0-1 scale)")

# 1. Get the filtered subset (cached)
subset_indexed = prepare_data_subset(df_raw.drop(columns=['year_month']), selected_year_months)

# 2. Apply normalization and create the plotting DataFrame
df_plot = subset_indexed.reset_index().rename(columns={'index': 'time'})

if normalize_flag:
    numeric_cols = df_plot.select_dtypes(include="number").columns
    
    # Exclude the wind direction column
    cols_to_standardize = numeric_cols.drop(WIND_DIRECTION_COL, errors='ignore')

    for col in cols_to_standardize:
        min_val = df_plot[col].min()
        max_val = df_plot[col].max()
        if max_val != min_val:
            df_plot[col] = (df_plot[col] - min_val) / (max_val - min_val)
        else:
            df_plot[col] = 0.5 

# Get month names for title
first_month_name = selected_year_months[0]
last_month_name = selected_year_months[1]

# --- PLOTTING LOGIC ---

fig = go.Figure()

# 1. Plot Line Traces
if selected_col == "All columns":
    plot_cols = [c for c in df_plot.columns if c not in ["time", WIND_DIRECTION_COL]]
    for i, col in enumerate(plot_cols):
        fig.add_trace(go.Scatter(
            x=df_plot["time"], y=df_plot[col], mode="lines",
            line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=2), name=col
        ))
else:
    plot_cols = [selected_col]
    fig.add_trace(go.Scatter(
        x=df_plot["time"], y=df_plot[selected_col], mode="lines",
        line=dict(color=COLOR_MAP['hydro'], width=2), name=selected_col
    ))


# 2. Arrow Parameters Calculation
numeric_cols_unnorm = subset_indexed.select_dtypes(include=[np.number]).columns.drop(WIND_DIRECTION_COL, errors="ignore")
global_y_min = subset_indexed[numeric_cols_unnorm].min().min() if not numeric_cols_unnorm.empty else 0
global_y_max = subset_indexed[numeric_cols_unnorm].max().max() if not numeric_cols_unnorm.empty else 10

if normalize_flag:
    global_y_min = 0.0
    global_y_max = 1.0

arrow_every = max(1, len(df_plot) // 90)
arrow_y = global_y_min - (global_y_max - global_y_min) * 0.1
arrow_len = (global_y_max - global_y_min) * 0.1

time_span = df_plot['time'].iloc[-1] - df_plot['time'].iloc[0]
FIXED_TIME_OFFSET_MAGNITUDE = time_span * 0.01 


# 3. Add Arrows showing Wind Direction 
if WIND_DIRECTION_COL in df_plot.columns and (selected_col == "All columns" or selected_col == WIND_DIRECTION_COL):
    
    for i in range(0, len(df_plot), arrow_every):
        t = df_plot["time"].iloc[i]
        wind_dir = df_plot[WIND_DIRECTION_COL].iloc[i]

        # Convert wind direction (degrees) to radians, adding 180 because arrows point from source
        theta = np.deg2rad(wind_dir + 180) 

        # Vertical movement (Y-axis): Uses the cosine component of the vector
        y_change = np.cos(theta) * arrow_len
        arrow_y2 = arrow_y + y_change * 0.8

        # Horizontal movement (X-axis/Time): Uses the sine component of the vector
        x_change_prop = np.sin(theta) 
        arrow_dx = FIXED_TIME_OFFSET_MAGNITUDE * x_change_prop
        arrow_x2 = t + arrow_dx
        
        # Add annotation
        fig.add_annotation(
            x=t, y=arrow_y,
            ax=arrow_x2, ay=arrow_y2, 
            xref="x", yref="y", axref="x", ayref="y",
            text="",
            showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=1.4,
            arrowcolor=WIND_ARROW_COLOR,
        )

    # Legend entry for wind direction
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines+markers",
        line=dict(color=WIND_ARROW_COLOR, width=2),
        marker=dict(symbol="triangle-right", color=WIND_ARROW_COLOR, size=10),
        name="Wind Direction", showlegend=True
    ))

# 4. Final Layout and Axes Updates
names = {
    "temperature_2m": "Temperature (°C)", "precipitation": "Precipitation (mm)",
    "wind_speed_10m": "Wind Speed (m/s)", "wind_direction_10m": "Wind Direction (°)",
    "wind_gusts_10m": "Wind Gusts (m/s)",
}
for trace in fig.data:
    if trace.name in names:
        trace.name = names[trace.name]

if first_month_name == last_month_name:
    header_text = f"{names.get(selected_col, selected_col)} for {first_month_name}" if selected_col != "All columns" else f"All columns for {first_month_name}"
else:
    header_text = f"{names.get(selected_col, selected_col)} for {first_month_name} – {last_month_name}" if selected_col != "All columns" else f"All columns for {first_month_name} – {last_month_name}"

fig.update_layout(
    title=dict(text=header_text, x=0.5, xanchor="center", font=dict(size=24)),
    xaxis_title="Time",
    template="plotly_white",
    height=470,
    width=800,
    legend=dict(orientation="h", y=-0.1),
)

# Determine Y-axis max based on the data used for the lines
y_max_data = df_plot[plot_cols].max().max() if plot_cols else 1.0

# Extend y-axis to make space for arrows
fig.update_yaxes(range=[arrow_y - (global_y_max - global_y_min) * 0.08, y_max_data], nticks=11)

st.plotly_chart(fig, use_container_width=True)
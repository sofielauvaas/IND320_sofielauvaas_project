import streamlit as st
import pandas as pd
import numpy as np 
import requests 
import plotly.graph_objects as go
from utilities.app_state import render_app_state_controls
from utilities import functions
from datetime import datetime, timezone 
import calendar 
import plotly.express as px # Import plotly.express for visualization logic

st.set_page_config(
    page_title="Weather Data Analysis",
    layout="wide"
)

# --- GLOBAL CONFIGURATION (MISSING VARIABLES RE-ADDED) ---
WIND_DIRECTION_COL = "wind_direction_10m"
# Color map is essential for aesthetic consistency
COLOR_MAP = {
    'hydro': '#035397', 'wind': '#128264', 'solar': '#f9c80e',
    'thermal': '#546e7a', 'other': '#9dc183'
}
# LINE_COLORS must be defined for the multi-variable plot
LINE_COLORS = list(COLOR_MAP.values()) 
# --------------------------------------------------------

# --- GLOBAL UTILITY: DATE RANGE CALCULATION (Needed for local filtering) ---
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


# --- CACHED DATA PREPARATION & LOADING (FIXED FOR TIMEZONE COMPARISON) ---
@st.cache_data(show_spinner=False)
def prepare_data_subset(df, start_dt, end_dt):
    """Filters the weather DataFrame based on global datetime objects, correcting timezone on entry."""
    
    if df.empty:
        return pd.DataFrame()

    # CRITICAL FIX: Ensure the index is explicitly labeled as UTC-aware for comparison.
    if df.index.tz is None:
        df = df.tz_localize('UTC')
    elif str(df.index.tz) != 'UTC':
        # If it has a timezone, ensure it's converted to UTC for consistency
        df = df.tz_convert('UTC')
        
    # Filter the subset using the datetime range
    subset = df[(df.index >= start_dt) & (df.index <= end_dt)].copy()
    
    return subset

@st.cache_data(show_spinner="Downloading weather data...")
def load_weather_data_local(area):
    # Calls the central function defined in utilities/functions.py
    return functions.download_weather_data(area)


# --- 1. RENDER GLOBAL CONTROLS IN SIDEBAR ---
with st.sidebar:
    render_app_state_controls()

# --- 2. ACCESS GLOBAL STATE AND DERIVE RANGE ---
selected_area = st.session_state.get('pricearea')
selected_groups = st.session_state.get('group', []) 
period_level = st.session_state.get('period_level')

# Calculate the effective date range
start_dt, end_dt = get_global_date_range()
start_str = start_dt.strftime('%Y-%m-%d')
end_str = end_dt.strftime('%Y-%m-%d')


# --- INITIAL CHECKS ---
if not selected_area:
    st.info("The global Price Area selector is not yet initialized. Please use the sidebar on the Explorer page.")
    st.stop() 

st.title("Weather Data Summary") 

# --- DISPLAY CONTEXT BOX (New Structure) ---
groups_text = ', '.join([g.capitalize() for g in selected_groups])

st.info(
    f"""
    **Current relevant filter settings:**
    
    * **Weather Location :** {selected_area}
    * **Time Period:** {period_level} ({start_str} to {end_str})

    *Inherited from the analysis scope filters configured in the sidebar.
    """
)


# --- DATA FETCHING ---
try:
    with st.spinner(f"Loading weather data for {selected_area}..."):
        # df_full contains all years of data (2021-2024)
        df_full = load_weather_data_local(selected_area) 
    
    if df_full is None or df_full.empty:
        st.warning(f"No weather data available for {selected_area}.")
        st.stop()
except Exception as e:
    st.error(f"Error during data fetching: {e}")
    st.stop() 


# --- WIDGETS AND DYNAMIC DATA PROCESSING ---
WIND_DIRECTION_COL = "wind_direction_10m"
columns = list(df_full.columns.drop(WIND_DIRECTION_COL, errors='ignore'))
selected_col = st.selectbox("Select a column", ["All columns"] + columns)


# --- FILTER DATA BY GLOBAL DATE RANGE (CRITICAL STEP) ---
df = prepare_data_subset(df_full, start_dt, end_dt)

if df.empty:
    st.warning(f"No data available for {selected_area} in the selected time range ({start_str} to {end_str}). Please adjust the sidebar filter.")
    st.stop()


# 2. Apply normalization and create the plotting DataFrame
df_plot = df.reset_index().rename(columns={'index': 'time'})

# Calculate normalization flag
normalize_flag = st.checkbox("Normalize numeric columns (0-1 scale)")


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
# Use the filtered dates for the plot title, adjusting if only one month is present
if len(df.index) > 0:
    first_date_str = df.index.min().strftime('%Y-%m')
    last_date_str = df.index.max().strftime('%Y-%m')
    title_range_str = f"{first_date_str} – {last_date_str}" if first_date_str != last_date_str else first_date_str
else:
    title_range_str = f"{start_str} – {end_str}"


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
        line=dict(color=COLOR_MAP.get(selected_col, LINE_COLORS[0]), width=2), name=selected_col
    ))


# 2. Arrow Parameters Calculation
numeric_cols_unnorm = df.select_dtypes(include=[np.number]).columns.drop(WIND_DIRECTION_COL, errors="ignore")
global_y_min = df[numeric_cols_unnorm].min().min() if not numeric_cols_unnorm.empty else 0
global_y_max = df[numeric_cols_unnorm].max().max() if not numeric_cols_unnorm.empty else 10

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
            arrowcolor="#128264", # Hardcoded the wind color
        )

    # Legend entry for wind direction
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines+markers",
        line=dict(color="#128264", width=2),
        marker=dict(symbol="triangle-right", color="#128264", size=10),
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

header_text = f"{names.get(selected_col, 'All Weather Variables')} in {selected_area} ({title_range_str})"

fig.update_layout(
    title=dict(text=header_text, x=0.01, xanchor="left", font=dict(size=24)),
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


df = prepare_data_subset(df_full, start_dt, end_dt)

if df.empty:
    st.warning(f"No data available for {selected_area} in the selected time range ({start_str} to {end_str}). Please adjust the sidebar filter.")
    st.stop()


# Prepare a table: one row per numeric column
st.subheader(f"Hourly Statistics for {selected_area}")
table_data = []

# Generate a trend column name dynamically based on the filtered range
trend_col_name = f"Trend ({start_str} to {end_str})"

for col in df.columns: 
    # Use the entire filtered df for the stats
    if pd.api.types.is_numeric_dtype(df[col]):
        
        chart_data = df[col].tolist()
        
        row = {
            "Column": col.replace('_', ' ').title(), # Clean column name for display
            trend_col_name: chart_data,
            "Min": df[col].min(),
            "Max": df[col].max(),
            "Mean": df[col].mean().round(2),
            "Median": df[col].median().round(2),
            "Std": df[col].std().round(2)
        }
        table_data.append(row)

df_table = pd.DataFrame(table_data)

# Configure the LineChartColumn
column_config = {
    trend_col_name: st.column_config.LineChartColumn(
        label=f"Hourly Trend",
        width="medium"
    )
}

# Display the table without index
st.dataframe(
    df_table,
    column_config=column_config,
    use_container_width=True,
    hide_index=True
)
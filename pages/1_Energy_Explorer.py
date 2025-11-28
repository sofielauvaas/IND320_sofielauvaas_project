import streamlit as st
from utilities import functions 
import pandas as pd
import plotly.express as px
import calendar
from datetime import datetime, timezone 
from utilities.app_state import render_app_state_controls
from utilities import data_loader # New: Contains the consolidated load_and_clean_energy_data


# Define the standardized group column name (must match data_loader.py)
GROUP_COLUMN = 'group' 
COLOR_MAP = {
    # --- Production Groups ---
    'hydro': '#0055A4',   
    'wind': '#128264',    
    'solar': '#f9c80e', 
    'thermal': '#546e7a',   
    'other': '#9dc183',    
    
    # --- Consumption Groups ---
    'household': '#0055A4',  
    'cabin': '#128264',   
    'primary': '#f9c80e',   
    'secondary': '#546e7a', 
    'tertiary': '#9dc183',  
}

def get_elhub_date_range():
    """
    Calculates the filtering date range based on the global session state.
    
    IMPORTANT: Ensures all returned datetime objects are timezone-aware (UTC)
    to match the dataframe's 'starttime' column.
    """
    
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

# --- 2. ACCESS GLOBAL STATE AND LOAD/FILTER DATA ---

# Retrieve state variables
selected_area = st.session_state.get('pricearea')
selected_groups = st.session_state.get('group', [])
data_type = st.session_state.get('data_type', "Production")
period_level = st.session_state.get('period_level')

st.title(f"Energy ({data_type} Analysis)")

# Calculate the effective date range
start_dt, end_dt = get_elhub_date_range()
start_str = start_dt.strftime('%Y-%m-%d')
end_str = end_dt.strftime('%Y-%m-%d')

# Load the base data using the consolidated loader
with st.spinner(f"Loading {data_type} data..."):
    # *** CRITICAL FIX: Use conditional loading based on data_type ***
    if data_type == "Production":
        df_raw = data_loader.load_and_clean_production_data()
    elif data_type == "Consumption":
        df_raw = data_loader.load_and_clean_consumption_data()
    else:
        st.error("Invalid data type selected.")
        st.stop()


if df_raw.empty:
    st.warning("No data found or failed to load data from MongoDB.", icon="❌")
    st.stop()

# --- DATA INTEGRITY CHECK ---
# Check if the needed grouping column is present after loading/cleaning
if GROUP_COLUMN not in df_raw.columns:
    st.error(
        f"""
        **FATAL ERROR: Grouping column missing.**
        
        The analysis requires the `{GROUP_COLUMN}` column for filtering, but it is missing.
        
        **Action Required:** This indicates your MongoDB collection for `{data_type}` contains aggregated data (only totals). 
        Please verify the Spark ingestion logic in your notebook to ensure 
        the raw grouping column (`productiongroup`/`consumptiongroup`) was not accidentally dropped 
        by an aggregation function before writing to MongoDB.
        """
    )
    st.stop()
# -----------------------------


# --- DISPLAY CONTEXT BOX (Scope Message) ---
groups_text = ', '.join([g.capitalize() for g in selected_groups])

st.info(
    f"""
    **Current relevant filter settings:**
    
    * **Data Type:** {data_type.capitalize()}
    * **Price Area:** {selected_area}
    * **Energy Sources:** {groups_text}
    * **Time Period:** {period_level} ({start_str} to {end_str})

    *Inherited from the analysis scope filters configured in the sidebar.
    """
)

# --- APPLY ALL FILTERS ---
try:
    # 1. Date filter
    df_filtered_time = df_raw[
        (df_raw["starttime"] >= start_dt) & 
        (df_raw["starttime"] <= end_dt)
    ]

    # 2. Area filter
    df_filtered_area = df_filtered_time[
        df_filtered_time["pricearea"] == selected_area
    ]

    # 3. Group filter
    # Use the universally named GROUP_COLUMN
    df_final = df_filtered_area[
        df_filtered_area[GROUP_COLUMN].isin(selected_groups)
    ].copy()

except Exception as e:
    st.error(f"Error applying filters: {e}")
    st.code(f"Traceback: {e}")
    st.stop()


if df_final.empty:
    st.warning(f"No data found for the selected combination within the range: {selected_area} and groups: {selected_groups}.")
    st.stop()


left_column, right_column = st.columns(2)

# =======================================================
# Left Column: Mix (Pie Chart) - Total over the period
# =======================================================
with left_column:
    st.subheader(f"Total {data_type} Mix (Selected Period)")
    
    # Aggregate ALL data in the filtered set for the period mix
    production_by_group = df_final.groupby(GROUP_COLUMN)["quantitykwh"].sum().reset_index()
    
    fig_pie = px.pie(
        production_by_group, 
        values='quantitykwh', 
        names=GROUP_COLUMN, 
        color=GROUP_COLUMN, 
        color_discrete_map=COLOR_MAP,
        title=f"Total {data_type} Share in {selected_area}",
        template="plotly_white",
        hole=0.3
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

# =======================================================
# Right Column: Trend (Line Chart) - Time Series
# =======================================================
with right_column:
    st.subheader(f"Hourly {data_type} Trend ({period_level} Granularity)")
    
    # Determine aggregation level for the trend plot
    agg_level = "M" if period_level == "Annual" else "H"
        
    df_plot = df_final.set_index('starttime')
    
    # Resample and sum the data for the required granularity
    if agg_level == "M":
        pivot_data = df_plot.groupby(GROUP_COLUMN)['quantitykwh'].resample(agg_level).sum().unstack(level=0)
    else:
        pivot_data = df_plot.pivot_table(
            values="quantitykwh",
            index="starttime",
            columns=GROUP_COLUMN,
            aggfunc="sum"
        )
    
    plot_index = pivot_data.index
    
    # Define plot_data outside the conditional blocks (ensuring it's always defined)
    plot_data = pivot_data # Use the aggregated data for plotting
    
    fig_line = px.line(
        plot_data,
        x=plot_index,
        y=plot_data.columns,
        color_discrete_map=COLOR_MAP,
        title=f"Energy {data_type} by Source in {selected_area}",
        labels={'value': f'{data_type} (kWh)', 'starttime': 'Time', GROUP_COLUMN: 'Group'},
        template="plotly_white"
    )
    fig_line.update_layout(
        xaxis_title="Time", 
        yaxis_title=f"{data_type} (kWh)",
        legend_title="Energy Group"
    )

    st.plotly_chart(fig_line, use_container_width=True)


with st.expander("Data Source and Pipeline Documentation"):
    st.markdown(f"""
    The data displayed on this page is hourly electricity **{data_type.lower()}** data for Norway's five price areas.
    
    **Source API:** Elhub's Energy Data API.
    
    **Data Pipeline (ETL):** The data was extracted via API, transformed using Apache Spark, and loaded into a MongoDB cluster. This Streamlit application connects directly to the MongoDB cluster.
    """)
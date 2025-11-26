import streamlit as st
import pandas as pd
from utilities.app_state import render_app_state_controls 
from utilities import functions
import requests

st.set_page_config(
    page_title="Weather Data Overview",
    layout="wide"
)

# --- 1. RENDER GLOBAL CONTROLS IN SIDEBAR ---
with st.sidebar:
    render_app_state_controls()

# --- 2. ACCESS GLOBAL STATE ---
selected_area = st.session_state.get('pricearea')


# --- INITIAL CHECKS ---
if not selected_area:
    st.info("The global Price Area selector is not yet initialized. Please use the sidebar on the Production Explorer page.")
    st.stop() 

st.title("Weather Data Summary (2021–2024)")

# --- DATA FETCHING ---
try:
    with st.spinner(f"Loading weather data for {selected_area}..."):
        df = functions.download_weather_data(selected_area) 
    
    if df is None:
        st.error("Failed to retrieve data from the API. Check network connection or API service status.")
        st.stop()

    if df.empty:
        st.warning(f"No weather data available for {selected_area} in the 2021-2024 range.")
        st.stop()
        
except requests.exceptions.RequestException as e:
    st.error(f"API Connection Error: Could not connect to weather service. Error: {e}")
    st.stop()
    
except Exception as e:
    st.error(f"Unexpected Error during data fetching: {e}")
    st.stop()
    
# Reset index to make 'time' a column for the first table preview
df_display = df.reset_index()

# Extract Year-Month strings for dynamic selection
df_display['year_month'] = df_display['time'].dt.strftime('%Y-%m')
available_months = sorted(df_display['year_month'].unique())

st.subheader(f"Hourly Statistics for {selected_area}")
selected_year_month = st.selectbox(
    "Select Month and Year for Trend Preview",
    options=available_months
)

# Filter the data for the selected month
selected_month_data = df[df.index.strftime('%Y-%m') == selected_year_month]

# Prepare a table: one row per numeric column
table_data = []
for col in df.columns: 
    # Use the filtered data for the stats
    if pd.api.types.is_numeric_dtype(selected_month_data[col]):
        
        chart_data = selected_month_data[col].tolist()
        
        row = {
            "Column": col,
            f"Trend in {selected_year_month}": chart_data,
            "Min": selected_month_data[col].min(),
            "Max": selected_month_data[col].max(),
            "Mean": selected_month_data[col].mean().round(2),
            "Median": selected_month_data[col].median().round(2),
            "Std": selected_month_data[col].std().round(2)
        }
        table_data.append(row)

df_table = pd.DataFrame(table_data)

# Configure the LineChartColumn
column_config = {
    f"Trend in {selected_year_month}": st.column_config.LineChartColumn(
        label=f"{selected_year_month} Hourly Trend",
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
import streamlit as st
import pandas as pd
from utilities.app_state import render_app_state_controls 
from utilities import functions
import requests

st.set_page_config(
    page_title="Weather Table with Line Charts",
    layout="wide"
)

# --- 1. RENDER GLOBAL CONTROLS IN SIDEBAR ---
with st.sidebar:
    render_app_state_controls()

# --- 2. ACCESS GLOBAL STATE ---
# Get the globally selected area (used for weather data source)
selected_area = st.session_state.get('price_area')


# --- INITIAL CHECKS ---
if not selected_area:
    st.info("The global Price Area selector is not yet initialized. Please use the sidebar.")
    st.stop() 

st.title("Weather Data Summary")

# --- DISPLAY CONTEXT BOX ---
st.info(
    f"""
    **Analysis Scope** (by the sidebar configuration):
    
    * **Weather Location (Price Area):** **{selected_area}**
    """
)


try:
    # This calls the fast, cached function from functions.py. 
    df = functions.download_weather_data(selected_area) 
    
    if df is None:
        st.error("Failed to retrieve data from the API. Check network connection or API service status.")
        st.stop()

    if df.empty:
        st.warning(f"No weather data available for {selected_area}.")
        st.stop()
        
except requests.exceptions.RequestException as e:
    st.error(f"API Connection Error: Could not connect to weather service. Error: {e}")
    st.stop()
    
except Exception as e:
    st.error(f"Unexpected Error during data fetching: {e}")
    st.stop()
    
# Reset index to make 'time' a column for the first table preview
df_display = df.reset_index()


# Identify the first month's data
first_month_num = df_display['time'].dt.month.min()
first_month_name = pd.to_datetime(f'2021-{first_month_num}-01').strftime('%B')
first_month = df[df.index.month == first_month_num]

# Prepare a table: one row per numeric column
table_data = []
for col in df.columns: 
    # Use the filtered data for the stats
    if pd.api.types.is_numeric_dtype(first_month[col]):
        
        chart_data = first_month[col].tolist()
        
        row = {
            "Column": col,
            f"{first_month_name} Trend": chart_data,
            "Min": first_month[col].min(),
            "Max": first_month[col].max(),
            "Mean": first_month[col].mean().round(2),
            "Median": first_month[col].median().round(2),
            "Std": first_month[col].std().round(2)
        }
        table_data.append(row)

df_table = pd.DataFrame(table_data)

# Configure the LineChartColumn
column_config = {
    f"{first_month_name} Trend": st.column_config.LineChartColumn(
        label=f"{first_month_name} Hourly Trend",
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
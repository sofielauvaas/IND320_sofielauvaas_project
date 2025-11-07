import streamlit as st
import pandas as pd
import functions # Import the helper functions file
import requests # Needed for API exception handling

st.set_page_config(
    page_title="Weather Table with Line Charts",
    layout="wide"
)

st.title("Weather Data Summary")

# --- 1. CHECK SESSION STATE & FETCH DATA ---

if 'weather_source_area' not in st.session_state:
    st.info("Please go back to the 'Elhub Production Data' page to select a Price Area first.")
    st.stop() 

# Get the selected area (used for weather data source)
selected_area = st.session_state['weather_source_area']
# Get the selected groups (used for context display)
selected_groups = st.session_state.get('elhub_selected_groups', ['No groups selected']) 

# --- DISPLAY CONTEXT BOX (Replaces subheader) ---
groups_text = ', '.join([g.capitalize() for g in selected_groups])
st.info(
    f"""
    **Chosen parameters from Elhub Production Page:**
    
    * **Weather Location (Price Area):** {selected_area}
    """
)


try:
    # This calls the fast, cached function from functions.py
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
first_month = df[df.index.month == first_month_num]

# Prepare a table: one row per numeric column
table_data = []
for col in df.columns: 
    if pd.api.types.is_numeric_dtype(first_month[col]):
        
        chart_data = first_month[col].tolist()
        
        row = {
            "Column": col,
            "First Month": chart_data, 
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
    "First Month": st.column_config.LineChartColumn(
        label="First Month Hourly Trend",
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
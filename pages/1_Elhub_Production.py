import streamlit as st
from utilities import functions 
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Production Data Explorer", layout="wide")
st.title("Norway Power Production Data (2021-2024)")
st.caption("Hourly production data for all Norwegian price areas, sourced from Elhub API.")

# Load the data using the cached function from functions.py
try:
    # NOTE: If this fails to load the new data, try clearing Streamlit's cache
    # by clicking Rerun (or Cmd+R) and selecting 'Clear cache and rerun'.
    df = functions.load_data_from_mongo() 
except Exception as e:
    st.error(f"Failed to load data from MongoDB. Error: {e}")
    st.stop() 

if df.empty:
    st.warning("No data found in the MongoDB collection.")
    st.stop()


# --- CRITICAL FIX: SANITIZE AND STANDARDIZE COLUMN NAMES ---
# This block aggressively cleans column names to resolve case and whitespace issues.

# 1. Clean up column names: strip whitespace and convert all to lowercase
df.columns = df.columns.str.strip().str.lower()

# 2. Re-verify the existence of 'productiongroup' and enforce renaming if 'productionGroup' (or similar) was present
# This accounts for MongoDB loading the original field name despite our Spark changes.
if 'productionGroup' in df.columns:
    df.rename(columns={'productionGroup': 'productiongroup'}, inplace=True)
    
# 3. Final check and error if the critical column is still missing
if 'productiongroup' not in df.columns:
    st.error(f"FATAL ERROR: The required column 'productiongroup' could not be found after cleanup. Found columns: {list(df.columns)}")
    st.stop()

# 4. Data Preparation (Ensure starttime is datetime and create month_name)
if 'starttime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['starttime']):
    df['starttime'] = pd.to_datetime(df['starttime'], utc=True, errors='coerce') 

df.dropna(subset=['starttime'], inplace=True)
# Use YYYY-MM format to distinguish between the same month in different years
df['month_name'] = df['starttime'].dt.strftime('%Y-%m') 
# -----------------------------------------------------------


# Get available options for filters
priceareaS = sorted(df["pricearea"].unique().tolist())
productiongroupS = sorted(df["productiongroup"].unique().tolist())
MONTH_NAMES = df['month_name'].unique().tolist() 

COLOR_MAP = {
    'hydro': '#035397',
    'wind': '#128264', 
    'solar': '#f9c80e',
    'thermal': '#546e7a',
    'other': '#9dc183'
}

left_column, right_column = st.columns(2)

# Left Column: Price Area Selection and Yearly Production Plot
with left_column:
    st.subheader("Production Mix")
    
    selected_area = st.radio(
        "Select Price Area",
        priceareaS,
        index=0,
        horizontal=True
    )
    
    st.session_state['pricearea'] = selected_area
    
    area_data = df[df["pricearea"] == selected_area]
    production_by_group = area_data.groupby("productiongroup")["quantitykwh"].sum().reset_index()
    
    fig_pie = px.pie(
        production_by_group, 
        values='quantitykwh', 
        names='productiongroup',
        color='productiongroup',
        color_discrete_map=COLOR_MAP,
        title=f"Total Production Share in {selected_area} ({df['starttime'].dt.year.min()}-{df['starttime'].dt.year.max()})",
        template="plotly_white",
        hole=0.3
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

# Right Column: Monthly Trend Plot
with right_column:
    st.subheader("Hourly Production Trend")
    
    selected_month_name = st.selectbox(
        "Select Month (Year-Month)",
        MONTH_NAMES
    )

    st.session_state['elhub_selected_month'] = selected_month_name
    
    selected_groups = st.pills(
        "Select Production Groups to Include",
        productiongroupS,
        default=productiongroupS,
        selection_mode="multi"
    )
    
    st.session_state['productiongroup'] = selected_groups
    
    if not selected_groups:
        st.warning("Please select at least one production group.")
        filtered_data = pd.DataFrame()
    else:
        filtered_data = df[
            (df["pricearea"] == selected_area) &
            (df["productiongroup"].isin(selected_groups)) &
            (df["month_name"] == selected_month_name)
        ]
    
    if filtered_data.empty:
        st.warning(f"No data found for the selected combination: {selected_area}, {selected_month_name}.")
    else:
        pivot_data = filtered_data.pivot_table(
            values="quantitykwh",
            index="starttime",
            columns="productiongroup",
            aggfunc="sum"
        )
        
        fig_line = px.line(
            pivot_data,
            x=pivot_data.index,
            y=pivot_data.columns,
            color_discrete_map=COLOR_MAP,
            title=f"Hourly Production in {selected_area} - {selected_month_name}",
            labels={'value': 'Production (kWh)', 'starttime': 'Time', 'productiongroup': 'Group'},
            template="plotly_white"
        )
        fig_line.update_layout(
            xaxis_title="Time", 
            yaxis_title="Production (kWh)",
            legend_title="Production Group"
        )

        st.plotly_chart(fig_line, use_container_width=True)


with st.expander("Data Source and Pipeline Documentation"):
    st.markdown("""
    The data displayed on this page is hourly electricity production data for Norway's five price areas.
    
    **Source API:** Elhub's Energy Data API.
    
    **Data Pipeline (ETL):** The data was extracted via API, transformed using Apache Spark, and loaded into a MongoDB cluster. This Streamlit application connects directly to the MongoDB cluster.
    """)
import streamlit as st
# FIX 1: Update import path to use your utilities folder
from utilities import functions 
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Production Data Explorer", layout="wide")
st.title("Norway Power Production Data (2021)")
st.caption("Hourly production data for all Norwegian price areas, sourced from Elhub API.")

# Load the data using the cached function from functions.py
try:
    df = functions.load_data_from_mongo() 
except Exception as e:
    st.error(f"Failed to load data from MongoDB. Error: {e}")
    st.stop() 

if df.empty:
    st.warning("No data found in the MongoDB collection.")
    st.stop()


# Get available options for filters
PRICE_AREAS = sorted(df["pricearea"].unique().tolist())
PRODUCTION_GROUPS = sorted(df["productiongroup"].unique().tolist())
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
    st.subheader("Yearly Production Mix")
    
    selected_area = st.radio(
        "Select Price Area",
        PRICE_AREAS,
        index=0,
        horizontal=True
    )
    
    # FIX 2: Write the selected area to the CANONICAL GLOBAL STATE KEY
    st.session_state['price_area'] = selected_area
    
    area_data = df[df["pricearea"] == selected_area]
    production_by_group = area_data.groupby("productiongroup")["quantitykwh"].sum().reset_index()
    
    fig_pie = px.pie(
        production_by_group, 
        values='quantitykwh', 
        names='productiongroup',
        color='productiongroup',
        color_discrete_map=COLOR_MAP,
        title=f"Total Production Share in {selected_area} (2021)",
        template="plotly_white",
        hole=0.3
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

# Right Column: Monthly Trend Plot
with right_column:
    st.subheader("Hourly Production Trend")
    
    selected_month_name = st.selectbox(
        "Select Month",
        MONTH_NAMES
    )
    # Keeping this local state key is fine as it's only used on this page
    st.session_state['elhub_selected_month'] = selected_month_name
    
    selected_groups = st.pills(
        "Select Production Groups to Include",
        PRODUCTION_GROUPS,
        default=PRODUCTION_GROUPS,
        selection_mode="multi"
    )
    
    # FIX 3: Write the selected groups to the CANONICAL GLOBAL STATE KEY
    st.session_state['production_group'] = selected_groups
    
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
        st.warning("No data found for the selected combination.")
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
            title=f"Hourly Production in {selected_area} - {selected_month_name} 2021",
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
    The data displayed on this page is hourly electricity production data for Norway's five price areas (NO1, NO2, NO3, NO4, NO5) for the calendar year 2021.
    
    **Source API:** Elhub's Energy Data API.
    
    **Data Pipeline (ETL):** The data was extracted via API, transformed using Apache Spark, and loaded into a MongoDB cluster. This Streamlit application connects directly to the MongoDB cluster.
    """)
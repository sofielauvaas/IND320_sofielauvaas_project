import streamlit as st
from pymongo import MongoClient
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Production Data Explorer", layout="wide")
st.title("Norway Power Production Data (2021)")
st.caption("Hourly production data for all Norwegian price areas, sourced from Elhub API.")

# 1. MongoDB Connection (Cached)
# Cache the database connection to avoid reconnecting on every interaction
@st.cache_resource
def init_connection():
    uri = st.secrets["mongodb"]["uri"]
    return MongoClient(uri)

client = init_connection()

# Cache the data loading function to avoid reloading on every interaction
# Set a TTL of 1 hour (3600 seconds) for the cached data
@st.cache_data(ttl=3600)
def load_data_from_mongo():
    db = client["IND320_elhub_db"]
    collection = db["production_data_2021"] 
    items = list(collection.find({}, {'_id': 0})) # Exclude the MongoDB-generated _id field
    df = pd.DataFrame(items)
    df['starttime'] = pd.to_datetime(df['starttime'])
    
    # Add a 'month_name' column for easier month selection
    df['month_name'] = df['starttime'].dt.strftime("%B")
    
    return df

# Load the data
# AI suggested adding error handling here, because database connections can fail
try:
    df = load_data_from_mongo()
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

# Define the custom color map
COLOR_MAP = {
    'hydro': '#035397',
    'wind': '#128264', 
    'solar': '#f9c80e',
    'thermal': '#546e7a', 
    'other': '#9dc183'
}


# 2. Main Content Layout 
left_column, right_column = st.columns(2)

# Left Column: Price Area Selection and Yearly Production Plot
with left_column:
    st.subheader("Yearly Production Mix")
    
    # Radio buttons for Price Area selection
    selected_area = st.radio(
        "Select Price Area",
        PRICE_AREAS,
        index=0,
        horizontal=True
    )
    
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
    
    # Selection element (selectbox) for Month
    selected_month_name = st.selectbox(
        "Select Month",
        MONTH_NAMES
    )
    
    # Using st.pills for multi-selection of Production Groups
    selected_groups = st.pills(
        "Select Production Groups to Include",
        PRODUCTION_GROUPS,
        default=PRODUCTION_GROUPS, # Default to all groups selected
        selection_mode="multi"     # Crucial for allowing multiple selections
    )
    
    # Filter data based on ALL selections
    if not selected_groups:
        st.warning("Please select at least one production group.")
        filtered_data = pd.DataFrame()
    else:
        # Filter using the cached 'month_name' column
        filtered_data = df[
            (df["pricearea"] == selected_area) &
            (df["productiongroup"].isin(selected_groups)) &
            (df["month_name"] == selected_month_name)
        ]
    
    if filtered_data.empty:
        st.warning("No data found for the selected combination.")
    else:
        # Create a pivot table: time as index, production groups as columns
        pivot_data = filtered_data.pivot_table(
            values="quantitykwh",
            index="starttime",
            columns="productiongroup",
            aggfunc="sum"
        )
        
        # Create line plot using Plotly
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


# 3. Data Source Documentation (Expander) 
st.markdown("---") 
with st.expander("Data Source and Pipeline Documentation"):
    st.markdown("""
    The data displayed on this page is hourly electricity production data for Norway's five price areas (NO1, NO2, NO3, NO4, NO5) for the calendar year 2021.
    
    **Source API:** Elhub's Energy Data API, specifically the `PRODUCTION_PER_GROUP_MBA_HOUR` dataset. You can find the main API documentation here: [api.elhub.no](https://api.elhub.no/).
    
    **Data Pipeline (ETL):**
    1.  **Extract:** Data retrieved from the Elhub API.
    2.  **Transform:** Processed using **Apache Spark** for cleaning, aggregation, and reformatting.
    3.  **Load:** Initially loaded into a **Cassandra** database, and then transferred to a remote **MongoDB** cluster.
    
    This Streamlit application connects directly to the **MongoDB** cluster to retrieve and visualize the final, clean dataset.
    """)
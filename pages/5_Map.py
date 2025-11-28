import streamlit as st
import folium
from streamlit_folium import st_folium
import json
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
# Import the necessary utility for sidebar controls and Elhub access
from utilities.app_state import render_app_state_controls 
from utilities import functions # Contains load_elhub_data

st.set_page_config(layout="wide")
st.title("Dynamic Energy Map Visualization")
st.markdown("Use the sidebar to select the data type and groups to visualize the mean energy quantity across price areas.")

# Define the standardized group column name (must match functions.py)
GROUP_COLUMN = 'group' 

# --- UTILITIES ---

def normalize_area_name(name):
    """Ensures area names like 'NO 1' are normalized to 'NO1' for data matching."""
    if isinstance(name, str):
        return name.replace(" ", "")
    return name

def get_global_date_range():
    """Calculates the filtering date range based on the global session state."""
    # This logic is adapted here for independence from the main Explorer page
    period_level = st.session_state.get('period_level')
    
    if period_level == "Annual":
        year = st.session_state.get('selected_year', datetime.now().year)
        start_dt = datetime(year, 1, 1).date()
        end_dt = datetime(year, 12, 31).date()
    elif period_level == "Custom Date Range":
        start_dt = st.session_state.get('start_date', datetime(2021, 1, 1).date())
        end_dt = st.session_state.get('end_date', datetime.now().date())
    else: # Default/Monthly uses simple date extraction for this page
        start_dt = st.session_state.get('start_date', datetime(2021, 1, 1).date())
        end_dt = st.session_state.get('end_date', datetime.now().date())

    return start_dt, end_dt

@st.cache_data(show_spinner="Calculating Mean kWh from MongoDB...")
def calculate_mean_data(df_raw, data_type, selected_group, start_date, end_date):
    """
    Filters ELHUB data by time/group, and calculates the mean quantity per price area 
    for Choropleth coloring.
    """
    
    if df_raw.empty or GROUP_COLUMN not in df_raw.columns:
        return pd.DataFrame(), 0 

    # 1. Apply Time Filter
    # Convert global date objects to Pandas Timestamps for filtering the raw data index
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # CRITICAL: Filter against the naive date component
    df_period = df_raw[(df_raw["starttime"].dt.date >= start_dt.date()) & 
                       (df_raw["starttime"].dt.date <= end_dt.date())].copy()


    # 2. Apply Group Filter (The Map visualizes the aggregation of a SINGLE selected group)
    df_filtered = df_period[df_period[GROUP_COLUMN] == selected_group.lower()]

    if df_filtered.empty:
        return pd.DataFrame(), 0

    # 3. Calculate Mean per Price Area (The Choropleth input)
    means_df = df_filtered.groupby("pricearea", as_index=False)["quantitykwh"].mean()
    
    means_df.rename(columns={'quantitykwh': 'mean_quantity'}, inplace=True)
    
    total_quantity_sum = df_filtered['quantitykwh'].sum()

    return means_df, total_quantity_sum


# --- 1. File Loading and Normalization (GeoJSON) ---

# Define path to the GeoJSON file: project_root/data/file.geojson
geojson_path = Path(__file__).parent.parent / "data" / "file.geojson"

try:
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
except FileNotFoundError:
    st.error(f"Error: GeoJSON boundary file not found at {geojson_path}. Please ensure 'file.geojson' is in your 'data' folder.")
    st.stop()

# Normalize names in the GeoJSON data
for feature in geojson_data["features"]:
    raw_name = feature["properties"].get("ElSpotOmr") 
    feature["properties"]["ElSpotOmrNorm"] = normalize_area_name(raw_name)


# --- 2. Session State Initialization ---

if "last_clicked" not in st.session_state:
    st.session_state.last_clicked = (63.0, 10.5) # Default coordinates
if "selected_area" not in st.session_state:
    st.session_state.selected_area = None # The NO1, NO2, etc. ID
if 'map_zoom' not in st.session_state:
    st.session_state['map_zoom'] = 5.5

# --- 3. UI Controls and Data Access (Using Global State) ---

# Render the centralized controls in the sidebar
with st.sidebar:
    render_app_state_controls()

# Access global filters (set by the sidebar)
data_type = st.session_state.get('data_type', "Production")
all_selected_groups = st.session_state.get('group', ["hydro"]) # List of ALL groups selected in pills

# Get Time Range
start_date, end_date = get_global_date_range()
period_level = st.session_state.get('period_level', "Annual")

# --- DISPLAY CONTEXT BOX (Top Info Box) ---
groups_text = ', '.join([g.capitalize() for g in all_selected_groups])
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')
st.info(
    f"""
    **Current relevant filter settings:**
    
    * **Data Type:** {data_type}
    * **Energy Sources: :** {groups_text}
    * **Time Period:** {period_level} ({start_str} to {end_str})

    *Inherited from the analysis scope filters configured in the sidebar.
    """
)

# --- Local Group Selection ---
# User chooses which SINGLE group (from the global selection) to display on the map
st.subheader("Map Visualization Control")
selected_group = st.selectbox(
    "Select Energy Group to Visualize",
    options=all_selected_groups,
    index=0
)



# --- Load Live Data ---

# Load the base hourly energy data
with st.spinner(f"Loading {data_type} data from MongoDB..."):
    df_raw_energy = functions.load_elhub_data(data_type) 
    
if df_raw_energy.empty:
    st.warning("Energy data loading failed or returned empty. Cannot generate Choropleth.")
    st.stop()

# Calculate the mean quantity per price area (Choropleth values)
# FIX: Pass the data_type variable into the function definition
means_df, total_quantity_sum = calculate_mean_data(df_raw_energy, data_type, selected_group, start_date, end_date)

if means_df.empty:
     st.warning(f"No data found for the selected group ({selected_group.capitalize()}) and time range.")
     st.stop()


means_dict = dict(zip(means_df["pricearea"], means_df["mean_quantity"]))

# --- 5. Map Rendering and Interaction Setup ---

col_map, col_info = st.columns([3, 1])

with col_map:
    # Center map on Norway's approximate center or last clicked location
    lat, lon = st.session_state.last_clicked
    m = folium.Map(location=[lat, lon], zoom_start=st.session_state['map_zoom'], tiles='cartodbpositron')

    # Calculate thresholds for the Choropleth color scale
    vmin = means_df["mean_quantity"].min()
    vmax = means_df["mean_quantity"].max()
    thresholds = np.linspace(vmin, vmax, 6).tolist() 
    if len(thresholds) < 2:
        thresholds = [vmin-1e-6, vmin, vmax+1e-6]

    # Add the main Choropleth layer (Color-coded areas)
    folium.Choropleth(
        geo_data=geojson_data,
        name="Energy Data Choropleth",
        data=means_df,
        columns=["pricearea", "mean_quantity"], # Use the new column name
        key_on="feature.properties.ElSpotOmrNorm", 
        fill_color="YlGnBu",
        fill_opacity=0.7,
        line_opacity=0.3,
        line_color="black",
        legend_name=f"Mean {data_type} of {selected_group.capitalize()} (kWh)",
        threshold_scale=thresholds,
        nan_fill_color="lightgray"
    ).add_to(m)

    # Add Tooltips for identifying the area on hover
    folium.GeoJson(
        geojson_data,
        name="tooltips",
        tooltip=folium.GeoJsonTooltip(
            fields=["ElSpotOmrNorm"],
            aliases=["Price area:"],
            labels=True,
            sticky=True
        ),
        style_function=lambda _: {"color": "transparent", "weight": 0, "fillOpacity": 0}
    ).add_to(m)

    # Highlight the currently selected area boundary
    if st.session_state.selected_area:
        def highlight_style(feat):
            is_selected = feat["properties"]["ElSpotOmrNorm"] == st.session_state.selected_area
            # Use a thicker, dashed red line for the boundary
            return {"color": "#d62728", "weight": 4, "fillOpacity": 0, "dashArray": "5, 5"} if is_selected else {"color": "transparent", "weight": 0, "fillOpacity": 0}

        folium.GeoJson(
            geojson_data,
            name="selected_highlight",
            style_function=highlight_style,
            tooltip=None,
        ).add_to(m)

    # Add a Marker for the Exact Clicked Location
    if st.session_state.last_clicked != (63.0, 10.5):
        lat, lon = st.session_state.last_clicked
        folium.Marker(
            location=[lat, lon],
            popup=f"Clicked: {lat:.4f}, {lon:.4f}",
            icon=folium.Icon(color='red', icon='fa-crosshairs', prefix='fa')
        ).add_to(m)

    # Display the map and capture interaction data
    map_data = st_folium(m, width=950, height=600, returned_objects=["last_clicked", "zoom"])

# --- 6. Handle Map Interaction and Selection ---

if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    
    current_click = (lat, lon)
    
    # Only process if the click is new
    if current_click != st.session_state.last_clicked:
        
        # Update zoom level
        if map_data.get('zoom') is not None:
            st.session_state['map_zoom'] = map_data['zoom']
            
        # Store the clicked point
        st.session_state.last_clicked = current_click

        # Determine which price area polygon contains the click
        point = Point(lon, lat)
        clicked_area = None
        
        for feature in geojson_data["features"]:
            geom = shape(feature["geometry"])
            if isinstance(geom, (Polygon, MultiPolygon)) and geom.contains(point):
                clicked_area = feature["properties"]["ElSpotOmrNorm"]
                break
        
        # Update the selected area ID and trigger a rerun to update the map highlight
        st.session_state.selected_area = clicked_area
        st.rerun() 
        

# --- 7. Display Selection Status ---

with col_info:
    st.subheader("Selection Details")
    
    current_area = st.session_state.selected_area
    lat, lon = st.session_state.last_clicked
    
    st.write(f"Price Area: **{current_area or 'n/a'}**")
    st.write(f"Clicked Location: {lat:.4f}, {lon:.4f}")

    if current_area:
        mean_val = means_dict.get(current_area)
        if mean_val is not None:
            # We display the units based on the assumption that the data is kWh per hour * days
            st.success(f"Mean {data_type} Quantity: **{mean_val:.2f} kWh**")
        else:
            st.warning(f'Mean {data_type} Quantity: n/a')
    else:
        st.info("Click a region on the map.")
    
    st.markdown("---")
    st.write("Choropleth Filters:")
    st.text(f"Type: {data_type}")
    st.text(f"Group: {selected_group.capitalize()}")
    st.text(f"Time Range: {start_date} to {end_date}")
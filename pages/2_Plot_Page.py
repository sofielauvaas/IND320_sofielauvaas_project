import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

st.title("Plot Page")

# Load CSV with caching
@st.cache_data
def load_data():
    df = pd.read_csv("data/open-meteo-subset.csv")
    # Ensure the 'time' column is a datetime object
    df["time"] = pd.to_datetime(df["time"]) 
    return df

df = load_data()

# Define the wind direction column
WIND_DIRECTION_COL = "wind_direction_10m (°)" 

# Column selection excluding 'time' and wind direction
columns = ["All columns"] + [c for c in df.columns if c != "time" and c != WIND_DIRECTION_COL]
selected_col = st.selectbox("Select a column", columns)

# Month selection slider
months = sorted(df["time"].dt.month.unique())
selected_months = st.select_slider(
    "Select month range",
    options=months,
    value=(months[0], months[0]),
    format_func=lambda x: pd.to_datetime(f"2023-{x}-01").strftime("%B")
)

df["time"] = pd.to_datetime(df["time"]) 

# Filter data by selected months
subset = df[(df["time"].dt.month >= selected_months[0]) & (df["time"].dt.month <= selected_months[1])]
df_filtered = subset.set_index("time")

# Optional normalization
normalize = st.checkbox("Normalize numeric columns (0-1 scale)")
if normalize:
    numeric_cols = df_filtered.select_dtypes(include="number").columns
    df_to_plot = df_filtered.copy()
    df_to_plot[numeric_cols] = (df_filtered[numeric_cols] - df_filtered[numeric_cols].min()) / \
                               (df_filtered[numeric_cols].max() - df_filtered[numeric_cols].min())
else:
    df_to_plot = df_filtered
    
# Create a version of df_to_plot that excludes wind direction for the 'All columns' line plot
df_for_line_plot = df_to_plot.drop(columns=[WIND_DIRECTION_COL], errors='ignore')


# Rename months for title
month_names = (
    pd.to_datetime(f"2023-{selected_months[0]}-01").strftime("%B"),
    pd.to_datetime(f"2023-{selected_months[1]}-01").strftime("%B")
)

# Plot the selected column(s)
if selected_col == "All columns":
    st.subheader(f"Data Summary from {month_names[0]} to {month_names[1]}")
    
    #!!! Use st.columns to place charts side-by-side
    # Create two columns, giving the line chart more space (e.g., 60% vs 40%)
    col1, col2 = st.columns([3, 2]) 

    # Column 1: Line Chart
    with col1:
        st.caption("All columns (excluding Wind Direction) at hourly resolution")
        st.line_chart(df_for_line_plot)

    # Column 2: Polar Plot
    if WIND_DIRECTION_COL in subset.columns:
        with col2:
            st.caption("Wind Direction Distribution")
            
            direction_data = subset[WIND_DIRECTION_COL].dropna()
            angles = np.deg2rad(direction_data)
            
            # Create Matplotlib figure with polar projection
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
            
            # Plot a histogram (bins=36 for 10-degree intervals)
            ax.hist(angles, bins=36, color='#035397', alpha=0.8) # Using a theme color
            
            # Format for compass directions
            ax.set_theta_zero_location("N") # Set 0 degrees (North) to the top
            ax.set_theta_direction(-1)      # Set rotation to clockwise
            ax.set_title(f"Wind Direction", va='bottom')
            ax.set_rlabel_position(22.5) 
            
            # Display the plot in Streamlit
            st.pyplot(fig)
        
else:
    st.subheader(f"{selected_col} from {month_names[0]} to {month_names[1]}")
    st.line_chart(df_to_plot[selected_col])
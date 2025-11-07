import streamlit as st
import functions
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Energy Time Series Analysis", layout="wide")
st.title("Time Series Decomposition and Frequency Analysis")

# --- DATA LOADING & DEPENDENCY CHECK ---

try:
    df_production = functions.load_data_from_mongo()
except Exception as e:
    st.error(f"Failed to load production data from MongoDB. Error: {e}")
    st.stop() 

PRICE_AREAS = sorted(df_production["pricearea"].unique().tolist())
PRODUCTION_GROUPS = sorted(df_production["productiongroup"].unique().tolist()) # All available groups

# Check session state for persistence and default if not set
default_area = st.session_state.get('elhub_selected_area', PRICE_AREAS[0])

# --- FETCH SELECTED GROUPS FROM PAGE 2 ---
# Get the list of groups selected on the main Production Plot page
selected_groups_from_page2 = st.session_state.get('elhub_selected_groups', PRODUCTION_GROUPS)

# Filter the options for the selectbox based on the selection from Page 2
if not selected_groups_from_page2:
    st.warning("Please select at least one Production Group on the Elhub Production Data page.")
    # Use all groups as a fallback, or stop the app. Using all as fallback is safer.
    analysis_options = PRODUCTION_GROUPS
else:
    analysis_options = selected_groups_from_page2
    
groups_text = ', '.join([g.capitalize() for g in selected_groups_from_page2])

# --- LOCAL SELECTORS (Define and Save Control) ---
st.subheader("Analysis Parameters")

col_area, col_group = st.columns(2)
with col_area:
    # Selector for the Price Area (can override the inherited choice)
    price_area = st.selectbox(
        "Select Price Area (Controls Weather Data Dependency):", 
        PRICE_AREAS, 
        index=PRICE_AREAS.index(default_area) if default_area in PRICE_AREAS else 0,
        key='stl_price_area_selector'
    ) 
    
    # Save the selected area back to the session state for all weather pages
    st.session_state['weather_source_area'] = price_area 
    st.session_state['elhub_selected_area'] = price_area
    
with col_group:
    # Selector for the single group required for STL/Spectrogram
    # FIX: Options are filtered to only those selected on Page 2
    selected_group_for_analysis = st.selectbox(
        "Select Single Production Group for Analysis:", 
        analysis_options,
        index=0
    )

# --- UPDATED CLARITY STATEMENT ---
st.info(
    f"""
    **Chosen parameters from Elhub Production Page:**
    
    * **Price Area:** {price_area}
    * **Available Groups:** {groups_text}
    """
)


# --- TABS AND ANALYSIS ---
tab1, tab2 = st.tabs(["STL Decomposition", "Spectrogram"])


# TAB 1: STL Decomposition
with tab1:
    st.subheader(f"STL Decomposition for {selected_group_for_analysis.capitalize()} Production in {price_area}")
    
    col_period, col_seasonal, col_trend = st.columns(3)
    with col_period:
        period = st.number_input("Period (e.g., 168 for weekly)", min_value=1, value=168, step=1, key='stl_period')
    with col_seasonal:
        seasonal = st.number_input("Seasonal Smoothing (Odd)", min_value=3, value=9, step=2, key='stl_seasonal')
    with col_trend:
        trend = st.number_input("Trend Smoothing (Odd)", min_value=3, value=241, step=2, key='stl_trend')
    
    
    try:
        # Call the function using consistent arguments: price_area and production_group
        fig_stl = functions.stl_decomposition_elhub(
            df_production, 
            price_area=price_area, 
            production_group=selected_group_for_analysis,
            period=period,
            seasonal=seasonal,
            trend=trend
        )
        st.plotly_chart(fig_stl, use_container_width=True)
    except Exception as e:
        st.error(f"Error during STL Decomposition. Please check parameters. Error: {e}")


# TAB 2: Spectrogram Analysis
with tab2:
    st.subheader(f"Spectrogram of {selected_group_for_analysis.capitalize()} Production in {price_area}")
    
    col_window, col_overlap = st.columns(2)
    with col_window:
        window_length = st.slider("Window Length (NPERSEG)", min_value=64, max_value=512, value=256, step=64, key='spec_window')
        st.caption("Length of each segment for analysis.")
    with col_overlap:
        overlap = st.slider("Overlap (NOVERLAP)", min_value=32, max_value=256, value=128, step=32, key='spec_overlap')
        st.caption("Number of samples overlapping between segments.")

    try:
        # Call the function using consistent arguments: price_area and production_group
        fig_spec = functions.create_spectrogram(
            df_production, 
            price_area=price_area, 
            production_group=selected_group_for_analysis,
            window_length=window_length,
            overlap=overlap
        )
        st.pyplot(fig_spec, use_container_width=True) # Assuming create_spectrogram returns a Matplotlib figure
    except Exception as e:
        st.error(f"Error during Spectrogram analysis. Error: {e}")
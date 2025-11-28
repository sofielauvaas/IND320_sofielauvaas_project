import streamlit as st
from datetime import datetime
import pandas as pd

# Global configuration lists (used for widget options and validation)
AREAS  = ["NO1", "NO2", "NO3", "NO4", "NO5"]
PRODUCTION_GROUPS = ["hydro", "wind", "solar", "thermal", "other"]
CONSUMPTION_GROUPS = ["cabin", "household", "primary", "secondary", "tertiary"]
DATA_TYPES = ["Production", "Consumption"]
TIME_PERIODS = ["Annual", "Monthly", "Custom Date Range"]

def get_available_years():
    """Returns a list of years from 2021 to 2024, aligning with available data."""
    # Note: Using 2025 in range() includes years up to 2024.
    return list(range(2021, 2025))

def _init_globals():
    """Initializes canonical session state variables with defaults."""
    
    current_year = 2024 # Use 2024 as the default year since it's the latest available
    
    st.session_state.setdefault("data_type", DATA_TYPES[0])
    st.session_state.setdefault("pricearea", AREAS[0])
    st.session_state.setdefault("group", PRODUCTION_GROUPS[:])
    st.session_state.setdefault("period_level", TIME_PERIODS[0])
    st.session_state.setdefault("selected_year", current_year)
    st.session_state.setdefault("selected_month", 1) # January
    st.session_state.setdefault("start_date", datetime(2021, 1, 1).date())
    st.session_state.setdefault("end_date", datetime(current_year, 12, 31).date())


def _sync_widgets_to_state():
    """Callback function to sync widget values to the canonical session state keys."""
    
    # Sync core selectors
    st.session_state["data_type"] = st.session_state["_data_type_selector"]
    st.session_state["pricearea"] = st.session_state["_area_selector"]
    st.session_state["period_level"] = st.session_state["_period_level_selector"]
    
    # Determine the correct group list based on data type
    group_options = PRODUCTION_GROUPS if st.session_state["data_type"] == "Production" else CONSUMPTION_GROUPS
    groups_selection = st.session_state["_group_selector"]
    
    # Check if the selection is valid for the current data type
    if not groups_selection or any(v not in group_options for v in groups_selection):
        # Default to the entire list of the new group type
        st.session_state["group"] = group_options 
    else:
        st.session_state["group"] = groups_selection

    # Sync time period selectors
    st.session_state["selected_year"] = st.session_state.get("_year_selector", st.session_state["selected_year"])
    
    # Handle month selection: convert month name index back to 1-based month number
    if "_month_selector" in st.session_state:
        month_names = list(pd.to_datetime(range(1, 13), format='%m').strftime('%B'))
        try:
            selected_month_name = st.session_state["_month_selector"]
            st.session_state["selected_month"] = month_names.index(selected_month_name) + 1
        except ValueError:
            # Fallback in case month name is unexpected
            st.session_state["selected_month"] = 1

    st.session_state["start_date"] = st.session_state.get("_start_date_selector", st.session_state["start_date"])
    st.session_state["end_date"] = st.session_state.get("_end_date_selector", st.session_state["end_date"])
    
    # Update query parameters for deep linking
    st.query_params.update(
        type=st.session_state["data_type"],
        area=st.session_state["pricearea"],
        groups=",".join(st.session_state["group"]),
        period=st.session_state["period_level"]
    )

def render_app_state_controls():
    """Renders the global application configuration controls in the sidebar, grouped by Area, Time, and Data Focus."""
    
    st.markdown("## Analysis Scope 🔎")
    _init_globals()

    # --- Setup Widget Keys for Canonical State ---
    st.session_state.setdefault("_data_type_selector", st.session_state["data_type"])
    st.session_state.setdefault("_area_selector", st.session_state["pricearea"])
    st.session_state.setdefault("_period_level_selector", st.session_state["period_level"])
    st.session_state.setdefault("_group_selector", st.session_state["group"])
    st.session_state.setdefault("_year_selector", st.session_state["selected_year"])
    
    month_names = list(pd.to_datetime(range(1, 13), format='%m').strftime('%B'))
    selected_month_name = month_names[st.session_state["selected_month"] - 1]
    st.session_state.setdefault("_month_selector", selected_month_name)
    
    st.session_state.setdefault("_start_date_selector", st.session_state["start_date"])
    st.session_state.setdefault("_end_date_selector", st.session_state["end_date"])

    # =======================================================
    # SECTION 1: GEOGRAPHICAL AREA
    # =======================================================

    st.markdown("#### Geographical Area")
    
    # 1. Price Area Dropdown
    st.selectbox(
        "Select Price Area", 
        AREAS,
        key="_area_selector",
        on_change=_sync_widgets_to_state
    )

    # =======================================================
    # SECTION 2: TIME PERIOD FILTERING
    # =======================================================

    st.markdown("#### Time Period Filtering")

    # 2a. Period Granularity Selector
    st.selectbox(
        "Select Period Granularity",
        TIME_PERIODS,
        key="_period_level_selector",
        on_change=_sync_widgets_to_state
    )
    
    # Dynamic Time Period Controls based on granularity selection
    period_level = st.session_state["period_level"]
    available_years = get_available_years()
    
    if period_level == "Annual":
        st.selectbox(
            "Select Year",
            available_years,
            index=available_years.index(st.session_state["selected_year"]) if st.session_state["selected_year"] in available_years else len(available_years) - 1,
            key="_year_selector",
            on_change=_sync_widgets_to_state
        )
        
    elif period_level == "Monthly":
        # Year first
        st.selectbox(
            "Select Year",
            available_years,
            index=available_years.index(st.session_state["selected_year"]) if st.session_state["selected_year"] in available_years else len(available_years) - 1,
            key="_year_selector",
            on_change=_sync_widgets_to_state
        )
        # Month second
        # month_names is already defined globally but re-calculated here for clarity/safety
        month_names = list(pd.to_datetime(range(1, 13), format='%m').strftime('%B'))
        
        # Calculate the current index based on the 1-based month number
        current_month_index = st.session_state["selected_month"] - 1
        
        st.selectbox(
            "Select Month",
            month_names,
            index=current_month_index,
            key="_month_selector",
            on_change=_sync_widgets_to_state
        )
        
    elif period_level == "Custom Date Range":
        min_date = datetime(2021, 1, 1).date()
        max_date = datetime(2024, 12, 31).date() # Max available data date is 2024-12-31
        
        st.date_input(
            "Select Start Date",
            value=st.session_state["start_date"],
            key="_start_date_selector",
            min_value=min_date,
            max_value=max_date,
            on_change=_sync_widgets_to_state
        )
        st.date_input(
            "Select End Date",
            value=st.session_state["end_date"],
            key="_end_date_selector",
            min_value=min_date,
            max_value=max_date,
            on_change=_sync_widgets_to_state
        )

    # =======================================================
    # SECTION 3: DATA FOCUS
    # =======================================================

    st.markdown("#### Data Focus")
    
    # 3a. Data Type Selector (Production vs Consumption)
    st.selectbox(
        "Select Data Type", 
        DATA_TYPES,
        key="_data_type_selector",
        on_change=_sync_widgets_to_state
    )
    
    # 3b. Production/Consumption Group Pills
    data_type = st.session_state["data_type"]
    current_groups = PRODUCTION_GROUPS if data_type == "Production" else CONSUMPTION_GROUPS
    
    # Reset the group selector if the data type changed to ensure valid options are shown
    if any(g not in current_groups for g in st.session_state["group"]):
        st.session_state["group"] = current_groups
        st.session_state["_group_selector"] = current_groups


    with st.expander(f"Filter {data_type} Sources", expanded=True):
        st.pills(
            "Select Energy Sources",
            options=current_groups,
            selection_mode="multi",
            default=st.session_state["group"], 
            key="_group_selector",
            on_change=_sync_widgets_to_state
        )
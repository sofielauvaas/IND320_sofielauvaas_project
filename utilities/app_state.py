# utilities/app_state.py
import streamlit as st

# Global configuration lists (used for widget options and validation)
AREAS  = ["NO1", "NO2", "NO3", "NO4", "NO5"]
GROUPS = ["hydro", "wind", "solar", "thermal", "other"]

def _init_globals():
    """Initializes canonical session state variables with defaults."""
    st.session_state.setdefault("pricearea", "NO1")
    st.session_state.setdefault("productiongroup", GROUPS[:])

    # Ensure canonical state is never empty, even on initialization
    if not st.session_state["productiongroup"]:
        st.session_state["productiongroup"] = GROUPS[:]

def _sync_widgets_to_state():
    """Callback function to sync widget values to the canonical session state keys."""
    # Read the updated values from the temporary widget keys
    st.session_state["pricearea"] = st.session_state["_area_selector"]
    groups_selection = st.session_state["_group_selector"]
    

    if not groups_selection or any(v not in GROUPS for v in groups_selection):
        st.session_state["productiongroup"] = GROUPS[:] # Set to full default list
    else:
        st.session_state["productiongroup"] = groups_selection
    
    st.query_params.update(
        area=st.session_state["pricearea"],
        groups=",".join(st.session_state["productiongroup"])
    )

def render_app_state_controls():
    """Renders the Price Area and Production Group selectors in the sidebar."""
    
    st.markdown("### Energy App Configuration ⚙️")
    _init_globals()

    # --- Setup Widget Keys and Defaults (Fixes the Session State Warning) ---
    if "_area_selector" not in st.session_state:
        st.session_state["_area_selector"] = st.session_state["pricearea"]
    if "_group_selector" not in st.session_state:
        st.session_state["_group_selector"] = st.session_state["productiongroup"]

    # --- 1. Price Area Dropdown (Using selectbox for difference) ---
    st.selectbox(
        "Select Price Area", 
        AREAS,
        index=AREAS.index(st.session_state["pricearea"]) if st.session_state["pricearea"] in AREAS else 0,
        key="_area_selector",
        on_change=_sync_widgets_to_state
    )
    
    # --- 2. Production Group Pills in Expander (Unique Visual) ---
    
    # Use the current canonical state as the default
    default_groups_for_pills = st.session_state["productiongroup"]

    with st.expander("Filter Production Sources", expanded=True): # New organizational element
        st.pills(
            "Select Energy Sources", # New widget label
            options=GROUPS,
            selection_mode="multi",
            # Use the canonical state as the guaranteed non-empty default
            default=default_groups_for_pills, 
            key="_group_selector",
            on_change=_sync_widgets_to_state
        )
    

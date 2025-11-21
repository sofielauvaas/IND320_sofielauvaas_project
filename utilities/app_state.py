# utilities/app_state.py
import streamlit as st

# Global configuration lists (used for widget options and validation)
AREAS  = ["NO1", "NO2", "NO3", "NO4", "NO5"]
GROUPS = ["hydro", "wind", "solar", "thermal", "other"]

def _init_globals():
    """Initializes canonical session state variables with defaults."""
    st.session_state.setdefault("price_area", "NO1")
    st.session_state.setdefault("production_group", GROUPS[:])

def _sync_widgets_to_state():
    """Callback function to sync widget values to the canonical session state keys."""
    # Read the updated values from the temporary widget keys
    st.session_state["price_area"] = st.session_state["_area_selector"]
    st.session_state["production_group"] = st.session_state["_group_selector"]
    
    # Optional: Update query parameters for bookmarking/sharing
    st.query_params.update(
        area=st.session_state["price_area"],
        groups=",".join(st.session_state["production_group"])
    )

def render_app_state_controls():
    """Renders the Price Area and Production Group selectors in the sidebar."""
    
    st.markdown("### Energy App Configuration ⚙️")
    _init_globals()

    # --- Setup Widget Keys and Defaults (Fixes the Session State Warning) ---
    if "_area_selector" not in st.session_state:
        st.session_state["_area_selector"] = st.session_state["price_area"]
    if "_group_selector" not in st.session_state:
        st.session_state["_group_selector"] = st.session_state["production_group"]

    # --- 1. Price Area Dropdown (Using selectbox for difference) ---
    st.selectbox(
        "Select Price Area", 
        AREAS,
        index=AREAS.index(st.session_state["price_area"]) if st.session_state["price_area"] in AREAS else 0,
        key="_area_selector",
        on_change=_sync_widgets_to_state
    )
    
    # --- 2. Production Group Pills in Expander (Unique Visual) ---
    
    # Use the current canonical state as the default
    default_groups = st.session_state["production_group"]
    current_selection = st.session_state.get("_group_selector", default_groups)

    with st.expander("Filter Production Sources", expanded=True): # New organizational element
        st.pills(
            "Select Energy Sources", # New widget label
            options=GROUPS,
            selection_mode="multi",
            default=current_selection, 
            key="_group_selector",
            on_change=_sync_widgets_to_state
        )
    
    # --- Post-render validation ---
    final_selection = st.session_state["_group_selector"]
    
    # Basic normalization: if nothing is selected or if the selection is invalid, reset to all
    if not final_selection or any(v not in GROUPS for v in final_selection):
        # We only reset the canonical state. Streamlit will fix the widget on the next run.
        st.session_state["production_group"] = GROUPS[:]
    else:
        st.session_state["production_group"] = final_selection
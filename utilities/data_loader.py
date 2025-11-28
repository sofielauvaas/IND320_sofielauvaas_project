import streamlit as st
from utilities import functions 
import pandas as pd

# Define the standardized column name that the functions.py loader should produce
GROUP_COLUMN = 'group' 

@st.cache_data(show_spinner="Loading and standardizing production data...")
def load_and_clean_production_data():
    """
    Loads Elhub production data from MongoDB using the cached function 
    and relies on functions.py to produce the standardized 'group' column.
    """
    # NOTE: The data type requested is 'Production'
    data_type_requested = "Production" 
    
    try:
        # Load data using the alias function that dispatches to the correct loader
        df = functions.load_data_from_mongo(data_type_requested) 
        
        if df.empty:
            st.error("Data loading failed: The returned DataFrame is empty.")
            return pd.DataFrame()
            
        # --- DATA CLEANING AND STANDARDIZATION ---
        # The main processing is done in functions.py. We only check integrity here.
        df.columns = df.columns.str.strip().str.lower()
        
        # 1. Final check for the standardized GROUP_COLUMN
        if GROUP_COLUMN not in df.columns:
             # If this fails, the issue is still in the ETL process where the group column was dropped.
             st.error(f"FATAL ERROR in data pipeline: Missing required standardized column '{GROUP_COLUMN}'.")
             st.error(f"Please check utilities/functions.py to ensure the original grouping column is mapped to '{GROUP_COLUMN}'.")
             return pd.DataFrame()
            
        # 2. Ensure starttime is datetime (assuming it's still needed here)
        if 'starttime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['starttime']):
            df['starttime'] = pd.to_datetime(df['starttime'], utc=True, errors='coerce') 

        df.dropna(subset=['starttime'], inplace=True)
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load data from MongoDB via functions.py. Error: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Loading and standardizing consumption data...")
def load_and_clean_consumption_data():
    """
    Loads Elhub consumption data from MongoDB, ensuring the output column is 'group'.
    """
    # NOTE: The data type requested is 'Consumption'
    data_type_requested = "Consumption"
    
    try:
        # Load data using the alias function that dispatches to the correct loader
        df = functions.load_data_from_mongo(data_type_requested) 
        
        if df.empty:
            st.error("Data loading failed: The returned DataFrame is empty.")
            return pd.DataFrame()
            
        # --- Data Cleaning and Standardization ---
        df.columns = df.columns.str.strip().str.lower()
        
        # 1. Final check for the standardized GROUP_COLUMN
        if GROUP_COLUMN not in df.columns:
             # If this fails, the data source is aggregated.
             st.error(f"FATAL ERROR: The required grouping column '{GROUP_COLUMN}' is missing from the consumption data. This usually means the data is aggregated.")
             return pd.DataFrame()

        # 2. Ensure starttime is datetime
        if 'starttime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['starttime']):
            df['starttime'] = pd.to_datetime(df['starttime'], utc=True, errors='coerce') 

        df.dropna(subset=['starttime'], inplace=True)
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load consumption data from MongoDB via functions.py. Error: {e}")
        return pd.DataFrame()
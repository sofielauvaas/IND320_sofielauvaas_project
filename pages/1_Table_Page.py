import streamlit as st
import pandas as pd

st.title("Data Table")

# Load CSV
@st.cache_data
def load_data():
    return pd.read_csv("open-meteo-subset.csv")




df = load_data()

st.subheader("Dataset preview")
st.dataframe(df.head(), use_container_width=True)

# --- Row-wise table with sparklines ---
st.subheader("First month summary (row-wise sparklines)")

# Take first 30 rows (≈ one month)
first_month = df.head(30)

# Get only numeric columns
numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

# Build new DataFrame: one row per numeric variable
chart_data = pd.DataFrame({
    "Variable": numeric_cols,
    "Values": [first_month[col].tolist() for col in numeric_cols]
})

chart_data = pd.DataFrame({
    "Variable": numeric_cols,
    "Values": [first_month[col].tolist() for col in numeric_cols]
})

st.dataframe(
    chart_data,
    column_config={
        "Values": st.column_config.LineChartColumn(
            "First Month", 
            y_min=None,  # auto-scale per row
            y_max=None
        ),
        "Variable": st.column_config.TextColumn("Column Name"),
    },
    hide_index=True,
    use_container_width=True,
)
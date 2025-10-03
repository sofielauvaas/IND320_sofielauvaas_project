import streamlit as st
import pandas as pd

st.title("Data Table")

# Load CSV
@st.cache_data
def load_data():
    df = pd.read_csv("open-meteo-subset.csv")
    df["time"] = pd.to_datetime(df["time"])
    return df

df = load_data()

st.subheader("Dataset preview")
st.dataframe(df.head(), use_container_width=True)

# First month summary header
st.subheader("First month summary")

# Filter first month
first_month = df[df["time"].dt.month == df["time"].dt.month.min()]

# Prepare a table: one row per column
table_data = []
for col in first_month.columns:
    if col == "time":
        continue
    row = {
        "Column": col,
        "First Month": first_month[col].tolist(),
        "Min": first_month[col].min(),
        "Max": first_month[col].max(),
        "Mean": first_month[col].mean(),
        "Median": first_month[col].median(),
        "Std": first_month[col].std()
    }
    table_data.append(row)

df_table = pd.DataFrame(table_data)

# Configure the LineChartColumn
column_config = {
    "First Month": st.column_config.LineChartColumn(
        label="First Month",
        width="medium"
    )
}

# Display the table without index
st.dataframe(
    df_table,
    column_config=column_config,
    use_container_width=True,
    hide_index=True
)

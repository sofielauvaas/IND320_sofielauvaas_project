import streamlit as st
import pandas as pd

st.title("Plot Page")

# Load CSV with caching
@st.cache_data
def load_data():
    df = pd.read_csv("data/open-meteo-subset.csv")
    df["time"] = pd.to_datetime(df["time"])
    return df

df = load_data()

# Column selection excluding 'time'
columns = ["All columns"] + [c for c in df.columns if c != "time"]
selected_col = st.selectbox("Select a column", columns)

# Month selection slider
months = sorted(df["time"].dt.month.unique())
selected_months = st.select_slider(
    "Select month range",
    options=months,
    value=(months[0], months[0]),
    format_func=lambda x: pd.to_datetime(f"2023-{x}-01").strftime("%B")
)

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

# Rename months for title
month_names = (
    pd.to_datetime(f"2023-{selected_months[0]}-01").strftime("%B"),
    pd.to_datetime(f"2023-{selected_months[1]}-01").strftime("%B")
)

# Plot the selected column(s)
if selected_col == "All columns":
    st.subheader(f"All columns from {month_names[0]} to {month_names[1]}")
    st.line_chart(df_to_plot)
else:
    st.subheader(f"{selected_col} from {month_names[0]} to {month_names[1]}")
    st.line_chart(df_to_plot[selected_col])

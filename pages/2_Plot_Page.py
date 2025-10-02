import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

st.title("Plot Page")

# Load CSV with caching
@st.cache_data
def load_data():
    df = pd.read_csv("open-meteo-subset.csv")
    # Try parsing first column as datetime (if it looks like time data)
    try:
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors="coerce")
        df = df.set_index(df.columns[0])
    except Exception:
        pass
    return df

df = load_data()

# Column selection
columns = ["All columns"] + list(df.columns)
selected_col = st.selectbox("Select a column", columns)

# If datetime index, allow month range selection
if isinstance(df.index, pd.DatetimeIndex):
    # Extract unique months (as "YYYY-MM")
    labels = df.index.to_period("M").astype(str).unique().tolist()

    if labels:  # only if not empty
        start, end = st.select_slider(
            "Select month range",
            options=labels,
            value=(labels[0], labels[0])  # default to the first month only
        )
        # Filter df between selected months
        mask = (df.index.to_period("M") >= start) & (df.index.to_period("M") <= end)
        df = df[mask]



# Plotting
st.subheader("Plot of selected data")

# Wanted the plot to look nicer, so using a colormap for multiple lines
plt.figure(figsize=(10, 4))
cmap = cm.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, len(df.columns)))


# Plot all numeric columns or the selected one
if selected_col == "All columns":
    for i, col in enumerate(df.columns):
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.plot(df.index, df[col], label=col, color=colors[i])
    plt.legend()
else:
    if pd.api.types.is_numeric_dtype(df[selected_col]):
        plt.plot(df.index, df[selected_col], label=selected_col, color="tab:red")
        plt.legend()

# Plot formatting
plt.xlabel("Time" if isinstance(df.index, pd.DatetimeIndex) else "Index")
plt.ylabel("Value")
plt.title("Data visualization")
plt.grid(True, alpha=0.3)
st.pyplot(plt)

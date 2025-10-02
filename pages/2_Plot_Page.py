import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 Plot Page")

# --- Load CSV with caching ---
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

# --- Sidebar controls ---
columns = ["All columns"] + list(df.columns)
selected_col = st.selectbox("Select a column", columns)

# For slider: create list of labels (use index if datetime, else row numbers)
if isinstance(df.index, pd.DatetimeIndex):
    labels = df.index.strftime("%Y-%m-%d").tolist()
else:
    labels = df.index.astype(str).tolist()

if labels:  # only if not empty
    start, end = st.select_slider(
        "Select range",
        options=labels,
        value=(labels[0], labels[min(len(labels)-1, 30)])  # default first ~month
    )
    # Filter the dataframe
    mask = (labels.index(start), labels.index(end))
    start_idx, end_idx = labels.index(start), labels.index(end)
    df = df.iloc[start_idx:end_idx+1]

# --- Plot ---
st.subheader("Plot of selected data")

plt.figure(figsize=(10, 4))
if selected_col == "All columns":
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.plot(df.index, df[col], label=col)
    plt.legend()
else:
    plt.plot(df.index, df[selected_col], label=selected_col)
    plt.legend()

plt.xlabel("Time" if isinstance(df.index, pd.DatetimeIndex) else "Index")
plt.ylabel("Value")
plt.title("Data visualization")
plt.grid(True, alpha=0.3)
st.pyplot(plt)

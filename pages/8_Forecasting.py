import streamlit as st
import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
import plotly.graph_objects as go
from utilities import functions

st.title("Energy Flow Forecasting")

# Helper: choose a time series from the dataset structure
def select_series(df, kind):
    """
    Expected incoming fields:
      - pricearea
      - group
      - starttime (index)
      - quantitykwh
    """

    if df.empty:
        return pd.Series(dtype=float), {}

    # Choose group
    groups = sorted(df["group"].dropna().unique())
    group = st.selectbox("Group", groups)

    # Choose price area
    priceareas = sorted(df["pricearea"].dropna().unique())
    pa = st.selectbox("Price Area", np.append(["ALL"], priceareas))

    # Basic filtering
    if pa == "ALL":
        filtered = df[df["group"] == group].copy()
    else:
        filtered = df[(df["group"] == group) & (df["pricearea"] == pa)].copy()

    if filtered.empty:
        return pd.Series(dtype=float), {"pricearea": pa, "group": group}

    # Aggregate values over timestamps
    series = (
        filtered.groupby(filtered.index)
        .agg({"quantitykwh": "sum"})
        .sort_index()["quantitykwh"]
    )

    # Convert to daily representation
    series = series.resample("D").sum().fillna(0)

    # Assign frequency if possible
    try:
        inferred = series.index.inferred_freq
        if inferred:
            series = series.asfreq(inferred)
    except Exception:
        pass

    return series, {"pricearea": pa, "group": group}

# SARIMAX model wrapper for caching
@st.cache_resource(show_spinner=True)
def fit_sarimax(series, order, seasonal_order, exog=None):
    model = sm.tsa.SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        exog=exog,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)

# Load data files
prod_df = functions.load_elhub_data("Production")
cons_df = functions.load_elhub_data("Consumption")

# Ensure timestamp index
for df in [prod_df, cons_df]:
    if not df.empty:
        df.set_index("starttime", inplace=True)
        df.sort_index(inplace=True)

# User selection sidebar
kind = st.radio("Dataset", ["production", "consumption"])

selected_df = prod_df if kind == "production" else cons_df
series, meta = select_series(selected_df, kind)

if series.empty:
    st.warning("No matching data found.")
    st.stop()

# Training data window selection
st.header("Select Training Window")

min_date = series.index.min().date()
max_date = series.index.max().date()

start_date, end_date = st.date_input(
    "Training period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if start_date > end_date:
    st.error("Starting date cannot be after the ending date.")
    st.stop()

# Convert date → timestamp range
start_dt = pd.Timestamp(start_date, tz="UTC")
end_dt = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
series = series.loc[start_dt:end_dt]

# Display the selected historical time series
st.subheader(f"Time Series Overview:")
fig_series = go.Figure()
fig_series.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines"))
fig_series.update_layout(
    title="Historical Data",
    xaxis_title="Date",
    yaxis_title="kWh"
)
st.plotly_chart(fig_series, use_container_width=True)

# SARIMAX parameter selection
st.subheader("Model Settings")

col_p, col_d, col_q, col_trend = st.columns(4)
p = col_p.number_input("AR (p)", 0, 5, 1, key='p_order')
d = col_d.number_input("Differencing (d)", 0, 2, 1, key='d_order')
q = col_q.number_input("MA (q)", 0, 5, 1, key='q_order')
trend = col_trend.selectbox("Trend", options=['n', 'c', 't', 'ct'], index=1, key='trend_select')

col_P, col_D, col_Q, col_m = st.columns(4)
P = col_P.number_input("Seasonal AR (P)", 0, 2, 0, key='P_order')
D = col_D.number_input("Seasonal Diff (D)", 0, 1, 0, key='D_order')
Q = col_Q.number_input("Seasonal MA (Q)", 0, 2, 0, key='Q_order')
m = col_m.number_input("Seasonal Period (m)", 1, 365, 7, key='m_order')

col_exog, col_steps = st.columns([3, 1])

with col_exog:
    weather_vars = [
        'temperature_2m',
        'wind_speed_10m',
        'precipitation',
        'wind_gusts_10m',
        'wind_direction_10m',
    ]
    exog_vars = st.multiselect("Optional Weather Inputs", options=weather_vars, default=[], key='exog_vars')

with col_steps:
    steps = st.number_input("Forecast Length (days)", 1, 365, 30, key='steps_order')

forecast_button = st.button("Generate Forecast", type="primary")

# Forecast execution
if forecast_button:
    with st.spinner("Running model..."):
        model_fit = fit_sarimax(
            series,
            order=(p, d, q),
            seasonal_order=(P, D, Q, m)
        )
        forecast_res = model_fit.get_forecast(steps=steps)
        forecast_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()

    # Build forecast plot
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=series.index, y=series.values, name="Observed"))
    fig_forecast.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean.values, name="Forecast", line=dict(color="red")))

    # Confidence band
    fig_forecast.add_trace(
        go.Scatter(
            x=conf_int.index.tolist() + conf_int.index[::-1].tolist(),
            y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(255,182,193,0.3)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval",
        )
    )

    fig_forecast.update_layout(
        title="Forecast Results",
        xaxis_title="Date",
        yaxis_title="kWh",
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

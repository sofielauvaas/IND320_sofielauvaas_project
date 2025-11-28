import streamlit as st
import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
import plotly.graph_objects as go
from utilities import functions 
from utilities.app_state import render_app_state_controls # For rendering sidebar controls

st.set_page_config(layout="wide")
st.title("Energy Flow Prediction (SARIMAX)")

# Define the standardized group column name (must match functions.py)
GROUP_COL = 'group' 


# CACHED DATA LOADERS
@st.cache_data(ttl=3600, show_spinner="Loading all energy data from MongoDB...")
def load_all_energy_data():
    """Loads all necessary dataframes (Production and Consumption) using existing dispatchers."""
    prod_df = functions.load_elhub_data("Production")
    cons_df = functions.load_elhub_data("Consumption")
    return prod_df, cons_df


# Function: Select a time series using your data structure
def select_series(df, kind):
    """
    Filters the time series based on local UI selections and aggregates to daily kWh.
    """

    if df.empty:
        return pd.Series(dtype=float), {"group": "N/A", "pricearea": "N/A"}

    # Select group
    groups = sorted(df[GROUP_COL].dropna().unique())
    group = st.selectbox("Select Energy Group", groups, key='local_group_select')

    # Select pricearea
    priceareas = sorted(df["pricearea"].dropna().unique())
    pa = st.selectbox("Select Price Area", np.append(["ALL"], priceareas), key='local_pa_select')

    # Filter logic
    if pa == "ALL":
        filtered = df[df[GROUP_COL] == group].copy()
    else:
        filtered = df[(df[GROUP_COL] == group) & (df["pricearea"] == pa)].copy()

    if filtered.empty:
        return pd.Series(dtype=float), {"pricearea": pa, "group": group}

    # Aggregate by time
    series = (
        filtered.groupby(filtered.index)
        .agg({"quantitykwh": "sum"})
        .sort_index()["quantitykwh"]
    )

    # Daily resampling
    series = series.resample("D").sum().fillna(0)

    # Restore inferred frequency
    try:
        inferred = series.index.inferred_freq
        if inferred:
            series = series.asfreq(inferred)
    except Exception:
        pass

    # Ensure final series is Naive (required by statsmodels)
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)

    return series, {"pricearea": pa, "group": group}


# Cached SARIMAX fit 
@st.cache_resource(show_spinner=True)
def fit_sarimax(series, order, seasonal_order, exog=None):
    """Fits SARIMAX model using the selected parameters."""
    model = sm.tsa.SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        exog=exog,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)


# PAGE EXECUTION
# 1. Load raw energy data
prod_df_raw, cons_df_raw = load_all_energy_data()

# Rename group columns for consistency across production/consumption frames
prod_df_raw.rename(columns={'productiongroup': GROUP_COL}, inplace=True, errors='ignore')
cons_df_raw.rename(columns={'consumptiongroup': GROUP_COL}, inplace=True, errors='ignore')

# Ensure index is datetime and set
for df in [prod_df_raw, cons_df_raw]:
    if not df.empty and "starttime" in df.columns:
        df.set_index("starttime", inplace=True)
        df.sort_index(inplace=True)

# 2. Render sidebar controls (CONTEXT ONLY)
with st.sidebar:
    render_app_state_controls()

# 3. Local Data Type Selection
kind = st.radio("Select Dataset Type", ["production", "consumption"], key='type_radio_main')

df_selected = prod_df_raw if kind == "production" else cons_df_raw

# --- Time Series Selection UI ---
st.header("1. Time Series Selection")

col_select_a, col_select_b = st.columns(2)

with col_select_a:
    series, meta = select_series(df_selected, kind)


if series.empty:
    st.warning("No data available for selected options.")
    st.stop()


# Training period selector
with col_select_b:
    st.subheader("Training/Forecast Split")
    min_date = series.index.min().date()
    max_date = series.index.max().date()

    start_date, end_date = st.date_input(
        "Select full data period (Training + Forecast)",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key='full_data_period'
    )

    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

# Convert dates to Naive Timestamps (matching series index)
start_dt = pd.Timestamp(start_date)
end_dt = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59)

series_filtered = series.loc[start_dt:end_dt]

# Plot observed series
st.subheader(f"Observed Series: {meta}")
fig_series = go.Figure()
fig_series.add_trace(go.Scatter(x=series_filtered.index, y=series_filtered.values, mode="lines"))
fig_series.update_layout(
    title="Observed Daily Energy Flow (kWh)",
    xaxis_title="Date",
    yaxis_title="Quantity kWh",
    height=300
)
st.plotly_chart(fig_series, use_container_width=True)


# 2. SARIMAX PARAMETER INPUTS (Main Page)
st.header("2. SARIMAX Model Parameters")

# --- PARAMETER ROW 1: AR, MA, Diff ---
col_p, col_d, col_q, col_trend = st.columns(4)
p = col_p.number_input("p (AR Order)", 0, 5, 1, key='p_order')
d = col_d.number_input("d (Diff Order)", 0, 2, 1, key='d_order')
q = col_q.number_input("q (MA Order)", 0, 5, 1, key='q_order')
trend = col_trend.selectbox("Trend Component", options=['n', 'c', 't', 'ct'], index=1, key='trend_select')

# --- PARAMETER ROW 2: Seasonal Parameters ---
col_P, col_D, col_Q, col_m = st.columns(4)
P = col_P.number_input("P (Seasonal AR)", 0, 2, 0, key='P_order')
D = col_D.number_input("D (Seasonal Diff)", 0, 1, 0, key='D_order')
Q = col_Q.number_input("Q (Seasonal MA)", 0, 2, 0, key='Q_order')
m = col_m.number_input("m (Seasonal Period)", 1, 365, 7, key='m_order')


# --- Exogenous & Horizon ---
col_exog, col_steps = st.columns([3, 1])

with col_exog:
    # Exogenous Variables (Weather Properties)
    weather_vars = ['temperature_2m', 'precipitation', 'wind_speed_10m', 'wind_gusts_10m', 'wind_direction_10m']
    exog_vars = st.multiselect("Exogenous Variables (Weather)", options=weather_vars, default=[], key='exog_vars')

with col_steps:
    steps = st.number_input("Forecast Horizon (steps)", 1, 365, 30, key='steps_order')

# --- Button and Final Check ---
if len(series_filtered) < (m * 2 + 1 + steps):
    st.error(f"Insufficient data for SARIMAX (Need at least {m * 2 + 1 + steps} periods). Found: {len(series_filtered)}")
    st.stop()

forecast_button = st.button("Run Forecast", type="primary")


# Run forecast
if forecast_button:
    # NOTE: The implementation below needs to perform the exog data prep and slicing here
    
    # 1. Prepare Exogenous Data for SARIMAX (Requires external function call)
    with st.spinner("Preparing exogenous data..."):
        # We need a function to fetch, aggregate, and slice weather data based on the selection
        
        # --- Simulating Exog Data for now (TO BE IMPLEMENTED LATER) ---
        exog_train = None
        exog_forecast = None
        
        # 2. Split Series into Training and Forecast Period
        y_train = series_filtered.iloc[:-steps]
        
        # 3. Run Fit
        model_fit = fit_sarimax(
            y_train,
            order=(p, d, q),
            seasonal_order=(P, D, Q, m),
            exog=exog_train # Keeping exog=None for stability
        )
        
        # 4. Generate Forecast
        forecast_res = model_fit.get_forecast(steps=steps, exog=exog_forecast)
        forecast_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()

        # Store results
        st.session_state['forecast_results'] = {
            'mean': forecast_mean,
            'ci': conf_int,
            'observed': series_filtered,
            'summary': model_fit.summary().as_text(),
            'train_end_date': y_train.index[-1]
        }
        st.success("Forecasting Complete!")


# VISUALIZATION
if 'forecast_results' in st.session_state:
    results = st.session_state['forecast_results']
    
    st.header("3. Forecast Visualization")
    
    fig_forecast = go.Figure()
    
    # 1. Observed
    fig_forecast.add_trace(go.Scatter(x=results['observed'].index, y=results['observed'].values, mode="lines", name="Observed"))
    
    # 2. Forecast Mean
    fig_forecast.add_trace(go.Scatter(x=results['mean'].index, y=results['mean'].values, name="Forecast Mean", line=dict(color="red", dash='dot')))

    # 3. Confidence interval
    conf_int = results['ci']
    fig_forecast.add_trace(
        go.Scatter(
            x=conf_int.index.tolist() + conf_int.index[::-1].tolist(),
            y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(255,182,193,0.3)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence Interval"
        )
    )
    
    # 4. Highlight Forecast Start 
    forecast_start_date_str = results['train_end_date'].strftime('%Y-%m-%d')
    fig_forecast.add_vline(
        x=forecast_start_date_str, 
        line_width=1,
        line_dash="dash",
        line_color="black",
        annotation_text="Forecast Start",
        annotation_position="top right"
    )


    fig_forecast.update_layout(
        title=f"SARIMAX Forecast for {meta['group'].capitalize()} in {meta['pricearea']}",
        xaxis_title="Date",
        yaxis_title="Quantity kWh",
        height=600
    )

    st.plotly_chart(fig_forecast, use_container_width=True)
    
    with st.expander("Model Summary", expanded=False):
        st.text(results['summary'])
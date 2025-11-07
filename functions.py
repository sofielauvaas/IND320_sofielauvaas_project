import pandas as pd
import numpy as np
import requests 
import requests_cache
from retry_requests import retry
import openmeteo_requests
from scipy.fft import dct, idct
from scipy.signal import spectrogram
from statsmodels.tsa.seasonal import STL
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pymongo import MongoClient
import warnings 

# --- GLOBAL CONFIGURATION AND INITIALIZATION ---
# Area coordinates for Norway price areas (latitude, longitude)
AREA_COORDINATES = {
    "NO1": (59.9122, 10.7313), "NO2": (58.1599, 8.0182), "NO3": (63.4305, 10.3951),
    "NO4": (69.6498, 18.9841), "NO5": (60.3913, 5.3221) 
}
TARGET_YEAR = 2021
BASE_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


# Global Robust Client Setup (No Streamlit caching function, per request)
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo_client = openmeteo_requests.Client(session=retry_session) 


# --- MONGODB CONNECTION AND DATA LOADING ---

@st.cache_data(ttl=3600, show_spinner="Loading Elhub production data...")
def load_data_from_mongo():
    """Establishes connection and loads all Elhub 2021 production data."""
    uri = st.secrets["mongodb"]["uri"]
    client = MongoClient(uri)

    db_name = st.secrets["mongodb"]["database"]
    col_name = st.secrets["mongodb"]["collection"]
    db = client[db_name]
    collection = db[col_name] 
    
    items = list(collection.find({}, {'_id': 0})) 
    df = pd.DataFrame(items)
    client.close()

    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={'start-time': 'starttime', 'quantity-kwh': 'quantitykwh'}, inplace=True) 

    if 'starttime' in df.columns:
        df['starttime'] = pd.to_datetime(df['starttime'], errors='coerce')
        df['month_name'] = df['starttime'].dt.strftime('%B')
        
    if 'quantitykwh' in df.columns:
        df['quantitykwh'] = pd.to_numeric(df['quantitykwh'], errors='coerce')
        
    return df


# --- GEO AND WEATHER DATA SOURCING ---

@st.cache_data(ttl=3600, show_spinner="Downloading weather data...")
def download_weather_data(price_area):
    """
    Downloads hourly historical ERA5 Reanalysis weather data for the given price area.
    (Simplified function signature to take only one argument, unlike the Notebook function)
    """
    area = price_area.upper()
    if area not in AREA_COORDINATES:
        raise ValueError(f"Price Area {area} not found in coordinates dictionary.")
        
    latitude, longitude = AREA_COORDINATES[area] # Lookup coordinates
    year = TARGET_YEAR # Use the global year
    
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    hourly_variables = ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]
    
    params = {
        "latitude": latitude, "longitude": longitude,
        "start_date": start_date, "end_date": end_date,
        "hourly": hourly_variables, 
        "models": "era5",
    }
    
    # Use the global robust client
    responses = openmeteo_client.weather_api(BASE_WEATHER_URL, params=params)
    response = responses[0]

    hourly = response.Hourly()
    
    hourly_data = {
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }

    for i, var in enumerate(hourly_variables):
        hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
    
    df = pd.DataFrame(hourly_data)
    df = df.set_index('time')

    return df


# --- ANOMALY DETECTION FUNCTIONS ---

def dct_highpass_filter(signal, keep_index):
    """High-pass filter using DCT to remove the slow trend, leaving the Seasonally Adjusted Variation."""
    x = np.asarray(signal, dtype=float)
    X = dct(x, norm="ortho")
    n = len(X)
    k_low = max(1, min(n, int(keep_index))) 
    
    X_lp = np.zeros_like(X)
    X_lp[:k_low] = X[:k_low]
    trend = idct(X_lp, norm="ortho")
    satv = x - trend
    return satv, trend

def temperature_spc_from_satv(time, temperature, keep_low_index=100, k=3.0):
    """Detects temperature outliers using robust Statistical Process Control (SPC)."""
    t = np.asarray(time)
    x = np.asarray(temperature, dtype=float)
    n = len(x)
    
    satv, trend = dct_highpass_filter(x, keep_low_index)
    
    center = np.median(satv)
    mad = np.median(np.abs(satv - center))
    spread = (1.4826 * mad)
    
    upper_satv = center + k * spread
    lower_satv = center - k * spread
    
    upper_curve = trend + upper_satv
    lower_curve = trend + lower_satv
    is_outlier = (satv > upper_satv) | (satv < lower_satv)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[~is_outlier], y=x[~is_outlier], mode="lines", name="Temperature (Inliers)", line=dict(color='#035397', width=1.0)))
    fig.add_trace(go.Scatter(x=t[is_outlier], y=x[is_outlier], mode="markers", name="Outliers (SPC)", marker=dict(color='#128264', size=6, opacity=0.9)))
    fig.add_trace(go.Scatter(x=t, y=upper_curve, mode="lines", name="UCL", line=dict(color='#f9c80e', dash="dash", width=1.5)))
    fig.add_trace(go.Scatter(x=t, y=lower_curve, mode="lines", name="LCL", line=dict(color='#f9c80e', dash="dash", width=1.5)))
    
    fig.update_layout(template="plotly_white", title=f"Temperature Outliers Detected via Robust SPC", xaxis_title="Date", yaxis_title="Temperature (°C)", title_x=0.5)
    
    summary = {"n_outliers": int(is_outlier.sum()), "n_total": int(n), "percent_outliers": round(100 * is_outlier.mean(), 2), "robust_std": spread}
    return fig, summary

def precipitation_lof_plot(time, precipitation, contamination=0.01, n_neighbors=30, variable_name="Precipitation"):
    """
    Detects and plots precipitation anomalies using Local Outlier Factor (LOF). 
    Returns the LOF score for deeper analysis.
    """
    
    X = np.array(precipitation.fillna(0)).reshape(-1, 1)

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = lof.fit_predict(X) 
    scores = -lof.negative_outlier_factor_
    is_outlier = labels == -1
    n_outliers = int(is_outlier.sum())
    n_total = len(X)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(time)[~is_outlier], y=X[~is_outlier, 0], mode="lines", name="Precipitation (Inliers)", line=dict(color='#035397', width=1.0)))
    fig.add_trace(go.Scatter(x=np.array(time)[is_outlier], y=X[is_outlier, 0], mode="markers", name="Anomalies (LOF)", marker=dict(color="#128264", size=6, opacity=0.8)))

    fig.update_layout(template="plotly_white", title=f"{variable_name} Anomalies via LOF (Contamination: {contamination*100:.1f}%)", xaxis_title="Date", yaxis_title=f"{variable_name} (Value)", title_x=0.5)

    summary = {
        "n_total": n_total, 
        "n_outliers": n_outliers, 
        "percent_outliers": round(100 * n_outliers / n_total, 2),
        "lof_scores": scores
    }
    return fig, summary


# --- TIME SERIES ANALYSIS FUNCTIONS ---

def stl_decomposition_elhub(df, price_area="NO5", production_group="hydro", period=168, seasonal=9, trend=241, robust=False):
    """Performs Seasonal-Trend Decomposition using LOESS (STL) on the selected production group's hourly data."""
    line_color = '#416287' 
    subset = df[(df["pricearea"].str.upper() == price_area.upper()) & (df["productiongroup"].str.lower() == production_group.lower())].copy()
    
    if subset.empty:
        raise ValueError(f"No data found for area '{price_area}' and group '{production_group}' for STL analysis.")

    subset = subset.set_index(pd.to_datetime(subset["starttime"]))
    ts = subset.groupby(level=0)['quantitykwh'].sum().asfreq('h').ffill().sort_index()
    
    result = STL(ts, period=period, seasonal=seasonal, trend=trend, robust=robust).fit()
    components_map = {"Observed": ts, "Trend": result.trend, "Seasonal": result.seasonal, "Remainder": result.resid}
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Observed", "Trend", "Seasonal", "Remainder"), vertical_spacing=0.04)
    for i, (name, component_series) in enumerate(components_map.items()):
        fig.add_trace(go.Scatter(x=component_series.index, y=component_series.values, mode="lines", line=dict(color=line_color, width=1), name=name), row=i + 1, col=1)
    
    fig.update_layout(height=950, template="plotly_white", title=f"STL Decomposition — {price_area.upper()} {production_group.capitalize()}", title_x=0.5, showlegend=False, margin=dict(t=80, b=50, l=50, r=20))
    fig.update_xaxes(title_text="Date", row=4, col=1)
    return fig


def create_spectrogram(
    df_prod, 
    price_area='NO5', 
    production_group='hydro', 
    window_length=24 * 7,
    overlap=24 * 4         
):
    """
    Creates a power spectrogram (STFT) where the Y-axis is converted to 
    cycles/day, including a bi-daily marker.
    """
    
    subset = df_prod[
        (df_prod['pricearea'].str.upper() == price_area.upper()) & 
        (df_prod['productiongroup'].str.lower() == production_group.lower())
    ].sort_values("starttime")

    subset = subset.set_index(pd.to_datetime(subset["starttime"]))
    ts = subset.groupby(level=0)['quantitykwh'].sum().asfreq('h').ffill().values
    
    if len(ts) < window_length:
         raise ValueError("Time series too short for Spectrogram with current window size.")

    fs = 1.0
    f, t_hours, Sxx = spectrogram(
        ts, 
        fs=fs, 
        nperseg=window_length, 
        noverlap=overlap, 
        detrend='constant'
    )

    power_db = 10 * np.log10(Sxx + 1e-12)
    
    # Convert axes for intuitive analysis
    t_days = t_hours / 24 
    f_cycles_per_day = f * 24 
    
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.pcolormesh(t_days, f_cycles_per_day, power_db, shading='gouraud', cmap='viridis')
    
    ax.set_title(f"Spectrogram (STFT) — {price_area.upper()} {production_group.capitalize()}", fontsize=16)
    ax.set_xlabel("Time [days]", fontsize=12)
    ax.set_ylabel("Frequency [cycles/day]", fontsize=12)
    
    # Highlight key cycles on the new axis scale, including Bi-Daily (2.0 cycles/day)
    ax.set_ylim(0, 5) 
    ax.set_yticks([1.0, 2.0, 1.0/7.0]) 
    ax.set_yticklabels([f'Daily Cycle (1.0)', f'Bi-Daily Cycle (2.0)', f'Weekly Cycle ({1/7:.2f})'])
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=1)
    ax.axhline(y=2.0, color='y', linestyle='--', linewidth=1) 
    ax.axhline(y=1.0/7.0, color='w', linestyle='--', linewidth=1)
    
    fig.colorbar(im, ax=ax, label="Power [dB]")
    plt.tight_layout()
    return fig
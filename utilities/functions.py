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
# Removed TARGET_YEAR = 2021
BASE_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


# Global Robust Client Setup
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo_client = openmeteo_requests.Client(session=retry_session) 


# --- MONGODB CONNECTION AND DATA LOADING (UPDATED FOR ALL YEARS) ---

@st.cache_data(ttl=3600, show_spinner="Loading Elhub production data...")
def load_data_from_mongo():
    """
    Establishes connection and loads ALL available production data from MongoDB (2021-2024).
    """
    uri = st.secrets["mongodb"]["uri"]
    client = MongoClient(uri)

    db_name = st.secrets["mongodb"]["database"]
    col_name = st.secrets["mongodb"]["collection"] # Assumes this collection now holds all 2021-2024 production data
    db = client[db_name]
    collection = db[col_name] 
    
    # Load all items without a year filter
    items = list(collection.find({}, {'_id': 0})) 
    df = pd.DataFrame(items)
    client.close()

    # 1. Convert all existing columns to lowercase
    df.columns = [c.lower().strip() for c in df.columns]
    
    # 2. Aggressive renaming for consistency
    df.rename(columns={
        'pricearea': 'pricearea',
        'productiongroup': 'productiongroup', 
        'starttime': 'starttime',
        'endtime': 'endtime',
        'quantitykwh': 'quantitykwh',
        'lastupdatedtime': 'lastupdatedtime',
        'start-time': 'starttime', 
        'quantity-kwh': 'quantitykwh'
    }, inplace=True, errors='ignore') 

    # --- Type Conversion and Feature Engineering ---
    if 'starttime' in df.columns:
        df['starttime'] = pd.to_datetime(df['starttime'], errors='coerce')
        df['month_name'] = df['starttime'].dt.strftime('%B')
        
    if 'quantitykwh' in df.columns:
        df['quantitykwh'] = pd.to_numeric(df['quantitykwh'], errors='coerce')
        
    return df

# --- GEO AND WEATHER DATA SOURCING (UPDATED FOR 2021-2024) ---

@st.cache_data(ttl=3600, show_spinner="Downloading weather data...")
def download_weather_data(pricearea):
    """
    Downloads hourly historical ERA5 Reanalysis weather data for the given price area,
    covering the full 2021-2024 range.
    """
    area = pricearea.upper()
    if area not in AREA_COORDINATES:
        raise ValueError(f"Price Area {area} not found in coordinates dictionary.")
        
    latitude, longitude = AREA_COORDINATES[area] # Lookup coordinates
    
    # UPDATED: Use fixed start and end dates to cover the entire data range
    start_date = "2021-01-01"
    end_date = "2024-12-31"

    hourly_variables = ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]
    
    params = {
        "latitude": latitude, "longitude": longitude,
        "start_date": start_date, "end_date": end_date,
        "hourly": hourly_variables, 
        "models": "era5",
    }
    
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


# --- ANOMALY DETECTION FUNCTIONS (No changes needed, they operate on data provided) ---

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
    
    # Handle NaNs before DCT
    series = pd.Series(x)
    series_clean = series.interpolate(method='linear')
    x_clean = series_clean.fillna(method='bfill').fillna(method='ffill').to_numpy()
    
    if np.any(np.isnan(x_clean)):
        warnings.warn("Data still contains unrecoverable NaNs.")
        return go.Figure().update_layout(title="Error: Data contains unrecoverable NaNs."), {"n_outliers": 0, "n_total": n, "percent_outliers": 0, "robust_std": 0}


    satv, trend = dct_highpass_filter(x_clean, keep_low_index)
    
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

def precipitation_lof_plot(time, data_series, outlier_frac=0.01, n_neighbors=30, variable_name="Precipitation"):
    """
    Detects and plots anomalies using Local Outlier Factor (LOF). 
    """
    
    precip = data_series.fillna(0)
    nonzero_mask = precip > 0
    
    X_nonzero = np.log1p(precip[nonzero_mask].values.reshape(-1, 1))
    
    n_nonzero = len(X_nonzero)
    
    full_anomaly_mask = np.zeros(len(precip), dtype=bool)
    n_outliers = 0
    scores = np.zeros(n_nonzero)

    if n_nonzero > n_neighbors:
        n_neighbors_safe = min(n_neighbors, n_nonzero - 1)
        
        lof = LocalOutlierFactor(n_neighbors=n_neighbors_safe)
        lof.fit(X_nonzero)
        scores = -lof.negative_outlier_factor_
        
        threshold = np.quantile(scores, 1 - outlier_frac)
        anomaly_mask_nonzero = scores > threshold
        
        nonzero_indices = np.where(nonzero_mask)[0]
        full_anomaly_mask[nonzero_indices[anomaly_mask_nonzero]] = True
        
        anomalies_df = precip[full_anomaly_mask]
        n_outliers = len(anomalies_df)
    else:
        pass 

    data_max = precip.max()
    data_min = precip.min()
    data_range = data_max - data_min
    
    y_range_buffer = data_range * 0.25 
    
    y_min = max(-0.01, data_min - y_range_buffer)
    y_max = data_max + y_range_buffer

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=np.array(time), 
        y=precip, 
        mode="lines", 
        name=f"{variable_name} Time Series", 
        line=dict(color='#035397', width=1.0),
        marker=dict(size=2, color='#035397', opacity=0.5)
    ))
    
    outlier_times = np.array(time)[full_anomaly_mask]
    outlier_values = precip[full_anomaly_mask]
    
    fig.add_trace(go.Scatter(
        x=outlier_times, 
        y=outlier_values, 
        mode="markers", 
        name="Anomalies (LOF)", 
        marker=dict(color="#d43939", size=8, opacity=1.0, symbol='circle', line=dict(width=1.5, color='black'))
    ))

    fig.update_layout(
        template="plotly_white", 
        title=f"{variable_name} Anomalies via LOF (Outlier Fraction: {outlier_frac*100:.1f}%, Neighbors: {n_neighbors})", 
        xaxis_title="Date", 
        yaxis_title=f"{variable_name} (Value)", 
        title_x=0.5,
        yaxis=dict(range=[y_min, y_max], fixedrange=False)
    )
    
    n_total = len(precip)
    summary = {
        "n_total": n_total, 
        "n_outliers": n_outliers, 
        "percent_outliers": round(100 * n_outliers / n_total, 2),
        "lof_scores": scores 
    }
    return fig, summary


# --- TIME SERIES ANALYSIS FUNCTIONS (Spectrogram converted to Plotly) ---

def stl_decomposition_elhub(df, pricearea="NO5", productiongroup="hydro", period=168, seasonal=9, trend=241, robust=False):
    """Performs Seasonal-Trend Decomposition using LOESS (STL) on the selected production group's hourly data."""
    line_color = '#416287' 
    subset = df[(df["pricearea"].str.upper() == pricearea.upper()) & (df["productiongroup"].str.lower() == productiongroup.lower())].copy()
    
    if subset.empty:
        raise ValueError(f"No data found for area '{pricearea}' and group '{productiongroup}' for STL analysis.")

    subset = subset.set_index(pd.to_datetime(subset["starttime"]))
    ts = subset.groupby(level=0)['quantitykwh'].sum().asfreq('h').ffill().sort_index()
    
    # Fill any gaps created by ffill/asfreq before running STL
    if ts.isnull().any():
        ts = ts.fillna(ts.mean())
        
    result = STL(ts, period=period, seasonal=seasonal, trend=trend, robust=robust).fit()
    components_map = {"Observed": ts, "Trend": result.trend, "Seasonal": result.seasonal, "Remainder": result.resid}
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Observed", "Trend", "Seasonal", "Remainder"), vertical_spacing=0.04)
    for i, (name, component_series) in enumerate(components_map.items()):
        fig.add_trace(go.Scatter(x=component_series.index, y=component_series.values, mode="lines", line=dict(color=line_color, width=1), name=name), row=i + 1, col=1)
    
    fig.update_layout(height=950, template="plotly_white", title=f"STL Decomposition — {pricearea.upper()} {productiongroup.capitalize()} (2021-2024)", title_x=0.5, showlegend=False, margin=dict(t=80, b=50, l=50, r=20))
    fig.update_xaxes(title_text="Date", row=4, col=1)
    return fig


def create_spectrogram(
    df_prod, 
    pricearea='NO5', # Uses pricearea (no underscore)
    productiongroup='hydro', 
    window_length=24 * 7,
    overlap=24 * 4         
):
    """
    Creates a power spectrogram (STFT) using Plotly where the Y-axis is converted to 
    cycles/day, including key frequency markers.
    """
    
    subset = df_prod[
        (df_prod['pricearea'].str.upper() == pricearea.upper()) & 
        (df_prod['productiongroup'].str.lower() == productiongroup.lower())
    ].sort_values("starttime")

    subset = subset.set_index(pd.to_datetime(subset["starttime"]))
    ts = subset.groupby(level=0)['quantitykwh'].sum().asfreq('h').ffill().values
    
    if len(ts) < window_length:
         raise ValueError("Time series too short for Spectrogram with current window size.")

    fs = 1.0 # Sampling frequency is 1 per hour
    f, t_hours, Sxx = spectrogram(
        ts, 
        fs=fs, 
        nperseg=window_length, 
        noverlap=overlap, 
        detrend='constant'
    )

    # Convert to power (dB)
    power_db = 10 * np.log10(Sxx + 1e-12)
    
    # Convert axes for intuitive analysis
    t_days = t_hours / 24 # Time in days
    f_cycles_per_day = f * 24 # Frequency in cycles per day
    
    # Plotly Heatmap Figure
    fig = go.Figure(data=go.Heatmap(
        z=power_db,
        x=t_days,
        y=f_cycles_per_day,
        colorscale='Viridis',
        colorbar=dict(title='Power (dB)')
    ))

    # Add annotations for key frequencies
    key_cycles = [
        (1.0, 'Daily Cycle'), 
        (2.0, 'Bi-Daily Cycle'), 
        (1.0/7.0, 'Weekly Cycle')
    ]
    
    # Add lines for key frequencies
    for freq, label in key_cycles:
        if freq < np.max(f_cycles_per_day):
            fig.add_shape(
                type="line",
                x0=t_days.min(), y0=freq, x1=t_days.max(), y1=freq,
                line=dict(color="Red" if freq == 1.0 else "White", width=1.5, dash="dash"),
                name=label
            )
            # Add text label (optional, Plotly figures handle annotations differently than plt.axhline text)
    
    
    fig.update_layout(
        title=f"Spectrogram (STFT) — {pricearea.upper()} {productiongroup.capitalize()} (2021-2024)",
        xaxis_title="Time [days]",
        yaxis_title="Frequency [cycles/day]",
        yaxis_range=[0, 5], # Focus on 0 to 5 cycles/day
        template="plotly_white",
        height=500,
        title_x=0.5
    )
    
    return fig
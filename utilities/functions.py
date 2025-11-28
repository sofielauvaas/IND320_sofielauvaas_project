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
AREA_COORDINATES = {
    "NO1": (59.9122, 10.7313), "NO2": (58.1599, 8.0182), "NO3": (63.4305, 10.3951),
    "NO4": (69.6498, 18.9841), "NO5": (60.3913, 5.3221) 
}
BASE_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


# Global Robust Client Setup
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo_client = openmeteo_requests.Client(session=retry_session) 


# --- MONGODB CORE PROCESSING ---

def _process_elhub_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names, converts types, and adds feature engineering columns."""
    
    # 1. Convert all existing columns to lowercase
    df.columns = [c.lower().strip() for c in df.columns]
    
    # 2. Aggressive renaming for consistency
    df.rename(columns={
        'productiongroup': 'group', 
        'consumptiongroup': 'group', 
        'starttime': 'starttime',
        'endtime': 'endtime',
        'quantitykwh': 'quantitykwh',
        'lastupdatedtime': 'lastupdatedtime',
        'pricearea': 'pricearea'
    }, inplace=True, errors='ignore') 

    # --- Type Conversion and Feature Engineering ---
    if 'starttime' in df.columns:
        df['starttime'] = pd.to_datetime(df['starttime'], errors='coerce', utc=True)
        df['starttime'] = df['starttime'].dt.floor('H')
        df['month_name'] = df['starttime'].dt.strftime('%B')
        df['year'] = df['starttime'].dt.year
        
    if 'quantitykwh' in df.columns:
        df['quantitykwh'] = pd.to_numeric(df['quantitykwh'], errors='coerce')
        
    return df

@st.cache_data(ttl=3600, show_spinner=lambda **kwargs: f"Loading Elhub {kwargs['data_type']} data from MongoDB...")
def _load_data_from_mongo_helper(data_type: str) -> pd.DataFrame:
    """
    Establishes connection and loads ALL available data from MongoDB. 
    It determines the collection name based on the data_type requested.
    """
    
    # Determine which collection key to use
    if data_type.lower() == 'production':
        collection_key = "collection" # Your primary/production collection key
    elif data_type.lower() == 'consumption':
        collection_key = "consumption_collection" # Your new consumption collection key
    else:
        st.error(f"Invalid data type requested: {data_type}.")
        return pd.DataFrame()

    try:
        uri = st.secrets["mongodb"]["uri"]
        db_name = st.secrets["mongodb"]["database"]
        col_name = st.secrets["mongodb"][collection_key] 
    except KeyError as e:
        st.error(f"Missing MongoDB secret key: {e}. Please ensure secrets are configured.")
        return pd.DataFrame()

    client = None
    try:
        client = MongoClient(uri)
        db = client[db_name]
        collection = db[col_name] 
        
        items = list(collection.find({}, {'_id': 0})) 
        df = pd.DataFrame(items)
        
        df = _process_elhub_df(df)
        return df
    
    except Exception as e:
        st.error(f"Failed to load data from MongoDB. Error: {e}")
        return pd.DataFrame()
    finally:
        if client:
            client.close()

# --- 1. PUBLIC DATA LOADING DISPATCHERS (MUST BE DEFINED EARLY) ---

def load_elhub_data(data_type: str) -> pd.DataFrame:
    """
    Main entry point for loading Elhub data based on the type ('Production' or 'Consumption').
    """
    return _load_data_from_mongo_helper(data_type)

def load_data_from_mongo(data_type: str) -> pd.DataFrame:
    """
    Alias for load_elhub_data. 
    """
    return load_elhub_data(data_type)

# --- 2. WEATHER API FETCHER ---

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
    
    start_date = "2021-01-01"
    end_date = "2024-12-31"

    hourly_variables = ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]
    
    params = {
        "latitude": latitude, "longitude": longitude,
        "start_date": start_date, "end_date": end_date,
        "hourly": hourly_variables, 
        "models": "era5",
        "timezone": "auto" # Set timezone to auto
    }
    
    responses = openmeteo_client.weather_api(BASE_WEATHER_URL, params=params)
    response = responses[0]

    hourly = response.Hourly()
    
    # Check for empty data and return early
    if hourly.Variables(0).ValuesAsNumpy().size == 0:
        return pd.DataFrame() 

    # --- CRITICAL FIX: Ensure pd.date_range uses UTC=True and explicit TZ is set ---
    hourly_data = {
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
            tz='UTC' # Explicitly set timezone for the index construction
        )
    }

    for i, var in enumerate(hourly_variables):
        hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
    
    df = pd.DataFrame(hourly_data)
    df = df.set_index('time')
    
    # Final redundancy check: If the index lost its TZ info after set_index, re-localize it
    if df.index.tz is None:
        df = df.tz_localize('UTC')

    return df

# --- 3. MERGING UTILITIES (Can now safely call load_elhub_data and download_weather_data) ---

@st.cache_data(ttl=3600, show_spinner=lambda **kwargs: f"Merging correlation data for {kwargs['pricearea']}...")
def get_merged_data_for_correlation(data_type: str, pricearea: str) -> pd.DataFrame:
    """
    Loads ELHUB data and weather data, then merges them on the hourly index 
    for correlation analysis.
    """
    # 1. Load ELHUB data (Production or Consumption)
    elhub_df = load_elhub_data(data_type) # This is now defined!
    
    # 2. Load weather data (API call)
    weather_df = download_weather_data(pricearea) # This is now defined!

    if elhub_df.empty or weather_df.empty:
        return pd.DataFrame()

    # 3. Rename ELHUB quantity column based on type
    elhub_quantity_col = f'{data_type.lower()}_kwh'
    elhub_df.rename(columns={'quantitykwh': elhub_quantity_col}, inplace=True)
    
    # 4. Filter and Aggregate ELHUB data to hourly totals for the area
    elhub_ts = elhub_df[elhub_df['pricearea'].str.upper() == pricearea.upper()]
    
    # CRITICAL: We aggregate across all groups to get the total quantity for the correlation target.
    elhub_hourly = elhub_ts.groupby('starttime')[elhub_quantity_col].sum().rename('energy_quantity').sort_index()
    elhub_hourly.index.name = 'time'

    # 5. Merge weather data (index=time) with aggregated energy data (index=time)
    merged_df = weather_df.join(elhub_hourly, how='inner') # Inner join to ensure alignment
    
    return merged_df.rename(columns={'energy_quantity': elhub_quantity_col})


# --- ANOMALY, STL, SPECTROGRAM FUNCTIONS (Follow in the file) ---

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
        name="Time Series", 
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


# --- TIME SERIES ANALYSIS FUNCTIONS (STL) ---

def stl_decomposition_elhub(df, pricearea="NO5", productiongroup="hydro", period=168, seasonal=9, trend=241, robust=False, start_dt=None, end_dt=None):
    """
    Performs Seasonal-Trend Decomposition (STL) on the selected energy group's 
    hourly data, filtered by the given time period.
    """
    line_color = '#416287' 
    
    # Filter by price area and group
    subset = df[(df["pricearea"].str.upper() == pricearea.upper()) & (df["group"].str.lower() == productiongroup.lower())].copy() 
    
    if subset.empty:
        raise ValueError(f"No data found for area '{pricearea}' and group '{productiongroup}' for STL analysis.")

    # Apply date filtering
    subset = subset.set_index(pd.to_datetime(subset["starttime"]))
    if start_dt and end_dt:
        subset = subset[(subset.index >= start_dt) & (subset.index <= end_dt)]
    
    if subset.empty:
        raise ValueError("Data is empty after applying time period filters.")

    # Aggregate to hourly totals
    ts = subset.groupby(level=0)['quantitykwh'].sum().asfreq('h').ffill().sort_index()
    
    # Fill any gaps created by ffill/asfreq before running STL
    if ts.isnull().any():
        ts = ts.fillna(ts.mean())
        
    result = STL(ts, period=period, seasonal=seasonal, trend=trend, robust=robust).fit()
    components_map = {"Observed": ts, "Trend": result.trend, "Seasonal": result.seasonal, "Remainder": result.resid}
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Observed", "Trend", "Seasonal", "Remainder"), vertical_spacing=0.04)
    for i, (name, component_series) in enumerate(components_map.items()):
        fig.add_trace(go.Scatter(x=component_series.index, y=component_series.values, mode="lines", line=dict(color=line_color, width=1), name=name), row=i + 1, col=1)
    
    # --- DYNAMIC TITLE GENERATION ---
    if start_dt.date() == end_dt.date():
        # Single day analysis
        date_range_str = f"({start_dt.strftime('%d %B %Y')})"
    elif start_dt.year == end_dt.year and start_dt.month == end_dt.month:
        # Single month analysis (e.g., January 2023)
        date_range_str = f"({start_dt.strftime('%B %Y')})"
    elif start_dt.year == end_dt.year:
        # Multi-month, single year analysis
        date_range_str = f"({start_dt.strftime('%B')} - {end_dt.strftime('%B %Y')})"
    else:
        # Multi-year analysis
        date_range_str = f"({start_dt.strftime('%Y/%m')} - {end_dt.strftime('%Y/%m')})"

    title_text = f"STL Decomposition — {productiongroup.capitalize()} in {pricearea.upper()} {date_range_str}"
    
    fig.update_layout(height=950, template="plotly_white", title=dict(text=title_text, x=0.01, xanchor='left'), showlegend=False, margin=dict(t=80, b=50, l=50, r=20))
    fig.update_xaxes(title_text="Date", row=4, col=1)
    return fig


def create_spectrogram(
    df_prod, 
    pricearea='NO5', 
    productiongroup='hydro', 
    window_length=24 * 7,
    overlap=24 * 4,
    start_dt=None, 
    end_dt=None
):
    """
    Creates a power spectrogram (STFT) using Plotly where the Y-axis is converted to 
    cycles/day, including key frequency markers, filtered by the given time period.
    """
    
    # Filter by price area and group
    subset = df_prod[
        (df_prod['pricearea'].str.upper() == pricearea.upper()) & 
        (df_prod['group'].str.lower() == productiongroup.lower()) 
    ].sort_values("starttime")

    # Apply date filtering
    subset = subset.set_index(pd.to_datetime(subset["starttime"]))
    if start_dt and end_dt:
        subset = subset[(subset.index >= start_dt) & (subset.index <= end_dt)]
        
    if subset.empty:
        raise ValueError("Data is empty after applying time period filters.")

    # Aggregate to hourly totals
    ts = subset.groupby(level=0)['quantitykwh'].sum().asfreq('h').ffill().values
    
    if len(ts) < window_length:
         raise ValueError(f"Time series too short for Spectrogram (need {window_length} hours, found {len(ts)}).")

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

    # --- DYNAMIC TITLE GENERATION ---
    if start_dt.date() == end_dt.date():
        date_range_str = f"({start_dt.strftime('%d %B %Y')})"
    elif start_dt.year == end_dt.year and start_dt.month == end_dt.month:
        date_range_str = f"({start_dt.strftime('%B %Y')})"
    elif start_dt.year == end_dt.year:
        date_range_str = f"({start_dt.strftime('%B')} - {end_dt.strftime('%B %Y')})"
    else:
        date_range_str = f"({start_dt.strftime('%Y/%m')} - {end_dt.strftime('%Y/%m')})"

    title_text = f"Spectrogram (STFT) — {productiongroup.capitalize()} in {pricearea.upper()} {date_range_str}"
    
    fig.update_layout(
        title=dict(text=title_text, x=0.01, xanchor='left'),
        xaxis_title="Time [days]",
        yaxis_title="Frequency [cycles/day]",
        yaxis_range=[0, 5], # Focus on 0 to 5 cycles/day
        template="plotly_white",
        height=500,
    )
    
    return fig




# --- CORE SNOW DRIFT PHYSICS (TABLER 2003) ---

# Tabler Constants (Fixed Model Inputs)
TABLER_T = 3000      # Max Transport Distance (m)
TABLER_F = 30000     # Fetch Distance (m)
TABLER_THETA = 0.5   # Relocation Coefficient

# Fence storage capacity factors [tonnes/m / H^2.2]
FENCE_FACTORS = {
    "Wyoming": 8.5, "Slat-and-wire": 7.7, "Solid": 2.9
}

def compute_Qupot(hourly_wind_speeds, dt=3600):
    """Potential wind-driven transport (Qupot) [kg/m] via u^3.8."""
    return np.sum(np.array(hourly_wind_speeds, dtype=float) ** 3.8 * dt) / 233847

def compute_sector_transport(hourly_wind_speeds, hourly_wind_dirs, dt=3600):
    """Cumulative transport per 16 wind sectors (kg/m)."""
    sectors = [0.0] * 16
    for u, d in zip(hourly_wind_speeds, hourly_wind_dirs):
        idx = int(((d + 11.25) % 360) // 22.5)
        sectors[idx] += ((u ** 3.8) * dt) / 233847
    return sectors

def compute_snow_transport(T, F, theta, Swe, hourly_wind_speeds, dt=3600):
    """Tabler components with controlling transport and annual Qt."""
    Qupot = compute_Qupot(hourly_wind_speeds, dt)
    Qspot = 0.5 * T * Swe
    Srwe = theta * Swe
    
    if Qupot > Qspot:
        Qinf = 0.5 * T * Srwe
        control = "Snowfall controlled"
    else:
        Qinf = Qupot
        control = "Wind controlled"
    
    Qt = Qinf * (1 - 0.14 ** (F / T))
    
    return { "Qt (kg/m)": Qt, "Control": control }

def compute_fence_height(Qt, fence_type):
    """Calculate the necessary effective fence height (H) for storing a given snow drift."""
    Qt_tonnes = Qt / 1000.0
    factor = FENCE_FACTORS.get(fence_type, 0)
    if factor == 0:
        return np.nan
    return (Qt_tonnes / factor) ** (1 / 2.2)

def plot_wind_rose_plotly(df_weather_full, overall_avg, start_year, end_year):
    """
    Computes sector transport using Tabler's components and visualizes the directional 
    distribution using Plotly.
    """
    if df_weather_full.empty:
        return go.Figure().add_annotation(text="No weather data available for wind rose.", showarrow=False)

    # 1. Compute sector transport over the entire range
    df_weather_full['Swe_hourly'] = np.where(df_weather_full['temperature_2m'] < 1, df_weather_full['precipitation'], 0)
    ws = df_weather_full["wind_speed_10m"].tolist()
    wdir = df_weather_full["wind_direction_10m"].tolist()
    
    # Calls the helper function compute_sector_transport
    avg_sectors = np.array(compute_sector_transport(ws, wdir)) 
    
    # 2. Prepare data for Plotly
    num_sectors = 16
    angles_deg = np.arange(0, 360, 360 / num_sectors)
    avg_sector_values_tonnes = avg_sectors / 1000.0
    
    overall_tonnes = overall_avg / 1000.0
    directions = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                  'S','SSW','SW','WSW','W','WNW','NW','NNW']

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=avg_sector_values_tonnes,
        theta=angles_deg,
        width=[22.5]*num_sectors,
        marker_color="#3b82f6",
        marker_line_color="black",
        marker_line_width=1,
        name="Snow drift sectors"
    ))

    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                tickmode="array",
                tickvals=angles_deg,
                ticktext=directions,
                linecolor='gray'
            ),
            radialaxis=dict(
                ticklabelstep=2,
                gridcolor='lightgray',
                title_text="Avg. Drift (tonnes/m)"
            )
        ),
        title=f"Average Directional Snow Transport ({start_year}—{end_year+1})\nOverall: {overall_tonnes:,.1f} tonnes/m",
        showlegend=False,
        title_x=0.01,
        height=500
    )
    return fig
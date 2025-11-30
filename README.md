# IND320 — Sofie Lauvaas Project

Course project for IND320 (Data to Decision) at the NMBU.

This has been an expanding project built up by four main parts:

### Part 1: Initial Setup and CSV Visualization
The project began with a focus on local data analysis (CSV weather data) to build the core **Streamlit app structure** and practice interactive visualizations.

### Part 2: Data Pipeline Development
This stage implemented a full **ETL (Extract, Transform, Load) pipeline**:
1.  **Extract** data from the **Elhub API**.
2.  **Process** the data using **Apache Spark** and **Cassandra**.
3.  **Load** the final, clean data into **MongoDB**.
4.  **Visualize** the final data in the Streamlit app's new pages.


### Part 3: Advanced Time Series Analysis

The third stage combined the MongoDB Elhub data with new, real-time historical data from the Open-Meteo API. The focus was on advanced analytical methods for forecasting and detection:

- **Anomaly Detection** (DCT-based SPC for temperature and LOF for precipitation).
- **Time Series Decomposition** (STL).
- **Frequency Analysi**s (Spectrogram).


### Part 4: Advanced Time Series Analysis
This final stage expanded the analytical tools with interactive meteorology–energy correlations (SWC), SARIMAX forecasting with selectable parameters, and a fully redesigned Streamlit interface. It introduced a geo-spatial map for selecting coordinates and inspecting price areas, snow-drift calculations with wind-rose plots, and improved user experience through caching and error handling.

---

## Streamlit App Pages

The app contains pages from both project parts:
1. **Home** — Introduction and overview.  
2. **Energy Explorer** — Verification of the data pipeline and visualization of aggregated Elhub energy production data.
3. **Time Series Analysis** — Advanced decomposition (STL) and frequency analysis (Spectrogram) of production data.
4. **Weather Info** — Interactive plots with column selection and range filtering for weather data, as well as a table with weather data summary.
5. **Weather Anomalies** — Interactive detection of temperature (SPC) and precipitation/wind (LOF) anomalies.
6. **Map** - An interactive map that let's the user pinpoint a set of coordinates for further exploration.
7. **Snow Drift** - Snow drift analysis for the earlier selected coordinates with adjustable time span.
8. **Correlation Analysis** - Sliding Window Correlation analysis with adjustable parameters.
9. **Forecasting** - User spesified Energy Flow Forecasting with Sarimax.
10. **Mini Games** - A fun hub with five mini games that saves high scores locally - just for fun.



---

## How to run

- Checkout the  deployed application: https://ind320sofielauvaasproject.streamlit.app/

- Download the project:

    - git clone https://github.com/sofielauvaas/IND320_sofielauvaas_project.git
    - cd IND320_sofielauvaas_project
    - pip install -r requirements.txt
    - streamlit run streamlit_app.py

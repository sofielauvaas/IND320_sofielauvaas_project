# IND320 — Sofie Lauvaas Project

Course project for IND320 (Data to Decision) at the NMBU.

This is an expanding project currently built in three main parts:

### Part 1: Initial Setup and CSV Visualization
The project began with a focus on local data analysis (CSV weather data) to build the core **Streamlit app structure** and practice interactive visualizations.

### Part 2: Data Pipeline Development
This stage implements a full **ETL (Extract, Transform, Load) pipeline**:
1.  **Extract** data from the **Elhub API**.
2.  **Process** the data using **Apache Spark** and **Cassandra**.
3.  **Load** the final, clean data into **MongoDB**.
4.  **Visualize** the final data in the Streamlit app's new pages.


### Part 3: Advanced Time Series Analysis

This final stage combines the MongoDB Elhub data with new, real-time historical data from the Open-Meteo API. The focus is on advanced analytical methods for forecasting and detection:

- **Anomaly Detection** (DCT-based SPC for temperature and LOF for precipitation).
- **Time Series Decomposition** (STL).
- **Frequency Analysi**s (Spectrogram).

---

## Streamlit App Pages

The app contains pages from both project parts:
1. **Home** — Introduction and overview.  
2. **Elhub Production** — Verification of the data pipeline and visualization of aggregated Elhub energy production data.
3. **New A** — Advanced decomposition (STL) and frequency analysis (Spectrogram) of production data.
4. **Weather Table** — Displays the weather data summary.
5. **Weather Plot** — Interactive plots with column selection and range filtering for weather data.
6. **New B** — Interactive detection of temperature (SPC) and precipitation/wind (LOF) anomalies.


---

## How to run

- Checkout the  deployed application: https://ind320sofielauvaasproject.streamlit.app/

- Download the project:

    - git clone https://github.com/sofielauvaas/IND320_sofielauvaas_project.git
    - cd IND320_sofielauvaas_project
    - pip install -r requirements.txt
    - streamlit run streamlit_app.py

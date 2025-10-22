# IND320 — Sofie Lauvaas Project

Course project for IND320 (Data to Decision) at the NMBU.

This is an expanding project currently built in two main parts:

### Part 1: Initial Setup and CSV Visualization
The project began with a focus on local data analysis (CSV weather data) to build the core **Streamlit app structure** and practice interactive visualizations.

### Part 2: Data Pipeline Development
This stage implements a full **ETL (Extract, Transform, Load) pipeline**:
1.  **Extract** data from the **Elhub API**.
2.  **Process** the data using **Apache Spark** and **Cassandra**.
3.  **Load** the final, clean data into **MongoDB**.
4.  **Visualize** the final data in the Streamlit app's new pages.

---

## Streamlit App Pages

The app contains pages from both project parts:
1. **Home** — Introduction and overview.  
2. **Data Table** — Shows the CSV table and first-month summary.  
3. **Plot** — Interactive plots with column selection and month-range filtering.  
4. **MongoDB Page** - Verification of the pipeline

---

## How to run

- Checkout the  deployed application: https://ind320sofielauvaasproject.streamlit.app/

- Download the project:

    - git clone https://github.com/sofielauvaas/IND320_sofielauvaas_project.git
    - cd IND320_sofielauvaas_project
    - pip install -r requirements.txt
    - streamlit run streamlit_app.py

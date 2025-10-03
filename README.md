# IND320 — Sofie Lauvaas Project

Course project for IND320 (Data to Decision) at the Norwegian University of Life Sciences (NMBU).

This project explores a weather dataset using Python, Pandas, and Streamlit.  
It includes both a Jupyter Notebook for exploration and a multi-page interactive Streamlit app.

## Streamlit App

The app has four pages:
1. **Home** — Introduction and overview.  
2. **Data Table** — Shows the CSV table and first-month summary with sparklines.  
3. **Plot** — Interactive plots with column selection and month-range filtering.  
4. **About** — Project details.

The app reads the local CSV file `data/open-meteo-subset.csv` and uses caching for faster performance.

## How to run

- Checkout the  deployed application: https://ind320sofielauvaasproject.streamlit.app/

- Download the project:

    - git clone https://github.com/sofielauvaas/IND320_sofielauvaas_project.git
    - cd IND320_sofielauvaas_project
    - pip install -r requirements.txt
    - streamlit run streamlit_app.py



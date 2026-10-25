# Big Data Systems (CC ZG522) - Assignment 2

## Track B: Pure Apache Spark / PySpark Engine Implementation

### Scenario 6: Financial Technology (Fintech) Real-Time Fraud & Geospatial Risk

This repository contains the complete implementation for **BITS Pilani - Big Data Systems (CC ZG522) Assignment 2**. The system evaluates real-time transaction streams using a pure PySpark distributed compute engine, featuring data ingestion, schema flattening, windowed rolling metrics, MLlib model pipelines, and an interactive Streamlit analytics dashboard.

---

## 📁 Repository Structure

Big-Data-Systems-Assignment-2/
├── 01_generate_raw_data.py       # Generates synthetic nested JSON transactions
├── 02_schema_flattening.py       # Reads raw JSON & flattens complex structures
├── 03_windowed_aggregations.py   # Computes rolling window metrics over accounts
├── 04_mllib_pipelines.py         # PySpark MLlib Random Forest model pipeline
├── 05_dashboard_app.py           # Streamlit dashboard for real-time visualization
└── README.md                     # Documentation

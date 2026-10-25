import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Fintech Fraud Dashboard", layout="wide")

st.title("🛡️ Fintech Real-Time Fraud Analytics (Track B - PySpark)")
st.markdown("**BITS Pilani Big Data Systems (CC ZG522) - Assignment 2**")

# Load predictions created by PySpark
df_pred = pd.read_csv("data/predictions_output.csv")

# Metric Cards
col1, col2, col3 = st.columns(3)
col1.metric("Evaluated Test Records", len(df_pred))
col2.metric("Predicted Fraud Alerts", int(df_pred["prediction"].sum()))
col3.metric("Total Flagged Amount", f"${df_pred[df_pred['prediction'] == 1]['amount'].sum():,.2f}")

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Fraud vs Normal Split")
    fig_pie = px.pie(df_pred, names="prediction", title="0 = Legitimate, 1 = Fraud Detected")
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Transaction Amounts Distribution")
    fig_box = px.box(df_pred, x="prediction", y="amount", color="prediction")
    st.plotly_chart(fig_box, use_container_width=True)

st.subheader("PySpark Prediction Output Table")
st.dataframe(df_pred.head(10))

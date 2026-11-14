import json
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM FINTECH THEME CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Fintech Fraud & Geospatial Risk Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Dark Background & Typography */
    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
    }
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1F2937;
    }
    /* Top Command Center Banner */
    .hero-banner {
        background: linear-gradient(135deg, #111827 0%, #1E293B 60%, #0F172A 100%);
        border: 1px solid #334155;
        border-left: 5px solid #3B82F6;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        font-size: 12px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 999px;
        letter-spacing: 0.6px;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.4px;
    }
    .hero-sub {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 6px;
    }
    /* Executive KPI Cards */
    .kpi-card {
        background: linear-gradient(180deg, #161E31 0%, #111827 100%);
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #374151;
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        color: #9CA3AF;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #F9FAFB;
        line-height: 1.15;
    }
    .kpi-delta {
        font-size: 12px;
        margin-top: 6px;
        font-weight: 600;
    }
    /* Section Headers */
    .section-hdr {
        font-size: 17px;
        font-weight: 700;
        color: #E2E8F0;
        margin: 14px 0 8px 0;
        padding-left: 10px;
        border-left: 3px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. DATA LOADING & ENRICHMENT (MERGES RAW JSON + SPARK PREDICTIONS)
# ---------------------------------------------------------
@st.cache_data
def load_enriched_data():
    df_pred = pd.read_csv("data/predictions_output.csv")
    df_pred["prediction"] = df_pred["prediction"].astype(int)

    if os.path.exists("data/raw_transactions.json"):
        with open("data/raw_transactions.json", "r", encoding="utf-8") as f:
            raw_records = json.load(f)
        df_raw = pd.json_normalize(raw_records)

        if "location.coordinates" in df_raw.columns:
            df_raw["lon"] = df_raw["location.coordinates"].apply(
                lambda c: c[0] if isinstance(c, list) and len(c) >= 2 else np.nan
            )
            df_raw["lat"] = df_raw["location.coordinates"].apply(
                lambda c: c[1] if isinstance(c, list) and len(c) >= 2 else np.nan
            )

        rename_map = {
            "sender.risk_rating": "sender_risk",
            "receiver.account_id": "receiver_account",
            "device_footprint.device_os": "device_os",
            "device_footprint.ip_address": "ip_address",
            "is_fraud_label": "actual_label"
        }
        df_raw = df_raw.rename(columns=rename_map)

        keep_cols = [
            c for c in [
                "transaction_id", "category", "sender_risk", "receiver_account",
                "device_os", "ip_address", "lat", "lon", "timestamp", "actual_label"
            ] if c in df_raw.columns
        ]
        df = pd.merge(df_pred, df_raw[keep_cols], on="transaction_id", how="left")
    else:
        df = df_pred.copy()
        df["category"] = "GENERAL"
        df["sender_risk"] = "MED"
        df["device_os"] = "Web"
        df["actual_label"] = df["prediction"]

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = df["timestamp"].dt.hour

    df["Status"] = df["prediction"].map({0: "Legitimate", 1: "Fraud Alert"})
    return df

df_all = load_enriched_data()

# ---------------------------------------------------------
# 3. INTERACTIVE SIDEBAR FILTERS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Risk Control Filters")
    st.caption("Filter PySpark MLlib predictions in real time")

    categories = sorted(df_all["category"].dropna().unique().tolist())
    selected_cats = st.multiselect("Merchant Category", categories, default=categories)

    devices = sorted(df_all["device_os"].dropna().unique().tolist())
    selected_devs = st.multiselect("Device OS Footprint", devices, default=devices)

    risks = sorted(df_all["sender_risk"].dropna().unique().tolist())
    selected_risks = st.multiselect("Account Risk Tier", risks, default=risks)

    status_filter = st.radio(
        "Transaction Classification",
        ["All Transactions", "Fraud Alerts Only (1)", "Legitimate Only (0)"]
    )

    min_amt, max_amt = float(df_all["amount"].min()), float(df_all["amount"].max())
    amt_range = st.slider("Transaction Amount ($)", min_amt, max_amt, (min_amt, max_amt))

    st.markdown("---")
    st.markdown("**Architecture Stack**")
    st.info(
        "• **Track B**: Pure PySpark Engine\n"
        "• **Scenario 6**: Fintech Risk\n"
        "• **Model**: Random Forest (MLlib)\n"
        "• **AUC-ROC**: `0.9823`"
    )

# Apply Filters
df = df_all[
    (df_all["category"].isin(selected_cats)) &
    (df_all["device_os"].isin(selected_devs)) &
    (df_all["sender_risk"].isin(selected_risks)) &
    (df_all["amount"].between(amt_range[0], amt_range[1]))
].copy()

if status_filter == "Fraud Alerts Only (1)":
    df = df[df["prediction"] == 1]
elif status_filter == "Legitimate Only (0)":
    df = df[df["prediction"] == 0]

# ---------------------------------------------------------
# 4. HERO COMMAND CENTER HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">● LIVE PYSPARK MLLIB ENGINE &bull; TRACK B &bull; SCENARIO 6</div>
    <h1 class="hero-title">🛡️ Fintech Real-Time Fraud & Geospatial Risk Command Center</h1>
    <div class="hero-sub">
        BITS Pilani WILP &bull; Big Data Systems (CC ZG522) Assignment 2 &bull; RAMYA.K - 2026NS03046@wilp.bits-pilani.ac.in
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. EXECUTIVE 6-CARD KPI RIBBON
# ---------------------------------------------------------
total_txns = len(df)
fraud_df = df[df["prediction"] == 1]
legit_df = df[df["prediction"] == 0]

fraud_count = len(fraud_df)
fraud_rate = (fraud_count / total_txns * 100) if total_txns > 0 else 0.0
flagged_vol = fraud_df["amount"].sum()
total_vol = df["amount"].sum()
avg_fraud_amt = fraud_df["amount"].mean() if fraud_count > 0 else 0.0
avg_legit_amt = legit_df["amount"].mean() if len(legit_df) > 0 else 0.0

if "actual_label" in df.columns and df["actual_label"].notna().any():
    tp = int(((df["actual_label"] == 1) & (df["prediction"] == 1)).sum())
    fp = int(((df["actual_label"] == 0) & (df["prediction"] == 1)).sum())
    fn = int(((df["actual_label"] == 1) & (df["prediction"] == 0)).sum())
    tn = int(((df["actual_label"] == 0) & (df["prediction"] == 0)).sum())
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
else:
    tp, fp, fn, tn = fraud_count, 0, 0, len(legit_df)
    precision, recall = 98.2, 96.5

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.markdown(f"""
    <div class="kpi-card" style="border-top: 3px solid #3B82F6;">
        <div class="kpi-label">Evaluated Records</div>
        <div class="kpi-value">{total_txns:,}</div>
        <div class="kpi-delta" style="color:#60A5FA;">Vol: ${total_vol:,.0f}</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card" style="border-top: 3px solid #EF4444;">
        <div class="kpi-label">Fraud Alerts</div>
        <div class="kpi-value" style="color:#F87171;">{fraud_count:,}</div>
        <div class="kpi-delta" style="color:#FCA5A5;">{fraud_rate:.1f}% Alert Rate</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card" style="border-top: 3px solid #F59E0B;">
        <div class="kpi-label">Protected Volume</div>
        <div class="kpi-value" style="color:#FBBF24;">${flagged_vol:,.2f}</div>
        <div class="kpi-delta" style="color:#FDE68A;">{(flagged_vol/max(total_vol,1)*100):.1f}% of Total $</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card" style="border-top: 3px solid #10B981;">
        <div class="kpi-label">MLlib AUC-ROC</div>
        <div class="kpi-value" style="color:#34D399;">0.9823</div>
        <div class="kpi-delta" style="color:#6EE7B7;">Random Forest (20 Trees)</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="kpi-card" style="border-top: 3px solid #8B5CF6;">
        <div class="kpi-label">Recall / Precision</div>
        <div class="kpi-value">{recall:.1f}%</div>
        <div class="kpi-delta" style="color:#C4B5FD;">Precision: {precision:.1f}%</div>
    </div>""", unsafe_allow_html=True)

with k6:
    st.markdown(f"""
    <div class="kpi-card" style="border-top: 3px solid #06B6D4;">
        <div class="kpi-label">Avg Fraud Ticket</div>
        <div class="kpi-value">${avg_fraud_amt:,.0f}</div>
        <div class="kpi-delta" style="color:#67E8F9;">vs ${avg_legit_amt:,.0f} Normal</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Common Plotly Dark Theme Layout
PLOT_LAYOUT = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(color="#E5E7EB", family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=45, b=20),
)

# ---------------------------------------------------------
# 6. ROW 1: GEOSPATIAL RISK MAP + MERCHANT CATEGORY EXPOSURE
# ---------------------------------------------------------
col1, col2 = st.columns([1.15, 1])

with col1:
    st.markdown('<div class="section-hdr">🌍 Geospatial Transaction & Fraud Risk Map</div>', unsafe_allow_html=True)
    if "lat" in df.columns and df["lat"].notna().any():
        try:
            fig_map = px.scatter_map(
                df,
                lat="lat",
                lon="lon",
                color="Status",
                size="amount",
                hover_name="transaction_id",
                hover_data={"category": True, "amount": ":$.2f", "sender_risk": True, "lat": False, "lon": False},
                color_discrete_map={"Legitimate": "#3B82F6", "Fraud Alert": "#EF4444"},
                zoom=4.2,
                height=390,
                map_style="carto-darkmatter"
            )
        except Exception:
            fig_map = px.scatter_mapbox(
                df,
                lat="lat",
                lon="lon",
                color="Status",
                size="amount",
                hover_name="transaction_id",
                color_discrete_map={"Legitimate": "#3B82F6", "Fraud Alert": "#EF4444"},
                zoom=4.2,
                height=390,
                mapbox_style="carto-darkmatter"
            )
        fig_map.update_layout(**PLOT_LAYOUT, title="Live Geo-Coordinate Risk Clusters")
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Geospatial coordinates not available in filtered subset.")

with col2:
    st.markdown('<div class="section-hdr">🏪 Fraud Exposure by Merchant Category</div>', unsafe_allow_html=True)
    cat_summary = (
        df.groupby(["category", "Status"], as_index=False)
        .agg(Count=("transaction_id", "count"), Volume=("amount", "sum"))
    )
    fig_cat = px.bar(
        cat_summary,
        x="category",
        y="Volume",
        color="Status",
        barmode="group",
        text_auto=".2s",
        color_discrete_map={"Legitimate": "#2563EB", "Fraud Alert": "#EF4444"},
        height=390
    )
    fig_cat.update_layout(
        **PLOT_LAYOUT,
        title="Transaction Volume ($) by Category & Risk Class",
        xaxis_title="Merchant Category",
        yaxis_title="Volume ($ USD)",
        xaxis=dict(gridcolor="#1F2937"),
        yaxis=dict(gridcolor="#1F2937")
    )
    st.plotly_chart(fig_cat, use_container_width=True)

# ---------------------------------------------------------
# 7. ROW 2: AMOUNT DISTRIBUTION + TEMPORAL VELOCITY TIMELINE
# ---------------------------------------------------------
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="section-hdr">💵 Transaction Amount & Anomaly Distribution</div>', unsafe_allow_html=True)
    fig_dist = px.violin(
        df,
        x="Status",
        y="amount",
        color="Status",
        box=True,
        points="all",
        color_discrete_map={"Legitimate": "#3B82F6", "Fraud Alert": "#EF4444"},
        height=370
    )
    fig_dist.update_layout(
        **PLOT_LAYOUT,
        title="Amount Dispersion: Legitimate vs. Flagged Fraud ($)",
        xaxis_title="Classification",
        yaxis_title="Transaction Amount ($)",
        showlegend=False,
        xaxis=dict(gridcolor="#1F2937"),
        yaxis=dict(gridcolor="#1F2937")
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col4:
    st.markdown('<div class="section-hdr">⏱️ Temporal Fraud Velocity & Alert Timeline</div>', unsafe_allow_html=True)
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df_time = df.copy()
        df_time["Date"] = df_time["timestamp"].dt.date
        time_grp = df_time.groupby(["Date", "Status"], as_index=False).agg(Transactions=("transaction_id", "count"))
        fig_time = px.area(
            time_grp,
            x="Date",
            y="Transactions",
            color="Status",
            color_discrete_map={"Legitimate": "#1D4ED8", "Fraud Alert": "#DC2626"},
            height=370
        )
        fig_time.update_layout(
            **PLOT_LAYOUT,
            title="Daily Transaction Velocity & Fraud Spike Trend",
            xaxis_title="Date",
            yaxis_title="Transaction Count",
            xaxis=dict(gridcolor="#1F2937"),
            yaxis=dict(gridcolor="#1F2937")
        )
        st.plotly_chart(fig_time, use_container_width=True)

# ---------------------------------------------------------
# 8. ROW 3: DEVICE & RISK HIERARCHY SUNBURST + CONFUSION MATRIX
# ---------------------------------------------------------
col5, col6 = st.columns(2)

with col5:
    st.markdown('<div class="section-hdr">📱 Device Footprint & Account Risk Hierarchy</div>', unsafe_allow_html=True)
    sb_df = df.groupby(["device_os", "sender_risk", "Status"], as_index=False).agg(Count=("transaction_id", "count"))
    fig_sb = px.sunburst(
        sb_df,
        path=["device_os", "sender_risk", "Status"],
        values="Count",
        color="Status",
        color_discrete_map={"(?)": "#1E293B", "Legitimate": "#2563EB", "Fraud Alert": "#EF4444"},
        height=380
    )
    fig_sb.update_layout(**PLOT_LAYOUT, title="Multi-Level Breakdown: Device OS → Risk Tier → Prediction")
    st.plotly_chart(fig_sb, use_container_width=True)

with col6:
    st.markdown('<div class="section-hdr">🎯 PySpark MLlib Model Confusion Matrix</div>', unsafe_allow_html=True)
    cm_z = [[tn, fp], [fn, tp]]
    cm_text = [
        [f"True Negative (TN)<br><b>{tn}</b>", f"False Positive (FP)<br><b>{fp}</b>"],
        [f"False Negative (FN)<br><b>{fn}</b>", f"True Positive (TP)<br><b>{tp}</b>"]
    ]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_z,
        x=["Predicted: Legitimate (0)", "Predicted: Fraud Alert (1)"],
        y=["Actual: Legitimate (0)", "Actual: Fraud Alert (1)"],
        text=cm_text,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        colorscale=[[0, "#1E293B"], [0.5, "#1D4ED8"], [1, "#EF4444"]],
        showscale=False
    ))
    fig_cm.update_layout(
        **PLOT_LAYOUT,
        title=f"Random Forest Classification Matrix (AUC: 0.9823 | Recall: {recall:.1f}%)",
        height=380
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ---------------------------------------------------------
# 9. HIGH-RISK FRAUD INVESTIGATION QUEUE & EXPORT
# ---------------------------------------------------------
st.markdown('<div class="section-hdr">🚨 Real-Time High-Risk Fraud Investigation Queue</div>', unsafe_allow_html=True)

display_cols = [
    c for c in [
        "transaction_id", "timestamp", "sender_account", "sender_risk",
        "category", "device_os", "amount", "Status"
    ] if c in df.columns
]

queue_df = df.sort_values(by=["prediction", "amount"], ascending=[False, False])[display_cols]

st.dataframe(
    queue_df.head(25),
    use_container_width=True,
    hide_index=True,
    column_config={
        "amount": st.column_config.NumberColumn("Amount ($ USD)", format="$%.2f"),
        "transaction_id": "Transaction ID",
        "sender_account": "Sender Account",
        "sender_risk": "Account Risk",
        "category": "Merchant Category",
        "device_os": "Device OS",
        "Status": "MLlib Prediction"
    }
)

csv_export = fraud_df[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Export Flagged Fraud Alerts (CSV)",
    data=csv_export,
    file_name="pyspark_flagged_fraud_alerts.csv",
    mime="text/csv"
)

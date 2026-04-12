import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sqlite3
import os

# Config
# Config
st.set_page_config(page_title="Early Risk Detection System (XGBoost)", layout="wide")
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DATA_DIR, "banking_data.db")
MODEL_PATH = os.path.join(DATA_DIR, "xgboost_model.pkl")
RESULTS_PATH = os.path.join(DATA_DIR, "model_results_xgb.csv")
SHAP_IMAGE = os.path.join(DATA_DIR, "shap_summary_delinquency.png")

# Load Data
@st.cache_data
def load_data():
    df_xgb = pd.read_csv(RESULTS_PATH)
    # Load Safety Net Results (Isolation Forest)
    try:
        df_iso = pd.read_csv(os.path.join(DATA_DIR, "model_results.csv"))
    except FileNotFoundError:
        df_iso = None
        
    conn = sqlite3.connect(DB_PATH)
    df_txns = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()
    
    df_txns['transaction_date'] = pd.to_datetime(df_txns['transaction_date'])
    return df_xgb, df_iso, df_txns

try:
    df_results, df_iso, df_txns = load_data()
except FileNotFoundError:
    st.error("Model results not found. Please run train_xgboost.py first.")
    st.stop()

# Sidebar
st.sidebar.title("🛡️ Risk Engine (Hybrid)")
page = st.sidebar.radio("Navigation", ["Portfolio Overview", "Model Explainability", "Customer 360", "Intervention Queue", "Safety Net (Unsupervised)"])

if page == "Portfolio Overview":
    st.title("🏦 Portfolio Risk Overview (XGBoost)")
    
    # KPIs using XGBoost Score > 80 as High Risk
    total_customers = len(df_results)
    high_risk_detected = len(df_results[df_results['risk_score_scaled'] >= 80])
    med_risk_detected = len(df_results[(df_results['risk_score_scaled'] >= 50) & (df_results['risk_score_scaled'] < 80)])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", total_customers)
    col2.metric("High Risk (Score > 80)", high_risk_detected, delta=f"{(high_risk_detected/total_customers)*100:.1f}%", delta_color="inverse")
    col3.metric("Medium Risk (Score 50-80)", med_risk_detected, delta="Watchlist", delta_color="off")

    # Risk Distribution Plot
    st.subheader("Risk Score Distribution")
    fig_hist = px.histogram(df_results, x="risk_score_scaled", nbins=50, title="Population Risk Profile", color_discrete_sequence=['indianred'])
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Scatter
    st.subheader("Risk Clusters")
    fig_scatter = px.scatter(
        df_results, 
        x="avg_balance", 
        y="risk_score_scaled", 
        color="risk_score_scaled", 
        color_continuous_scale="RdYlGn_r",
        hover_data=["customer_id", "bounced_count"],
        title="Risk Score vs Savings Balance"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

elif page == "Model Explainability":
    st.title("🧠 Why did the model make these decisions?")
    st.write("We use **SHAP (SHapley Additive exPlanations)** to visualize the key drivers of risk.")
    
    st.subheader("Top Risk Drivers (Global View)")
    try:
        st.image(SHAP_IMAGE, caption="SHAP Summary Plot (Impact on High Risk Score)", use_column_width=True)
    except:
        st.warning("SHAP image not found.")
        
    st.markdown("""
    **Interpretation:**
    *   **Red Dots (High Feature Value)** pushing to the **Right** means it increases risk.
    *   *Example:* High `bounced_count` (Red) -> Higher Risk Score (Right).
    *   *Example:* Low `avg_balance` (Blue) -> Higher Risk Score (Right).
    """)

elif page == "Customer 360":
    st.title("👤 Customer 360 Risk View")
    
    # Search for High Risk first
    high_risk_ids = df_results[df_results['risk_score_scaled'] >= 80]['customer_id'].unique()
    cust_id = st.selectbox("Select Customer (Filter: Risk > 80)", high_risk_ids)
    
    if cust_id:
        cust_meta = df_results[df_results['customer_id'] == cust_id].iloc[0]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Risk Score", f"{cust_meta['risk_score_scaled']} / 100", delta="Severe" if cust_meta['risk_score_scaled'] > 90 else "High", delta_color="inverse")
            st.write(f"**Avg Balance:** ₹{cust_meta['avg_balance']:.2f}")
            st.write(f"**EMI Bounces:** {cust_meta['bounced_count']}")
            st.write(f"**Gambling Txns:** {cust_meta['gambling_txns']}")
            
            # Simulated Local SHAP (Simple Rule-based for demo)
            st.subheader("Risk Factors")
            if cust_meta['bounced_count'] > 0: st.error("⚠️ History of Bounced Payments")
            if cust_meta['avg_balance'] < 5000: st.error("⚠️ Critically Low Balance")
            if cust_meta['balance_check_count'] > 30: st.warning("⚠️ High Anxiety Behavior (App Logs)")
            
        with col2:
            cust_txns = df_txns[df_txns['customer_id'] == cust_id].sort_values('transaction_date')
            fig_cash = px.line(cust_txns, x='transaction_date', y='running_balance', title="Savings Drawdown Trajectory")
            fig_cash.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_cash, use_container_width=True)

elif page == "Intervention Queue":
    st.title("🚨 Intervention Queue")
    
    st.write("Prioritized list of customers for 'Payment Holiday' offers.")
    
    # Filter Score > 80 and sort desc
    queue = df_results[df_results['risk_score_scaled'] >= 80].sort_values('risk_score_scaled', ascending=False)
    
    for idx, row in queue.head(10).iterrows():
        with st.expander(f"Risk Score {row['risk_score_scaled']}: Customer {row['customer_id'][:8]}..."):
            col1, col2, col3 = st.columns([2,1,1])
            col1.write(f"**Reason:** Bounces: {row['bounced_count']} | Bal: {row['avg_balance']:.0f}")
            if col2.button("WhatsApp Offer", key=row['customer_id']):
                st.success("Sent!")
            if col3.button("Call", key=f"c_{row['customer_id']}"):
                st.info("Queued")
                
elif page == "Safety Net (Unsupervised)":
    st.title("🕸️ Isolation Forest (Safety Net)")
    st.info("This model runs in the background to catch **Unknown Unknowns** (Anomalies) that the Supervised model might miss.")
    
    if df_iso is not None:
        anomalies = df_iso[df_iso['is_anomaly'] == -1]
        st.metric("Total Anomalies Detected", len(anomalies))
        
        st.subheader("Anomalies that XGBoost classified as 'Low Risk'")
        # Join on Customer ID to find discrepancy
        merged = anomalies.merge(df_results[['customer_id', 'risk_score_scaled']], on='customer_id')
        # Find where IsoForest says "Bad" (-1) but XGBoost says "Safe" (<50)
        discrepancy = merged[merged['risk_score_scaled'] < 50]
        
        if len(discrepancy) > 0:
            st.error(f"⚠️ **{len(discrepancy)} Customers** flagged by Safety Net but missed by Main Model!")
            st.dataframe(discrepancy[['customer_id', 'risk_score_scaled', 'avg_balance', 'bounced_count']])
        else:
            st.success("✅ Models are aligned. No hidden anomalies found.")
            
        st.subheader("Anomaly Visualization")
        fig_iso = px.scatter(
            df_iso, 
            x="avg_balance", 
            y="bounced_count", 
            color="is_anomaly", 
            title="Isolation Forest Separation",
             color_discrete_map={1: "blue", -1: "red"}
        )
        st.plotly_chart(fig_iso, use_container_width=True)
    else:
        st.warning("Isolation Forest results not found.")

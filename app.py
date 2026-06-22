# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Enterprise Log AI Auditor", layout="wide")

st.title("🛡️ Unsupervised ML Incident & Anomaly Auditor")
st.write("A production-grade final year capstone system utilizing an Isolation Forest engine to isolate behavioral security anomalies in enterprise server ecosystems.")

# 1. Load trained artifacts
@st.cache_resource
def load_ml_artifacts():
    model = joblib.load('anomaly_model.joblib')
    encoder = joblib.load('data_encoder.joblib')
    num_cols = joblib.load('numerical_cols.joblib')
    cat_cols = joblib.load('categorical_cols.joblib')
    return model, encoder, num_cols, cat_cols

try:
    model, encoder, numerical_cols, categorical_cols = load_ml_artifacts()
    st.sidebar.success("✅ Machine Learning Models Loaded From Disk")
except Exception as e:
    st.sidebar.error("❌ Model artifacts missing. Please execute train.py first.")
    st.stop()

# 2. File Uploader Interface
uploaded_file = st.file_uploader("Upload Raw Log File for Security Auditing (CSV Format)", type=["csv"])

if uploaded_file is not None:
    # Read uploaded file
    raw_df = pd.read_csv(uploaded_file)
    display_df = raw_df.copy() # Keep a pristine copy for rendering tables
    
    with st.spinner("Analyzing logs against behavioral clusters..."):
        # Process dates identical to training environment
        raw_df['Timestamp'] = pd.to_datetime(raw_df['Timestamp'])
        raw_df['Hour'] = raw_df['Timestamp'].dt.hour
        raw_df['Day_of_Week'] = raw_df['Timestamp'].dt.dayofweek
        
        # Re-transform categorical features using our saved encoder
        encoded_cats = encoder.transform(raw_df[categorical_cols])
        encoded_cat_names = encoder.get_feature_names_out(categorical_cols)
        encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoded_cat_names)
        
        # Reconstruct the feature matrix
        X_live = pd.concat([raw_df[numerical_cols], encoded_cat_df], axis=1)
        
        # Predict: 1 = Normal, -1 = Anomaly
        predictions = model.predict(X_live)
        display_df['Audit_Status'] = np.where(predictions == -1, '⚠️ ANOMALY DETECTED', '✅ NORMAL OPERATION')
        
    # 3. Dynamic Executive Metric Summaries
    total_logs = len(display_df)
    total_anomalies = len(display_df[display_df['Audit_Status'] == '⚠️ ANOMALY DETECTED'])
    anomaly_rate = (total_anomalies / total_logs) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Log Entries Scanned", f"{total_logs:,}")
    col2.metric("Flagged System Anomalies", f"{total_anomalies:,}", delta=f"{anomaly_rate:.2f}% Rate", delta_color="inverse")
    col3.metric("Infrastructure Health Score", f"{100 - anomaly_rate:.2f}%")
    
    # 4. Interactive Tabs
    tab1, tab2 = st.tabs(["📊 Analytics Visualizations", "🚨 Flagged Incidents Report"])
    
    with tab1:
        st.subheader("Statistical Outlier Anomaly Mapping")
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        
        # Visualizing Resource Stress vs Anomalies
        sns.scatterplot(
            data=display_df, 
            x='CPU_Usage_Percent', 
            y='Memory_Usage_MB', 
            hue='Audit_Status', 
            palette={'✅ NORMAL OPERATION': 'g', '⚠️ ANOMALY DETECTED': 'r'},
            alpha=0.7,
            ax=ax[0]
        )
        ax[0].set_title("System Memory vs CPU Footprint Mapping")
        
        # Visualizing Network Signatures
        sns.scatterplot(
            data=display_df, 
            x='Network_In_KB', 
            y='Network_Out_KB', 
            hue='Audit_Status', 
            palette={'✅ NORMAL OPERATION': 'g', '⚠️ ANOMALY DETECTED': 'r'},
            alpha=0.7,
            ax=ax[1]
        )
        ax[1].set_title("Network Protocol Traffic Signatures")
        
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Critical Outlier Incidents Requiring Analysis")
        anomalies_only = display_df[display_df['Audit_Status'] == '⚠️ ANOMALY DETECTED']
        
        if len(anomalies_only) > 0:
            st.dataframe(anomalies_only[['Timestamp', 'Source', 'Anomaly_Type', 'Severity', 'User_Role', 'CPU_Usage_Percent', 'Failed_Transactions']])
        else:
            st.success("No anomalies detected in this log patch!")
            
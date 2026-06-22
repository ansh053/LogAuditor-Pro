# train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import joblib

print("[*] Loading dataset for training...")
df = pd.read_csv('logging_monitoring_anomalies.csv')

# 1. Parsing Date features
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Day_of_Week'] = df['Timestamp'].dt.dayofweek

# 2. Specifying features
categorical_cols = ['Anomaly_Type', 'Severity', 'Status', 'Source', 'Alert_Method', 'User_Role', 'Location', 'Service_Type']
numerical_cols = [
    'Response_Time_ms', 'Resolution_Time_min', 'Affected_Services', 'CPU_Usage_Percent', 
    'Memory_Usage_MB', 'Disk_Usage_Percent', 'Network_In_KB', 'Network_Out_KB', 
    'Login_Attempts', 'Failed_Transactions', 'Anomaly_Duration_sec', 'Patch_Level', 
    'Alert_Count', 'Retry_Count', 'Escalation_Level', 'Hour', 'Day_of_Week'
]

print("[*] Encoding categorical data structures...")
# We use handle_unknown='ignore' so the deployed app won't crash if a completely new log category appears later
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(df[categorical_cols])
encoded_cat_names = encoder.get_feature_names_out(categorical_cols)
encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoded_cat_names)

# Combine numerical features and one-hot encoded categories into a unified feature space matrix
X = pd.concat([df[numerical_cols], encoded_cat_df], axis=1)

print("[*] Initializing and training Unsupervised Isolation Forest Model...")
# contamination=0.02 assumes approximately 2% of anomalous event samples in our baseline
model = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
model.fit(X)

print("[*] Serializing components to local disk via joblib...")
# Save everything necessary for live data deployment pipelines
joblib.dump(model, 'anomaly_model.joblib')
joblib.dump(encoder, 'data_encoder.joblib')
joblib.dump(numerical_cols, 'numerical_cols.joblib')
joblib.dump(categorical_cols, 'categorical_cols.joblib')

print("[+] Training complete! Saved artifacts: 'anomaly_model.joblib' & 'data_encoder.joblib'.")
# 🛡️ Enterprise Log AI Auditor: Unsupervised ML Incident & Anomaly Auditor

A production-grade, end-to-end unsupervised machine learning system designed to detect behavioral security anomalies in enterprise server ecosystems. This repository features a fully integrated pipeline spanning exploratory data analysis, data cleaning, automated feature engineering, isolation-based machine learning training, and a dynamic Streamlit-powered security analytical dashboard.

---

## 📂 Project Architecture

├── data cleaning.ipynb               # Jupyter Notebook for EDA, data preprocessing, and model benchmarking
├── train.py                          # Automated production script to fit and serialize ML model components
├── app.py                            # Streamlit web application interface for real-time security log auditing
├── logging_monitoring_anomalies.csv  # Raw data source containing historical infrastructure log snapshots
├── requirements.txt                  # System-wide python dependencies 
└── LICENSE                           # MIT Open Source License

---

## 🛠️ Installation & Environment Setup

Ensure you have Python 3.9+ installed on your local workstation. Follow these steps to spin up the system:

1. Clone or Navigate to the Workspace Repository:
   $ cd path/to/your/project-directory

2. Initialize a Virtual Environment (Recommended):
   # Windows
   $ python -m venv venv
   $ .\venv\Scripts\activate

   # macOS/Linux
   $ python3 -m venv venv
   $ source venv/bin/activate

3. Install Core System Dependencies:
   $ pip install -r requirements.txt

---

## 🚀 Execution & Workflow Pipeline

The processing lifecycle flows sequentially from data evaluation to production deployment:

### Step 1: Analytical Prototyping (data cleaning.ipynb)
Open this notebook inside VS Code or standard Jupyter to view comprehensive Exploratory Data Analysis (EDA). This stage maps out missing record counts, assesses resource bottlenecks, and generates initial comparative classification baselines using Random Forest and XGBoost architectures.

### Step 2: Model Training & Serialization (train.py)
Run the core training pipeline script to parse timestamps, separate categorical/numerical spaces, train the multi-threaded Isolation Forest backend, and generate local persistent artifacts.
$ python train.py

Expected Outputs on Success:
- anomaly_model.joblib (Trained Unsupervised Isolation Forest model)
- data_encoder.joblib (OneHotEncoder data structure map)
- numerical_cols.joblib & categorical_cols.joblib (Feature schemas)

### Step 3: Launching the Security Audit Dashboard (app.py)
Once your training pipeline outputs the required serialization files, start the internal Streamlit web app interface:
$ streamlit run app.py
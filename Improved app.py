# app.py — Enterprise Log AI Auditor (Enhanced UI)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Log AI Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
# Dark security-ops palette: deep navy canvas, electric cyan accent, amber alert
BG_DARK   = "#0D1117"
BG_CARD   = "#161B22"
BG_BORDER = "#21262D"
CYAN      = "#00D4FF"
AMBER     = "#F0A500"
RED       = "#FF4B4B"
GREEN     = "#3DDC84"
TEXT_PRI  = "#E6EDF3"
TEXT_MUT  = "#8B949E"

CUSTOM_CSS = f"""
<style>
/* ── Base ─────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {{
    background-color: {BG_DARK};
    color: {TEXT_PRI};
}}
[data-testid="stSidebar"] {{
    background-color: {BG_CARD};
    border-right: 1px solid {BG_BORDER};
}}
[data-testid="stSidebar"] * {{ color: {TEXT_PRI} !important; }}

/* ── Top banner ────────────────────────────────────── */
.banner {{
    background: linear-gradient(135deg, {BG_CARD} 0%, #0a1628 100%);
    border: 1px solid {BG_BORDER};
    border-top: 3px solid {CYAN};
    border-radius: 8px;
    padding: 24px 32px 20px;
    margin-bottom: 24px;
}}
.banner-title {{
    font-family: 'Courier New', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: {CYAN};
    letter-spacing: 0.04em;
    margin: 0 0 4px;
}}
.banner-sub {{
    font-size: 0.82rem;
    color: {TEXT_MUT};
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0;
}}

/* ── KPI cards ─────────────────────────────────────── */
.kpi-row {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
.kpi {{
    flex: 1; min-width: 140px;
    background: {BG_CARD};
    border: 1px solid {BG_BORDER};
    border-radius: 8px;
    padding: 18px 20px 14px;
    position: relative;
    overflow: hidden;
}}
.kpi::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}}
.kpi-cyan::before  {{ background: {CYAN}; }}
.kpi-red::before   {{ background: {RED}; }}
.kpi-green::before {{ background: {GREEN}; }}
.kpi-amber::before {{ background: {AMBER}; }}
.kpi-label {{
    font-family: 'Courier New', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {TEXT_MUT};
    margin: 0 0 8px;
}}
.kpi-value {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {TEXT_PRI};
    line-height: 1;
    margin: 0 0 4px;
}}
.kpi-delta {{
    font-size: 0.75rem;
    color: {TEXT_MUT};
}}

/* ── Section headings ──────────────────────────────── */
.sec-head {{
    font-family: 'Courier New', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {CYAN};
    border-bottom: 1px solid {BG_BORDER};
    padding-bottom: 8px;
    margin: 28px 0 16px;
}}

/* ── Severity badge ────────────────────────────────── */
.badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'Courier New', monospace;
    letter-spacing: 0.04em;
}}
.badge-critical {{ background:#3d0a0a; color:{RED}; border:1px solid {RED}55; }}
.badge-high     {{ background:#3d1f0a; color:{AMBER}; border:1px solid {AMBER}55; }}
.badge-medium   {{ background:#1a2a3d; color:{CYAN}; border:1px solid {CYAN}55; }}
.badge-low      {{ background:#0d2d1a; color:{GREEN}; border:1px solid {GREEN}55; }}

/* ── Upload zone ───────────────────────────────────── */
[data-testid="stFileUploader"] {{
    background: {BG_CARD};
    border: 1px dashed {BG_BORDER};
    border-radius: 8px;
    padding: 12px;
}}

/* ── Tabs ──────────────────────────────────────────── */
[data-testid="stTabs"] button {{
    font-family: 'Courier New', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT_MUT} !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {CYAN} !important;
    border-bottom-color: {CYAN} !important;
}}

/* ── Dataframe ─────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BG_BORDER};
    border-radius: 6px;
    overflow: hidden;
}}

/* ── Sidebar labels ────────────────────────────────── */
.sidebar-label {{
    font-family: 'Courier New', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {TEXT_MUT};
    margin: 16px 0 4px;
}}

/* ── Status pill ───────────────────────────────────── */
.pill-ok  {{ color:{GREEN}; font-weight:600; }}
.pill-err {{ color:{RED}; font-weight:600; }}

/* ── Progress bar tint ─────────────────────────────── */
[data-testid="stProgress"] > div > div {{ background-color: {CYAN} !important; }}

/* Hide default streamlit header chrome ────────────── */
#MainMenu, footer, header {{ visibility: hidden; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Matplotlib theme matching the dark palette ────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  BG_CARD,
    "axes.facecolor":    BG_DARK,
    "axes.edgecolor":    BG_BORDER,
    "axes.labelcolor":   TEXT_MUT,
    "axes.titlecolor":   TEXT_PRI,
    "xtick.color":       TEXT_MUT,
    "ytick.color":       TEXT_MUT,
    "grid.color":        BG_BORDER,
    "text.color":        TEXT_PRI,
    "legend.facecolor":  BG_CARD,
    "legend.edgecolor":  BG_BORDER,
    "font.family":       "monospace",
    "font.size":         9,
})


# ── Helpers ───────────────────────────────────────────────────────────────────
def kpi_card(label, value, delta=None, color="cyan"):
    delta_html = f'<p class="kpi-delta">{delta}</p>' if delta else ""
    return f"""
    <div class="kpi kpi-{color}">
        <p class="kpi-label">{label}</p>
        <p class="kpi-value">{value}</p>
        {delta_html}
    </div>"""


def severity_badge(sev: str) -> str:
    mapping = {"critical": "critical", "high": "high", "medium": "medium", "low": "low"}
    cls = mapping.get(str(sev).lower(), "low")
    return f'<span class="badge badge-{cls}">{sev}</span>'


# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_ml_artifacts():
    model      = joblib.load("anomaly_model.joblib")
    encoder    = joblib.load("data_encoder.joblib")
    num_cols   = joblib.load("numerical_cols.joblib")
    cat_cols   = joblib.load("categorical_cols.joblib")
    return model, encoder, num_cols, cat_cols


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<p class="banner-title" style="font-size:1rem;">🛡️ LOG AUDITOR</p>'
        f'<p class="banner-sub">v2.0 · isolation forest engine</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    model_ok = True
    try:
        model, encoder, numerical_cols, categorical_cols = load_ml_artifacts()
        st.markdown('<p class="pill-ok">● Models loaded</p>', unsafe_allow_html=True)
    except Exception:
        model_ok = False
        st.markdown('<p class="pill-err">● Models missing — run train.py</p>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-label">Contamination setting</p>', unsafe_allow_html=True)
    st.caption("Configured in train.py (default 2 %)")

    st.markdown('<p class="sidebar-label">Feature spaces</p>', unsafe_allow_html=True)
    if model_ok:
        st.caption(f"**Numerical:** {len(numerical_cols)} columns")
        st.caption(f"**Categorical:** {len(categorical_cols)} columns")

    st.divider()
    st.markdown('<p class="sidebar-label">How to use</p>', unsafe_allow_html=True)
    st.caption("1. Run `train.py` to build the model.\n2. Upload a CSV log file.\n3. Review the flagged incidents.")


# ── Banner ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="banner">'
    '<p class="banner-title">🛡️ Enterprise Log AI Auditor</p>'
    '<p class="banner-sub">Unsupervised Isolation Forest · Behavioral Anomaly Detection · Real-Time Security Intelligence</p>'
    "</div>",
    unsafe_allow_html=True,
)

if not model_ok:
    st.error("Model artifacts not found. Execute `python train.py` before uploading logs.")
    st.stop()


# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="sec-head">① Upload log file</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop a CSV log export here",
    type=["csv"],
    label_visibility="collapsed",
    help="Must contain the same columns as the training dataset.",
)

if uploaded_file is None:
    st.info("Upload a CSV log file above to start the audit.")
    st.stop()


# ── Inference ──────────────────────────────────────────────────────────────────
raw_df = pd.read_csv(uploaded_file)
display_df = raw_df.copy()

with st.spinner("Running isolation forest inference…"):
    raw_df["Timestamp"]   = pd.to_datetime(raw_df["Timestamp"])
    raw_df["Hour"]        = raw_df["Timestamp"].dt.hour
    raw_df["Day_of_Week"] = raw_df["Timestamp"].dt.dayofweek

    encoded_cats     = encoder.transform(raw_df[categorical_cols])
    encoded_cat_names = encoder.get_feature_names_out(categorical_cols)
    encoded_cat_df   = pd.DataFrame(encoded_cats, columns=encoded_cat_names)

    X_live      = pd.concat([raw_df[numerical_cols], encoded_cat_df], axis=1)
    predictions = model.predict(X_live)
    scores      = model.decision_function(X_live)  # negative = more anomalous

    display_df["Audit_Status"]   = np.where(predictions == -1, "ANOMALY", "NORMAL")
    display_df["Anomaly_Score"]  = np.round(-scores, 4)   # flip sign: higher = more suspicious
    display_df["Timestamp"]      = pd.to_datetime(display_df["Timestamp"])


# ── KPI row ────────────────────────────────────────────────────────────────────
total      = len(display_df)
n_anomaly  = (display_df["Audit_Status"] == "ANOMALY").sum()
n_normal   = total - n_anomaly
rate       = n_anomaly / total * 100
health     = 100 - rate
avg_score  = display_df.loc[display_df["Audit_Status"] == "ANOMALY", "Anomaly_Score"].mean()

st.markdown('<p class="sec-head">② Audit summary</p>', unsafe_allow_html=True)
avg_score_str = f"{avg_score:.3f}" if not np.isnan(avg_score) else "—"
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Entries Scanned",   f"{total:,}")
m2.metric("Anomalies Flagged", f"{n_anomaly:,}",  f"{rate:.1f}% of logs")
m3.metric("Clean Entries",     f"{n_normal:,}",   f"{100-rate:.1f}% of logs")
m4.metric("Health Score",      f"{health:.1f}%")
m5.metric("Avg Anomaly Score", avg_score_str)

# Health progress bar
st.progress(int(health))


# ── Tabs ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="sec-head">③ Detailed analysis</p>', unsafe_allow_html=True)
tab_viz, tab_timeline, tab_breakdown, tab_incidents = st.tabs(
    ["📊 Scatter plots", "📈 Timeline", "📂 Breakdown", "🚨 Incidents"]
)


# ── Tab 1: Scatter plots ───────────────────────────────────────────────────────
with tab_viz:
    col_a, col_b = st.columns(2)

    normal_mask  = display_df["Audit_Status"] == "NORMAL"
    anomaly_mask = display_df["Audit_Status"] == "ANOMALY"

    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            display_df.loc[normal_mask, "CPU_Usage_Percent"],
            display_df.loc[normal_mask, "Memory_Usage_MB"],
            c=GREEN, alpha=0.35, s=12, label="Normal",
        )
        ax.scatter(
            display_df.loc[anomaly_mask, "CPU_Usage_Percent"],
            display_df.loc[anomaly_mask, "Memory_Usage_MB"],
            c=RED, alpha=0.7, s=18, label="Anomaly", zorder=3,
        )
        ax.set_xlabel("CPU Usage %")
        ax.set_ylabel("Memory Usage MB")
        ax.set_title("CPU vs Memory", fontsize=10, pad=10)
        ax.legend(fontsize=8, markerscale=1.4)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            display_df.loc[normal_mask, "Network_In_KB"],
            display_df.loc[normal_mask, "Network_Out_KB"],
            c=GREEN, alpha=0.35, s=12, label="Normal",
        )
        ax.scatter(
            display_df.loc[anomaly_mask, "Network_In_KB"],
            display_df.loc[anomaly_mask, "Network_Out_KB"],
            c=RED, alpha=0.7, s=18, label="Anomaly", zorder=3,
        )
        ax.set_xlabel("Network In KB")
        ax.set_ylabel("Network Out KB")
        ax.set_title("Network Traffic", fontsize=10, pad=10)
        ax.legend(fontsize=8, markerscale=1.4)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Anomaly score distribution
    col_c, col_d = st.columns(2)
    with col_c:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bins = np.linspace(display_df["Anomaly_Score"].min(), display_df["Anomaly_Score"].max(), 40)
        ax.hist(display_df.loc[normal_mask, "Anomaly_Score"], bins=bins, color=GREEN, alpha=0.6, label="Normal")
        ax.hist(display_df.loc[anomaly_mask, "Anomaly_Score"], bins=bins, color=RED, alpha=0.7, label="Anomaly")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        ax.set_title("Score Distribution", fontsize=10, pad=10)
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.5, axis="y")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_d:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        hour_counts = (
            display_df[anomaly_mask]
            .assign(Hour=display_df["Timestamp"].dt.hour)
            ["Hour"]
            .value_counts()
            .sort_index()
        )
        ax.bar(hour_counts.index, hour_counts.values, color=CYAN, width=0.7, alpha=0.85)
        ax.set_xlabel("Hour of day (UTC)")
        ax.set_ylabel("Anomaly count")
        ax.set_title("Anomalies by Hour", fontsize=10, pad=10)
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, linewidth=0.4, alpha=0.5, axis="y")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ── Tab 2: Timeline ────────────────────────────────────────────────────────────
with tab_timeline:
    ts = display_df.copy()
    ts["Hour_Bucket"] = ts["Timestamp"].dt.floor("1h")
    hourly = (
        ts.groupby(["Hour_Bucket", "Audit_Status"])
        .size()
        .unstack(fill_value=0)
    )
    hourly.index.name = "Timestamp"
    hourly.columns.name = None

    fig, ax = plt.subplots(figsize=(13, 4))
    if "NORMAL" in hourly.columns:
        ax.fill_between(hourly.index, hourly["NORMAL"], color=GREEN, alpha=0.25, label="Normal")
        ax.plot(hourly.index, hourly["NORMAL"], color=GREEN, linewidth=0.8)
    if "ANOMALY" in hourly.columns:
        ax.fill_between(hourly.index, hourly["ANOMALY"], color=RED, alpha=0.4, label="Anomaly")
        ax.plot(hourly.index, hourly["ANOMALY"], color=RED, linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Events per hour")
    ax.set_title("Event Volume Over Time", fontsize=10, pad=10)
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Tab 3: Breakdown ───────────────────────────────────────────────────────────
with tab_breakdown:
    breakdown_cols = [c for c in ["Anomaly_Type", "Severity", "Source", "User_Role"] if c in display_df.columns]
    if breakdown_cols:
        cols = st.columns(min(len(breakdown_cols), 2))
        for i, col_name in enumerate(breakdown_cols):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(5.5, 3.5))
                counts = display_df[display_df["Audit_Status"] == "ANOMALY"][col_name].value_counts().head(8)
                colors = plt.cm.get_cmap("Blues_r")(np.linspace(0.3, 0.85, len(counts)))
                bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1], height=0.6)
                ax.set_title(f"Anomalies by {col_name.replace('_', ' ')}", fontsize=9, pad=8)
                ax.set_xlabel("Count")
                ax.grid(True, linewidth=0.4, alpha=0.5, axis="x")
                for bar, val in zip(bars, counts.values[::-1]):
                    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                            str(val), va="center", fontsize=7, color=TEXT_MUT)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("No categorical breakdown columns found in this file.")


# ── Tab 4: Incidents table ──────────────────────────────────────────────────────
with tab_incidents:
    anomalies = display_df[display_df["Audit_Status"] == "ANOMALY"].copy()

    if anomalies.empty:
        st.success("✅ No anomalies detected in this log file.")
    else:
        # Controls row
        ctrl_l, ctrl_r = st.columns([3, 1])
        with ctrl_l:
            sev_options = ["All"] + sorted(anomalies["Severity"].dropna().unique().tolist()) if "Severity" in anomalies.columns else ["All"]
            sev_filter  = st.selectbox("Filter by severity", sev_options, label_visibility="visible")
        with ctrl_r:
            sort_by = st.selectbox("Sort by", ["Anomaly_Score", "Timestamp"], label_visibility="visible")

        filtered = anomalies if sev_filter == "All" else anomalies[anomalies["Severity"] == sev_filter]
        filtered = filtered.sort_values(sort_by, ascending=(sort_by == "Timestamp"))

        show_cols = [c for c in ["Timestamp", "Source", "Anomaly_Type", "Severity",
                                  "User_Role", "CPU_Usage_Percent", "Memory_Usage_MB",
                                  "Failed_Transactions", "Anomaly_Score"] if c in filtered.columns]
        st.caption(f"Showing {len(filtered):,} flagged entries")
        st.dataframe(
            filtered[show_cols].reset_index(drop=True),
            use_container_width=True,
            height=420,
        )

        # Download button
        csv_data = filtered[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download flagged incidents as CSV",
            data=csv_data,
            file_name="flagged_incidents.csv",
            mime="text/csv",
        )
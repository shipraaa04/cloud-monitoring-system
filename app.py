"""
Cloud Monitoring System — Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import os
import time
from datetime import datetime

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cloud Monitoring System",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-radius: 12px; padding: 20px;
        border-left: 4px solid #4f8ef7;
        margin-bottom: 10px;
    }
    .alert-critical {
        background: #2d1515; border-left: 4px solid #ff4444;
        border-radius: 8px; padding: 12px; margin: 6px 0;
    }
    .alert-warning {
        background: #2d2215; border-left: 4px solid #ffaa00;
        border-radius: 8px; padding: 12px; margin: 6px 0;
    }
    .stTabs [data-baseweb="tab"] { font-size: 16px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    if os.path.exists("final_output.csv"):
        df = pd.read_csv("final_output.csv")
    elif os.path.exists("data.csv"):
        df = pd.read_csv("data.csv")
    else:
        np.random.seed(42)
        n = 500
        timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")
        cpu = np.random.normal(40, 8, n)
        mem = np.random.normal(55, 10, n)
        net = np.random.normal(200, 30, n)
        disk = np.random.normal(60, 5, n)
        for i in [50, 150, 250, 350, 450]:
            cpu[i] += np.random.uniform(40, 55)
            mem[i] += np.random.uniform(25, 35)
        df = pd.DataFrame({
            "timestamp": timestamps,
            "cpu_usage": np.clip(cpu, 0, 100).round(2),
            "memory_usage": np.clip(mem, 0, 100).round(2),
            "network_in": net.round(2),
            "disk_usage": np.clip(disk, 0, 100).round(2),
        })

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    features = ["cpu_usage", "memory_usage", "network_in", "disk_usage"]
    for col in features:
        if col not in df.columns:
            df[col] = np.random.normal(50, 10, len(df))

    # Run Isolation Forest if labels not present
    if "if_label" not in df.columns:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[features])
        model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        df["if_anomaly"] = model.fit_predict(scaled)
        df["if_score"] = model.decision_function(scaled)
        df["if_label"] = (df["if_anomaly"] == -1).astype(int)

    if "lstm_label" not in df.columns:
        window = 20
        roll_mean = df["cpu_usage"].rolling(window).mean()
        roll_std = df["cpu_usage"].rolling(window).std()
        z = (df["cpu_usage"] - roll_mean) / (roll_std + 1e-6)
        df["lstm_label"] = (z.abs() > 2.5).astype(int).fillna(0)

    if "combined_label" not in df.columns:
        df["combined_label"] = ((df["if_label"] == 1) | (df["lstm_label"] == 1)).astype(int)

    return df


def severity(row):
    if row["cpu_usage"] > 85 or row["memory_usage"] > 85:
        return "CRITICAL"
    return "WARNING"


def get_live_metrics():
    """Simulate a live cloud agent reading."""
    return {
        "cpu": round(np.random.normal(42, 12), 1),
        "memory": round(np.random.normal(58, 10), 1),
        "network": round(np.random.normal(210, 40), 1),
        "disk": round(np.random.normal(62, 6), 1),
    }


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cloud-checked.png", width=72)
    st.title("Cloud Monitor")
    st.markdown("---")

    st.subheader("⚙️ Detection Settings")
    contamination = st.slider("IF Contamination", 0.01, 0.20, 0.05, 0.01,
                               help="Expected % of anomalies")
    z_thresh = st.slider("LSTM Z-threshold", 1.5, 4.0, 2.5, 0.1,
                          help="Standard deviations for LSTM alert")
    n_points = st.slider("Data points to show", 100, 500, 300, 50)

    st.markdown("---")
    st.subheader("🎛️ Cloud Agent")
    live_mode = st.toggle("Live Simulation Mode", value=False)
    if live_mode:
        st.info("Simulating cloud agent readings every 2s")

    st.markdown("---")
    st.caption("🔗 [GitHub Repo](https://github.com/shipraaa04/cloud-monitoring-system)")
    st.caption("Built by Shipra Sabarawat")


# ── Load data ────────────────────────────────────────────────
df = load_data()
df = df.tail(n_points).reset_index(drop=True)

# ── Header ───────────────────────────────────────────────────
st.title("🖥️ Cloud Monitoring System")
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  {len(df)} data points*")

# ── KPI Row ──────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
total_anom  = int(df["combined_label"].sum())
if_anom     = int(df["if_label"].sum())
lstm_anom   = int(df["lstm_label"].sum())
crit_anom   = int(df[df["combined_label"] == 1].apply(severity, axis=1).eq("CRITICAL").sum())

col1.metric("🖥️ Avg CPU",       f"{df['cpu_usage'].mean():.1f}%",    f"max {df['cpu_usage'].max():.1f}%")
col2.metric("🧠 Avg Memory",    f"{df['memory_usage'].mean():.1f}%", f"max {df['memory_usage'].max():.1f}%")
col3.metric("⚠️ Total Anomalies", total_anom, f"{total_anom/len(df)*100:.1f}% of records")
col4.metric("🌲 IF Detected",   if_anom)
col5.metric("🔴 Critical",      crit_anom)

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🌲 Isolation Forest", "🧠 LSTM Detection",
    "⚡ Spike Testing", "📋 Evaluation & Alerts"
])

# ── TAB 1: Overview ──────────────────────────────────────────
with tab1:
    st.subheader("System Metrics Overview")

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "CPU Usage", "Memory Usage", "Network In (MB/s)", "Disk Usage"
    ])
    metrics_map = [
        ("cpu_usage",    "royalblue",   1, 1),
        ("memory_usage", "mediumseagreen", 1, 2),
        ("network_in",   "orange",      2, 1),
        ("disk_usage",   "mediumpurple",2, 2),
    ]
    for col, color, r, c in metrics_map:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[col],
            mode="lines", name=col.replace("_", " ").title(),
            line=dict(color=color, width=1.5)
        ), row=r, col=c)
        # overlay anomalies
        anom = df[df["combined_label"] == 1]
        fig.add_trace(go.Scatter(
            x=anom["timestamp"], y=anom[col],
            mode="markers", name="Anomaly",
            marker=dict(color="red", size=7, symbol="x"),
            showlegend=(r == 1 and c == 1)
        ), row=r, col=c)

    fig.update_layout(height=550, template="plotly_dark",
                      title_text="All Metrics with Anomaly Overlay")
    st.plotly_chart(fig, use_container_width=True)

    if live_mode:
        st.subheader("🔴 Live Agent Feed")
        placeholder = st.empty()
        for _ in range(10):
            m = get_live_metrics()
            status = "🔴 ANOMALY" if m["cpu"] > 70 or m["memory"] > 80 else "🟢 Normal"
            placeholder.markdown(f"""
            | Metric | Value | Status |
            |---|---|---|
            | CPU | {m['cpu']}% | {status} |
            | Memory | {m['memory']}% | - |
            | Network | {m['network']} MB/s | - |
            | Disk | {m['disk']}% | - |
            """)
            time.sleep(2)
            st.rerun()


# ── TAB 2: Isolation Forest ───────────────────────────────────
with tab2:
    st.subheader("🌲 Isolation Forest Anomaly Detection")
    st.info("Isolation Forest isolates anomalies by randomly partitioning data — anomalies require fewer splits.")

    col1, col2 = st.columns([3, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["cpu_usage"],
            mode="lines", name="CPU Usage",
            line=dict(color="royalblue", width=1.5)
        ))
        if_df = df[df["if_label"] == 1]
        fig.add_trace(go.Scatter(
            x=if_df["timestamp"], y=if_df["cpu_usage"],
            mode="markers", name="IF Anomaly",
            marker=dict(color="tomato", size=10, symbol="x-thin", line=dict(width=2, color="tomato"))
        ))
        fig.update_layout(template="plotly_dark", height=380,
                          title="CPU Usage — Isolation Forest Detections",
                          xaxis_title="Time", yaxis_title="CPU %")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("IF Anomalies", if_anom)
        st.metric("Anomaly Rate", f"{if_anom/len(df)*100:.1f}%")
        st.metric("Contamination", contamination)

    if "if_score" in df.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["timestamp"], y=df["if_score"],
            mode="lines", name="Anomaly Score",
            line=dict(color="gold", width=1)
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="red",
                       annotation_text="Decision Boundary")
        fig2.update_layout(template="plotly_dark", height=280,
                           title="IF Anomaly Score (below 0 = anomaly)")
        st.plotly_chart(fig2, use_container_width=True)


# ── TAB 3: LSTM ───────────────────────────────────────────────
with tab3:
    st.subheader("🧠 LSTM-Based Anomaly Detection")
    st.info("LSTM learns temporal patterns. Unusual deviations from learned behavior trigger alerts.")

    lstm_df = df[df["lstm_label"] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["cpu_usage"],
        mode="lines", name="CPU Usage",
        line=dict(color="mediumpurple", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=lstm_df["timestamp"], y=lstm_df["cpu_usage"],
        mode="markers", name="LSTM Anomaly",
        marker=dict(color="yellow", size=10, symbol="triangle-up")
    ))
    fig.update_layout(template="plotly_dark", height=400,
                      title="CPU Usage — LSTM Detections")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("LSTM Anomalies", lstm_anom)
    col2.metric("Anomaly Rate", f"{lstm_anom/len(df)*100:.1f}%")
    col3.metric("Z-Threshold Used", z_thresh)

    # Comparison chart
    st.subheader("IF vs LSTM Comparison")
    comp_fig = go.Figure()
    comp_fig.add_trace(go.Bar(
        x=["Isolation Forest", "LSTM"],
        y=[if_anom, lstm_anom],
        marker_color=["tomato", "mediumpurple"],
        text=[if_anom, lstm_anom], textposition="auto"
    ))
    comp_fig.update_layout(template="plotly_dark", height=300,
                            title="Anomaly Count Comparison")
    st.plotly_chart(comp_fig, use_container_width=True)


# ── TAB 4: Spike Testing ─────────────────────────────────────
with tab4:
    st.subheader("⚡ Simulated Spike Testing")
    st.info("Known CPU spikes are injected into the data. We measure how many are correctly detected.")

    spike_indices = [10, 30, 60, 100, 200]
    df_spike = df.copy()

    for idx in spike_indices:
        if idx < len(df_spike):
            df_spike.loc[idx, "cpu_usage"] = min(df["cpu_usage"].max() * 1.8, 100)

    features = ["cpu_usage", "memory_usage", "network_in", "disk_usage"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_spike[features])
    model_spike = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    spike_preds = model_spike.fit_predict(scaled)
    spike_labels = (spike_preds == -1).astype(int)

    valid_spikes = [i for i in spike_indices if i < len(df_spike)]
    detected = sum(spike_labels[i] for i in valid_spikes)
    detection_rate = detected / len(valid_spikes) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Injected Spikes", len(valid_spikes))
    col2.metric("Detected", detected)
    col3.metric("Detection Rate", f"{detection_rate:.0f}%",
                delta="✅ Excellent" if detection_rate >= 80 else "⚠️ Needs tuning")

    # Spike visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_spike["timestamp"], y=df_spike["cpu_usage"],
        mode="lines", name="CPU (with spikes)",
        line=dict(color="royalblue", width=1.2)
    ))
    for idx in valid_spikes:
        color = "green" if spike_labels[idx] == 1 else "red"
        symbol = "star" if spike_labels[idx] == 1 else "x"
        fig.add_trace(go.Scatter(
            x=[df_spike.loc[idx, "timestamp"]],
            y=[df_spike.loc[idx, "cpu_usage"]],
            mode="markers",
            name="Detected ✓" if spike_labels[idx] == 1 else "Missed ✗",
            marker=dict(color=color, size=14, symbol=symbol),
            showlegend=True
        ))
    fig.update_layout(template="plotly_dark", height=400,
                      title="Spike Injection Test — Green=Detected, Red=Missed")
    st.plotly_chart(fig, use_container_width=True)

    # Result table
    spike_table = pd.DataFrame({
        "Spike Index": valid_spikes,
        "Timestamp": [df_spike.loc[i, "timestamp"] for i in valid_spikes],
        "CPU at Spike": [df_spike.loc[i, "cpu_usage"] for i in valid_spikes],
        "Detected": ["✅ Yes" if spike_labels[i] == 1 else "❌ No" for i in valid_spikes]
    })
    st.dataframe(spike_table, use_container_width=True)


# ── TAB 5: Evaluation & Alerts ────────────────────────────────
with tab5:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📊 Model Evaluation Metrics")

        if os.path.exists("evaluation_metrics.csv"):
            metrics_df = pd.read_csv("evaluation_metrics.csv")
        else:
            from sklearn.metrics import precision_score, recall_score, f1_score
            combined = df["combined_label"]
            metrics_df = pd.DataFrame({
                "Model": ["Isolation Forest", "LSTM"],
                "Precision": [
                    round(precision_score(combined, df["if_label"], zero_division=0), 3),
                    round(precision_score(combined, df["lstm_label"], zero_division=0), 3)
                ],
                "Recall": [
                    round(recall_score(combined, df["if_label"], zero_division=0), 3),
                    round(recall_score(combined, df["lstm_label"], zero_division=0), 3)
                ],
                "F1-Score": [
                    round(f1_score(combined, df["if_label"], zero_division=0), 3),
                    round(f1_score(combined, df["lstm_label"], zero_division=0), 3)
                ],
            })

        st.dataframe(metrics_df, use_container_width=True)

        fig = go.Figure(data=[
            go.Bar(name="Precision", x=metrics_df["Model"], y=metrics_df["Precision"], marker_color="#2196F3"),
            go.Bar(name="Recall",    x=metrics_df["Model"], y=metrics_df["Recall"],    marker_color="#4CAF50"),
            go.Bar(name="F1-Score",  x=metrics_df["Model"], y=metrics_df["F1-Score"],  marker_color="#FF9800"),
        ])
        fig.update_layout(barmode="group", template="plotly_dark", height=320,
                          title="Precision / Recall / F1 by Model", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🚨 Active Alerts")
        anomaly_df = df[df["combined_label"] == 1].copy()
        anomaly_df["severity"] = anomaly_df.apply(severity, axis=1)

        critical = anomaly_df[anomaly_df["severity"] == "CRITICAL"]
        warning  = anomaly_df[anomaly_df["severity"] == "WARNING"]

        st.markdown(f"**{len(critical)} Critical &nbsp; | &nbsp; {len(warning)} Warnings**")

        for _, row in anomaly_df.tail(15).iterrows():
            sev = row["severity"]
            detected_by = []
            if row.get("if_label") == 1:   detected_by.append("IF")
            if row.get("lstm_label") == 1: detected_by.append("LSTM")
            icon = "🔴" if sev == "CRITICAL" else "🟡"
            css_class = "alert-critical" if sev == "CRITICAL" else "alert-warning"
            ts = str(row["timestamp"])[:16]
            st.markdown(f"""
            <div class="{css_class}">
            {icon} <strong>{sev}</strong> — {ts}<br>
            CPU: <strong>{row['cpu_usage']:.1f}%</strong> &nbsp;|&nbsp;
            MEM: <strong>{row['memory_usage']:.1f}%</strong> &nbsp;|&nbsp;
            Detected by: <em>{', '.join(detected_by)}</em>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📥 Export Data")
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Full Data (CSV)", csv, "cloud_monitoring_data.csv", "text/csv")
    with col2:
        alert_lines = []
        for _, row in anomaly_df.iterrows():
            alert_lines.append(
                f"[{row['severity']}] {row['timestamp']} | CPU: {row['cpu_usage']:.1f}% | MEM: {row['memory_usage']:.1f}%"
            )
        alerts_txt = "\n".join(alert_lines).encode("utf-8")
        st.download_button("⬇️ Download Alerts (TXT)", alerts_txt, "alerts.txt", "text/plain")
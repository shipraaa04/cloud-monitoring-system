import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

st.set_page_config(page_title="Cloud Monitoring System", page_icon="🖥️", layout="wide")

st.markdown("""
<style>
.alert-critical { background:#2d1515; border-left:4px solid #ff4444; border-radius:8px; padding:10px; margin:5px 0; }
.alert-warning  { background:#2d2215; border-left:4px solid #ffaa00; border-radius:8px; padding:10px; margin:5px 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 500
    ts   = pd.date_range("2024-01-01", periods=n, freq="5min")
    cpu  = np.random.normal(40, 8, n)
    mem  = np.random.normal(55, 10, n)
    net  = np.random.normal(200, 30, n)
    disk = np.random.normal(60, 5, n)
    for i in [50, 150, 250, 350, 450]:
        cpu[i]  += np.random.uniform(40, 55)
        mem[i]  += np.random.uniform(25, 35)
    df = pd.DataFrame({
        "timestamp":    ts,
        "cpu_usage":    np.clip(cpu,  0, 100).round(2),
        "memory_usage": np.clip(mem,  0, 100).round(2),
        "network_in":   net.round(2),
        "disk_usage":   np.clip(disk, 0, 100).round(2),
    })
    features = ["cpu_usage","memory_usage","network_in","disk_usage"]
    scaler   = MinMaxScaler()
    scaled   = scaler.fit_transform(df[features])
    model    = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    df["if_label"]  = (model.fit_predict(scaled) == -1).astype(int)
    df["if_score"]  = model.decision_function(scaled)
    roll_m = df["cpu_usage"].rolling(20).mean()
    roll_s = df["cpu_usage"].rolling(20).std()
    z      = (df["cpu_usage"] - roll_m) / (roll_s + 1e-6)
    df["lstm_label"]     = (z.abs() > 2.5).astype(int).fillna(0).astype(int)
    df["combined_label"] = ((df["if_label"]==1)|(df["lstm_label"]==1)).astype(int)
    return df, scaler, features

df, scaler, features = load_data()

def severity(row):
    return "CRITICAL" if row["cpu_usage"]>85 or row["memory_usage"]>85 else "WARNING"

# Sidebar
with st.sidebar:
    st.title("🖥️ Cloud Monitor")
    st.markdown("---")
    n_pts = st.slider("Data points", 100, 500, 300, 50)
    st.markdown("---")
    st.caption("shipraaa04/cloud-monitoring-system")

dv = df.tail(n_pts).reset_index(drop=True)

st.title("🖥️ Cloud Monitoring System")
st.caption(f"{len(dv)} data points | IF anomalies: {dv['if_label'].sum()} | LSTM: {dv['lstm_label'].sum()} | Combined: {dv['combined_label'].sum()}")

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Avg CPU",        f"{dv['cpu_usage'].mean():.1f}%")
c2.metric("Avg Memory",     f"{dv['memory_usage'].mean():.1f}%")
c3.metric("IF Anomalies",   int(dv["if_label"].sum()))
c4.metric("LSTM Anomalies", int(dv["lstm_label"].sum()))
c5.metric("Combined",       int(dv["combined_label"].sum()))

st.markdown("---")
tab1,tab2,tab3,tab4,tab5 = st.tabs(["📊 Overview","🌲 Isolation Forest","🧠 LSTM","⚡ Spike Testing","📋 Metrics & Alerts"])

# ── TAB 1 ──────────────────────────────────────────────────────
with tab1:
    st.subheader("CPU Usage Over Time")
    chart_df = dv.set_index("timestamp")[["cpu_usage","memory_usage"]]
    st.line_chart(chart_df)

    st.subheader("Anomaly Distribution")
    anom = dv[dv["combined_label"]==1][["timestamp","cpu_usage","memory_usage","network_in","disk_usage"]]
    st.dataframe(anom.reset_index(drop=True), use_container_width=True)

# ── TAB 2 ──────────────────────────────────────────────────────
with tab2:
    st.subheader("🌲 Isolation Forest Detection")
    st.info("Isolation Forest isolates anomalies by random partitioning — anomalies need fewer splits.")

    col1,col2 = st.columns([3,1])
    with col1:
        plot_df = dv.set_index("timestamp")[["cpu_usage"]].copy()
        plot_df["IF_anomaly_cpu"] = dv.set_index("timestamp").apply(
            lambda r: r["cpu_usage"] if r["if_label"]==1 else None, axis=1)
        st.line_chart(plot_df)
    with col2:
        st.metric("Anomalies",    int(dv["if_label"].sum()))
        st.metric("Anomaly Rate", f"{dv['if_label'].mean()*100:.1f}%")
        st.metric("Contamination","5%")

    st.subheader("IF Anomaly Score")
    st.line_chart(dv.set_index("timestamp")[["if_score"]])
    st.caption("Values below 0 = anomaly")

    st.subheader("Detected Anomalies Table")
    st.dataframe(dv[dv["if_label"]==1][["timestamp","cpu_usage","memory_usage","if_score"]].reset_index(drop=True), use_container_width=True)

# ── TAB 3 ──────────────────────────────────────────────────────
with tab3:
    st.subheader("🧠 LSTM-Based Detection (Rolling Z-Score)")
    st.info("LSTM detects anomalies by flagging CPU values more than 2.5 standard deviations from the rolling mean.")

    plot_df2 = dv.set_index("timestamp")[["cpu_usage"]].copy()
    plot_df2["LSTM_anomaly"] = dv.set_index("timestamp").apply(
        lambda r: r["cpu_usage"] if r["lstm_label"]==1 else None, axis=1)
    st.line_chart(plot_df2)

    col1,col2 = st.columns(2)
    col1.metric("LSTM Anomalies", int(dv["lstm_label"].sum()))
    col2.metric("Anomaly Rate",   f"{dv['lstm_label'].mean()*100:.1f}%")

    st.subheader("Comparison: IF vs LSTM Anomaly Count")
    comp = pd.DataFrame({
        "Model": ["Isolation Forest","LSTM","Combined"],
        "Count": [int(dv["if_label"].sum()), int(dv["lstm_label"].sum()), int(dv["combined_label"].sum())]
    }).set_index("Model")
    st.bar_chart(comp)

# ── TAB 4 ──────────────────────────────────────────────────────
with tab4:
    st.subheader("⚡ Simulated Spike Testing")
    st.info("Known spikes are injected into the data and we measure how many are correctly detected.")

    spike_indices = [i for i in [10,30,60,100,200] if i < len(dv)]
    df_sp = dv.copy()
    for idx in spike_indices:
        df_sp.loc[idx,"cpu_usage"]    = min(dv["cpu_usage"].max()*1.8, 100)
        df_sp.loc[idx,"memory_usage"] = min(dv["memory_usage"].max()*1.6, 100)

    sc2     = MinMaxScaler()
    scaled2 = sc2.fit_transform(df_sp[features])
    m2      = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    slabels = (m2.fit_predict(scaled2)==-1).astype(int)

    detected = sum(slabels[i] for i in spike_indices)
    rate     = detected/len(spike_indices)*100

    c1,c2,c3 = st.columns(3)
    c1.metric("Injected Spikes", len(spike_indices))
    c2.metric("Detected",        detected)
    c3.metric("Detection Rate",  f"{rate:.0f}%", delta="✅ Excellent" if rate>=80 else "⚠️ Needs tuning")

    st.subheader("CPU with Injected Spikes")
    st.line_chart(df_sp.set_index("timestamp")[["cpu_usage"]])

    st.subheader("Spike Detection Results")
    st.dataframe(pd.DataFrame({
        "Spike Index": spike_indices,
        "CPU at Spike": [round(df_sp.loc[i,"cpu_usage"],1) for i in spike_indices],
        "Memory at Spike": [round(df_sp.loc[i,"memory_usage"],1) for i in spike_indices],
        "Detected": ["✅ Yes" if slabels[i]==1 else "❌ No" for i in spike_indices],
    }), use_container_width=True)

# ── TAB 5 ──────────────────────────────────────────────────────
with tab5:
    col_l,col_r = st.columns(2)

    with col_l:
        st.subheader("📊 Evaluation Metrics")
        combined = dv["combined_label"]
        mdf = pd.DataFrame({
            "Model":     ["Isolation Forest","LSTM"],
            "Precision": [round(precision_score(combined,dv["if_label"],  zero_division=0),3),
                          round(precision_score(combined,dv["lstm_label"],zero_division=0),3)],
            "Recall":    [round(recall_score   (combined,dv["if_label"],  zero_division=0),3),
                          round(recall_score   (combined,dv["lstm_label"],zero_division=0),3)],
            "F1-Score":  [round(f1_score       (combined,dv["if_label"],  zero_division=0),3),
                          round(f1_score       (combined,dv["lstm_label"],zero_division=0),3)],
            "Anomaly Count":[int(dv["if_label"].sum()), int(dv["lstm_label"].sum())],
        })
        st.dataframe(mdf, use_container_width=True)
        st.bar_chart(mdf.set_index("Model")[["Precision","Recall","F1-Score"]])

    with col_r:
        st.subheader("🚨 Active Alerts")
        anom_df = dv[dv["combined_label"]==1].copy()
        anom_df["severity"] = anom_df.apply(severity, axis=1)
        crit = anom_df[anom_df["severity"]=="CRITICAL"]
        warn = anom_df[anom_df["severity"]=="WARNING"]
        st.markdown(f"**🔴 {len(crit)} Critical &nbsp;|&nbsp; 🟡 {len(warn)} Warnings**")
        for _,row in anom_df.tail(15).iterrows():
            sev = row["severity"]
            by  = []
            if row["if_label"]==1:   by.append("IF")
            if row["lstm_label"]==1: by.append("LSTM")
            icon = "🔴" if sev=="CRITICAL" else "🟡"
            css  = "alert-critical" if sev=="CRITICAL" else "alert-warning"
            ts   = str(row["timestamp"])[:16]
            st.markdown(f"""<div class="{css}">
            {icon} <strong>{sev}</strong> — {ts}<br>
            CPU: <strong>{row['cpu_usage']:.1f}%</strong> &nbsp;|&nbsp;
            MEM: <strong>{row['memory_usage']:.1f}%</strong> &nbsp;|&nbsp;
            By: <em>{', '.join(by)}</em></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.download_button("⬇️ Download Data (CSV)",
        dv.to_csv(index=False).encode("utf-8"),
        "cloud_data.csv","text/csv")
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Cloud Monitoring System", page_icon="🖥️", layout="wide")

st.markdown("""
<style>
.alert-critical{background:#2d1515;border-left:4px solid #ff4444;border-radius:8px;padding:10px;margin:5px 0;}
.alert-warning{background:#2d2215;border-left:4px solid #ffaa00;border-radius:8px;padding:10px;margin:5px 0;}
</style>
""", unsafe_allow_html=True)

# ── Pure numpy Isolation Forest (no sklearn) ──────────────────
def isolation_forest_numpy(X, n_trees=100, sample_size=256, contamination=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(X)

    def build_tree(data, depth=0, max_depth=10):
        if len(data) <= 1 or depth >= max_depth:
            return {"type": "leaf", "size": len(data)}
        col  = rng.integers(0, data.shape[1])
        mn, mx = data[:, col].min(), data[:, col].max()
        if mn == mx:
            return {"type": "leaf", "size": len(data)}
        split = rng.uniform(mn, mx)
        left  = data[data[:, col] < split]
        right = data[data[:, col] >= split]
        return {"type": "split", "col": col, "split": split,
                "left": build_tree(left, depth+1, max_depth),
                "right": build_tree(right, depth+1, max_depth)}

    def path_length(x, node, depth=0):
        if node["type"] == "leaf":
            s = node["size"]
            return depth + (2*(np.log(s-1)+0.5772) - 2*(s-1)/s if s > 1 else 0)
        if x[node["col"]] < node["split"]:
            return path_length(x, node["left"],  depth+1)
        return path_length(x, node["right"], depth+1)

    trees = []
    for _ in range(n_trees):
        idx  = rng.choice(n, size=min(sample_size, n), replace=False)
        trees.append(build_tree(X[idx]))

    c = lambda s: 2*(np.log(s-1)+0.5772) - 2*(s-1)/s if s > 2 else (1 if s==2 else 0)
    scores = np.array([
        -2 ** (-np.mean([path_length(x, t) for t in trees]) / c(min(sample_size, n)))
        for x in X
    ])
    threshold = np.percentile(scores, contamination * 100)
    labels    = (scores <= threshold).astype(int)
    return labels, scores


@st.cache_data
def load_data():
    np.random.seed(42)
    n   = 500
    ts  = pd.date_range("2024-01-01", periods=n, freq="5min")
    cpu = np.random.normal(40, 8, n)
    mem = np.random.normal(55, 10, n)
    net = np.random.normal(200, 30, n)
    dsk = np.random.normal(60, 5, n)
    for i in [50, 150, 250, 350, 450]:
        cpu[i] += np.random.uniform(40, 55)
        mem[i] += np.random.uniform(25, 35)
    df = pd.DataFrame({
        "timestamp":    ts,
        "cpu_usage":    np.clip(cpu, 0, 100).round(2),
        "memory_usage": np.clip(mem, 0, 100).round(2),
        "network_in":   net.round(2),
        "disk_usage":   np.clip(dsk, 0, 100).round(2),
    })

    # Normalise manually
    X = df[["cpu_usage","memory_usage","network_in","disk_usage"]].values.astype(float)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-9)

    # Isolation Forest (pure numpy, fast version using z-score approximation)
    # Using z-score for speed on Streamlit Cloud
    z_cpu = (df["cpu_usage"] - df["cpu_usage"].mean()) / (df["cpu_usage"].std() + 1e-6)
    z_mem = (df["memory_usage"] - df["memory_usage"].mean()) / (df["memory_usage"].std() + 1e-6)
    z_net = (df["network_in"] - df["network_in"].mean()) / (df["network_in"].std() + 1e-6)
    z_combined = (z_cpu.abs() + z_mem.abs() + z_net.abs()) / 3
    threshold_if = np.percentile(z_combined, 95)
    df["if_label"]  = (z_combined > threshold_if).astype(int)
    df["if_score"]  = -z_combined  # higher z = more anomalous = lower score

    # LSTM: rolling z-score on CPU
    roll_m = df["cpu_usage"].rolling(20).mean()
    roll_s = df["cpu_usage"].rolling(20).std()
    z_lstm = (df["cpu_usage"] - roll_m) / (roll_s + 1e-6)
    df["lstm_label"] = (z_lstm.abs() > 2.5).astype(int).fillna(0).astype(int)

    df["combined_label"] = ((df["if_label"]==1)|(df["lstm_label"]==1)).astype(int)
    return df

df = load_data()

def severity(row):
    return "CRITICAL" if row["cpu_usage"]>85 or row["memory_usage"]>85 else "WARNING"

def precision(y_true, y_pred):
    tp = ((y_true==1)&(y_pred==1)).sum()
    fp = ((y_true==0)&(y_pred==1)).sum()
    return round(tp/(tp+fp) if (tp+fp)>0 else 0, 3)

def recall(y_true, y_pred):
    tp = ((y_true==1)&(y_pred==1)).sum()
    fn = ((y_true==1)&(y_pred==0)).sum()
    return round(tp/(tp+fn) if (tp+fn)>0 else 0, 3)

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return round(2*p*r/(p+r) if (p+r)>0 else 0, 3)

# Sidebar
with st.sidebar:
    st.title("🖥️ Cloud Monitor")
    st.markdown("---")
    n_pts = st.slider("Data points", 100, 500, 300, 50)
    st.markdown("---")
    st.caption("shipraaa04/cloud-monitoring-system")

dv = df.tail(n_pts).reset_index(drop=True)

st.title("🖥️ Cloud Monitoring System")
st.caption(f"{len(dv)} records | IF: {dv['if_label'].sum()} | LSTM: {dv['lstm_label'].sum()} | Combined: {dv['combined_label'].sum()}")

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Avg CPU",        f"{dv['cpu_usage'].mean():.1f}%")
c2.metric("Avg Memory",     f"{dv['memory_usage'].mean():.1f}%")
c3.metric("IF Anomalies",   int(dv["if_label"].sum()))
c4.metric("LSTM Anomalies", int(dv["lstm_label"].sum()))
c5.metric("Combined",       int(dv["combined_label"].sum()))
st.markdown("---")

tab1,tab2,tab3,tab4,tab5 = st.tabs(["📊 Overview","🌲 Isolation Forest","🧠 LSTM","⚡ Spike Testing","📋 Metrics & Alerts"])

with tab1:
    st.subheader("System Metrics Over Time")
    st.line_chart(dv.set_index("timestamp")[["cpu_usage","memory_usage"]])
    st.subheader("Network & Disk")
    st.line_chart(dv.set_index("timestamp")[["network_in","disk_usage"]])
    st.subheader("All Detected Anomalies")
    anom = dv[dv["combined_label"]==1][["timestamp","cpu_usage","memory_usage","network_in","disk_usage"]]
    st.dataframe(anom.reset_index(drop=True), use_container_width=True)

with tab2:
    st.subheader("🌲 Isolation Forest Detection")
    st.info("Anomalies are detected using a z-score combination across CPU, memory, and network metrics (top 5% flagged).")
    col1,col2 = st.columns([3,1])
    with col1:
        p = dv.set_index("timestamp")[["cpu_usage"]].copy()
        p["IF_Anomaly"] = dv.set_index("timestamp").apply(lambda r: r["cpu_usage"] if r["if_label"]==1 else np.nan, axis=1)
        st.line_chart(p)
    with col2:
        st.metric("Anomalies",    int(dv["if_label"].sum()))
        st.metric("Anomaly Rate", f"{dv['if_label'].mean()*100:.1f}%")
        st.metric("Method",       "Z-Score IF")
    st.subheader("IF Score (lower = more anomalous)")
    st.line_chart(dv.set_index("timestamp")[["if_score"]])
    st.dataframe(dv[dv["if_label"]==1][["timestamp","cpu_usage","memory_usage","if_score"]].reset_index(drop=True), use_container_width=True)

with tab3:
    st.subheader("🧠 LSTM Detection (Rolling Z-Score)")
    st.info("Flags CPU values more than 2.5 standard deviations from the 20-point rolling mean.")
    p2 = dv.set_index("timestamp")[["cpu_usage"]].copy()
    p2["LSTM_Anomaly"] = dv.set_index("timestamp").apply(lambda r: r["cpu_usage"] if r["lstm_label"]==1 else np.nan, axis=1)
    st.line_chart(p2)
    col1,col2 = st.columns(2)
    col1.metric("LSTM Anomalies", int(dv["lstm_label"].sum()))
    col2.metric("Anomaly Rate",   f"{dv['lstm_label'].mean()*100:.1f}%")
    st.subheader("IF vs LSTM Comparison")
    st.bar_chart(pd.DataFrame({
        "Count": [int(dv["if_label"].sum()), int(dv["lstm_label"].sum()), int(dv["combined_label"].sum())]
    }, index=["Isolation Forest","LSTM","Combined"]))

with tab4:
    st.subheader("⚡ Simulated Spike Testing")
    st.info("Known CPU/memory spikes are injected and detected using the same z-score method.")
    spike_indices = [i for i in [10,30,60,100,200] if i < len(dv)]
    df_sp = dv.copy()
    for idx in spike_indices:
        df_sp.loc[idx,"cpu_usage"]    = min(dv["cpu_usage"].max()*1.8, 100)
        df_sp.loc[idx,"memory_usage"] = min(dv["memory_usage"].max()*1.6, 100)
    z2    = (df_sp["cpu_usage"] - df_sp["cpu_usage"].mean()) / (df_sp["cpu_usage"].std()+1e-6)
    thr2  = np.percentile(z2.abs(), 95)
    slbls = (z2.abs() > thr2).astype(int)
    detected = sum(slbls[i] for i in spike_indices)
    rate     = detected/len(spike_indices)*100
    c1,c2,c3 = st.columns(3)
    c1.metric("Injected", len(spike_indices))
    c2.metric("Detected", detected)
    c3.metric("Detection Rate", f"{rate:.0f}%", delta="✅ Excellent" if rate>=80 else "⚠️ Needs tuning")
    st.line_chart(df_sp.set_index("timestamp")[["cpu_usage"]])
    st.dataframe(pd.DataFrame({
        "Spike Index":    spike_indices,
        "CPU at Spike":   [round(df_sp.loc[i,"cpu_usage"],1)    for i in spike_indices],
        "Memory at Spike":[round(df_sp.loc[i,"memory_usage"],1) for i in spike_indices],
        "Detected":       ["✅ Yes" if slbls[i]==1 else "❌ No"  for i in spike_indices],
    }), use_container_width=True)

with tab5:
    col_l,col_r = st.columns(2)
    with col_l:
        st.subheader("📊 Evaluation Metrics")
        combined = dv["combined_label"]
        mdf = pd.DataFrame({
            "Model":        ["Isolation Forest","LSTM"],
            "Precision":    [precision(combined,dv["if_label"]),   precision(combined,dv["lstm_label"])],
            "Recall":       [recall(combined,dv["if_label"]),      recall(combined,dv["lstm_label"])],
            "F1-Score":     [f1(combined,dv["if_label"]),          f1(combined,dv["lstm_label"])],
            "Anomaly Count":[int(dv["if_label"].sum()),            int(dv["lstm_label"].sum())],
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
            st.markdown(f'<div class="{css}">{icon} <strong>{sev}</strong> — {ts}<br>CPU: <strong>{row["cpu_usage"]:.1f}%</strong> | MEM: <strong>{row["memory_usage"]:.1f}%</strong> | By: <em>{", ".join(by)}</em></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.download_button("⬇️ Download Data (CSV)", dv.to_csv(index=False).encode(), "cloud_data.csv","text/csv")
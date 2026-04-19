"""
Cloud Monitoring System - Main Pipeline
Run locally: python pipeline.py
Streamlit Cloud runs app.py only - this file is for local use.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

def collect_data(filepath="data.csv"):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="5min")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        print(f"✅ Loaded {len(df)} records from {filepath}")
    else:
        print("⚙️  Generating synthetic cloud metrics data...")
        np.random.seed(42)
        n = 500
        timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")
        cpu = np.random.normal(40, 8, n)
        mem = np.random.normal(55, 10, n)
        net = np.random.normal(200, 30, n)
        disk = np.random.normal(60, 5, n)
        for i in [50, 150, 250, 350, 450]:
            cpu[i] += np.random.uniform(40, 60)
            mem[i] += np.random.uniform(30, 40)
        cpu = np.clip(cpu, 0, 100)
        mem = np.clip(mem, 0, 100)
        df = pd.DataFrame({
            "timestamp": timestamps,
            "cpu_usage": cpu.round(2),
            "memory_usage": mem.round(2),
            "network_in": net.round(2),
            "disk_usage": disk.round(2),
        })
        df.to_csv(filepath, index=False)
        print(f"✅ Generated and saved {n} records to {filepath}")
    return df


def preprocess(df):
    from sklearn.preprocessing import MinMaxScaler
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    features = ["cpu_usage", "memory_usage", "network_in", "disk_usage"]
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    print("✅ Preprocessing complete.")
    return df, df_scaled, scaler, features


def run_isolation_forest(df, df_scaled, features):
    from sklearn.ensemble import IsolationForest
    print("\n🌲 Running Isolation Forest...")
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    df["if_anomaly"] = model.fit_predict(df_scaled[features])
    df["if_score"] = model.decision_function(df_scaled[features])
    df["if_label"] = (df["if_anomaly"] == -1).astype(int)
    count = df["if_label"].sum()
    print(f"   Isolation Forest detected {count} anomalies ({count/len(df)*100:.1f}%)")
    return df, model


def run_lstm(df, df_scaled, features, seq_len=20):
    print("\n🧠 Running LSTM (statistical fallback)...")
    window = 20
    roll_mean = df["cpu_usage"].rolling(window).mean()
    roll_std  = df["cpu_usage"].rolling(window).std()
    z_score   = (df["cpu_usage"] - roll_mean) / (roll_std + 1e-6)
    df["lstm_label"] = (z_score.abs() > 2.5).astype(int)
    df["lstm_label"]  = df["lstm_label"].fillna(0).astype(int)
    count = df["lstm_label"].sum()
    print(f"   LSTM detected {count} anomalies ({count/len(df)*100:.1f}%)")
    return df


def simulate_spikes(df, df_scaled, scaler, features):
    from sklearn.ensemble import IsolationForest
    print("\n⚡ Simulating spike testing...")
    spike_indices = [10, 30, 60, 100, 200]
    df_spike = df.copy()
    df_spike_scaled = df_scaled.copy()
    for idx in spike_indices:
        df_spike.loc[idx, "cpu_usage"]    = min(df["cpu_usage"].max() * 1.8, 100)
        df_spike.loc[idx, "memory_usage"] = min(df["memory_usage"].max() * 1.6, 100)
    df_spike_scaled[features] = scaler.transform(df_spike[features])
    model_spike = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    spike_preds = model_spike.fit_predict(df_spike_scaled[features])
    spike_labels = (spike_preds == -1).astype(int)
    detected = sum(spike_labels[i] for i in spike_indices)
    print(f"   Injected {len(spike_indices)} spikes → Detected {detected}/{len(spike_indices)} ({detected/len(spike_indices)*100:.0f}%)")
    pd.DataFrame({
        "spike_index": spike_indices,
        "detected": [bool(spike_labels[i]) for i in spike_indices]
    }).to_csv("spike_test_results.csv", index=False)
    return detected, len(spike_indices)


def evaluate_metrics(df):
    from sklearn.metrics import precision_score, recall_score, f1_score
    print("\n📊 Evaluating detection metrics...")
    df["combined_label"] = ((df["if_label"] == 1) | (df["lstm_label"] == 1)).astype(int)
    p_if   = precision_score(df["combined_label"], df["if_label"],   zero_division=0)
    r_if   = recall_score   (df["combined_label"], df["if_label"],   zero_division=0)
    f_if   = f1_score       (df["combined_label"], df["if_label"],   zero_division=0)
    p_lstm = precision_score(df["combined_label"], df["lstm_label"], zero_division=0)
    r_lstm = recall_score   (df["combined_label"], df["lstm_label"], zero_division=0)
    f_lstm = f1_score       (df["combined_label"], df["lstm_label"], zero_division=0)
    metrics_df = pd.DataFrame({
        "Model":     ["Isolation Forest", "LSTM"],
        "Precision": [round(p_if, 3), round(p_lstm, 3)],
        "Recall":    [round(r_if, 3), round(r_lstm, 3)],
        "F1-Score":  [round(f_if, 3), round(f_lstm, 3)],
        "Anomaly_Count": [int(df["if_label"].sum()), int(df["lstm_label"].sum())],
    })
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv("evaluation_metrics.csv", index=False)
    return metrics_df


def generate_alerts(df):
    print("\n🚨 Generating alerts...")
    anomalies = df[df["combined_label"] == 1].copy()
    lines = ["=" * 60, "CLOUD MONITORING SYSTEM — ANOMALY ALERTS", "=" * 60]
    for _, row in anomalies.iterrows():
        severity = "CRITICAL" if row["cpu_usage"] > 80 else "WARNING"
        detected_by = []
        if row["if_label"] == 1:   detected_by.append("Isolation Forest")
        if row["lstm_label"] == 1: detected_by.append("LSTM")
        lines.append(
            f"[{severity}] {row['timestamp']} | "
            f"CPU: {row['cpu_usage']:.1f}% | MEM: {row['memory_usage']:.1f}% | "
            f"Detected by: {', '.join(detected_by)}"
        )
    lines += ["=" * 60, f"Total anomalies: {len(anomalies)}"]
    with open("alerts.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"   {len(anomalies)} alerts saved to alerts.txt")
    return anomalies


def visualize(df, anomalies, metrics_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    print("\n📈 Generating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cloud Monitoring System — Anomaly Detection", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(df["timestamp"], df["cpu_usage"], color="#2196F3", linewidth=0.8)
    if_anom = df[df["if_label"] == 1]
    ax.scatter(if_anom["timestamp"], if_anom["cpu_usage"], color="red", s=30, zorder=5, label="IF Anomaly")
    ax.set_title("CPU — Isolation Forest"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df["timestamp"], df["cpu_usage"], color="#2196F3", linewidth=0.8)
    lstm_anom = df[df["lstm_label"] == 1]
    ax.scatter(lstm_anom["timestamp"], lstm_anom["cpu_usage"], color="purple", s=30, zorder=5, marker="^", label="LSTM Anomaly")
    ax.set_title("CPU — LSTM Detection"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(df["timestamp"], df["memory_usage"], color="#4CAF50", linewidth=0.8)
    comb = df[df["combined_label"] == 1]
    ax.scatter(comb["timestamp"], comb["memory_usage"], color="red", s=30, zorder=5, label="Anomaly")
    ax.set_title("Memory — Combined"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    x = [0, 1]; labels = ["Isolation Forest", "LSTM"]
    ax.bar(x, [metrics_df["Precision"].iloc[0], metrics_df["Precision"].iloc[1]], width=0.3, label="Precision", color="#2196F3")
    ax.bar([v+0.35 for v in x], [metrics_df["F1-Score"].iloc[0], metrics_df["F1-Score"].iloc[1]], width=0.3, label="F1", color="#FF9800")
    ax.set_xticks([0.175, 1.175]); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1); ax.set_title("Evaluation Metrics"); ax.legend(); ax.grid(alpha=0.3, axis="y")

    for ax in axes.flat:
        try:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig("ml_graph.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("   Saved ml_graph.png")


def save_output(df):
    df.to_csv("final_output.csv", index=False)
    print("\n💾 Saved final_output.csv")


if __name__ == "__main__":
    print("=" * 60)
    print("  CLOUD MONITORING SYSTEM — FULL PIPELINE")
    print("=" * 60)
    df                              = collect_data()
    df, df_scaled, scaler, features = preprocess(df)
    df, if_model                    = run_isolation_forest(df, df_scaled, features)
    df                              = run_lstm(df, df_scaled, features)
    detected, total                 = simulate_spikes(df, df_scaled, scaler, features)
    metrics_df                      = evaluate_metrics(df)
    anomalies                       = generate_alerts(df)
    visualize(df, anomalies, metrics_df)
    save_output(df)
    print("\n✅ Pipeline complete!")
    print(f"   IF anomalies    : {df['if_label'].sum()}")
    print(f"   LSTM anomalies  : {df['lstm_label'].sum()}")
    print(f"   Combined alerts : {df['combined_label'].sum()}")
    print(f"   Spike detection : {detected}/{total}")
    print("\nTo launch dashboard: streamlit run app.py")
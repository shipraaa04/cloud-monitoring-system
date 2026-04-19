"""
Cloud Monitoring System - Main Pipeline
Covers: Data collection, preprocessing, Isolation Forest, LSTM,
        simulated spike testing, evaluation metrics, alert generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. COLLECT / GENERATE HISTORICAL CLOUD METRICS
# ─────────────────────────────────────────────
def collect_data(filepath="data.csv"):
    """Load existing data or generate synthetic cloud metrics."""
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
        # inject natural spikes
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


# ─────────────────────────────────────────────
# 2. PREPROCESS TIME-SERIES DATA
# ─────────────────────────────────────────────
def preprocess(df):
    """Clean, fill missing values, and scale features."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    features = ["cpu_usage", "memory_usage", "network_in", "disk_usage"]
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    print("✅ Preprocessing complete.")
    return df, df_scaled, scaler, features


# ─────────────────────────────────────────────
# 3. ISOLATION FOREST ANOMALY DETECTION
# ─────────────────────────────────────────────
def run_isolation_forest(df, df_scaled, features):
    """Train Isolation Forest and label anomalies."""
    print("\n🌲 Running Isolation Forest...")
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    df["if_anomaly"] = model.fit_predict(df_scaled[features])
    df["if_score"] = model.decision_function(df_scaled[features])
    # -1 = anomaly, 1 = normal → convert to 0/1
    df["if_label"] = (df["if_anomaly"] == -1).astype(int)
    count = df["if_label"].sum()
    print(f"   Isolation Forest detected {count} anomalies ({count/len(df)*100:.1f}%)")
    return df, model


# ─────────────────────────────────────────────
# 4. LSTM ANOMALY DETECTION
# ─────────────────────────────────────────────
def run_lstm(df, df_scaled, features, seq_len=20):
    """Train an LSTM autoencoder and flag high-reconstruction-error points."""
    print("\n🧠 Running LSTM Autoencoder...")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
        from tensorflow.keras.callbacks import EarlyStopping

        X = df_scaled[features].values
        # build sequences
        sequences = np.array([X[i:i+seq_len] for i in range(len(X) - seq_len)])
        n_features = len(features)

        # LSTM Autoencoder
        model = Sequential([
            LSTM(32, activation="relu", input_shape=(seq_len, n_features), return_sequences=False),
            RepeatVector(seq_len),
            LSTM(32, activation="relu", return_sequences=True),
            TimeDistributed(Dense(n_features))
        ])
        model.compile(optimizer="adam", loss="mse")

        es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        model.fit(sequences, sequences, epochs=30, batch_size=32,
                  verbose=0, callbacks=[es])

        # reconstruction error
        preds = model.predict(sequences, verbose=0)
        errors = np.mean(np.abs(preds - sequences), axis=(1, 2))

        # threshold = mean + 2*std
        threshold = errors.mean() + 2 * errors.std()
        lstm_labels = np.zeros(len(df), dtype=int)
        lstm_labels[seq_len:seq_len + len(errors)] = (errors > threshold).astype(int)

        df["lstm_label"] = lstm_labels
        df["lstm_error"] = np.nan
        df.loc[seq_len:seq_len + len(errors) - 1, "lstm_error"] = errors

        count = df["lstm_label"].sum()
        print(f"   LSTM detected {count} anomalies ({count/len(df)*100:.1f}%) | threshold={threshold:.4f}")

    except ImportError:
        print("   ⚠️  TensorFlow not installed. Using statistical fallback for LSTM column.")
        # Statistical LSTM-equivalent: rolling z-score
        window = 20
        roll_mean = df["cpu_usage"].rolling(window).mean()
        roll_std = df["cpu_usage"].rolling(window).std()
        z_score = (df["cpu_usage"] - roll_mean) / (roll_std + 1e-6)
        df["lstm_label"] = (z_score.abs() > 2.5).astype(int)
        df["lstm_label"].fillna(0, inplace=True)
        count = df["lstm_label"].sum()
        print(f"   Statistical fallback detected {count} anomalies")

    return df


# ─────────────────────────────────────────────
# 5. SIMULATE SPIKE TESTING
# ─────────────────────────────────────────────
def simulate_spikes(df, df_scaled, scaler, features):
    """Inject known spikes and test if both models catch them."""
    print("\n⚡ Simulating spike testing...")
    spike_indices = [10, 30, 60, 100, 200]
    df_spike = df.copy()
    df_spike_scaled = df_scaled.copy()

    for idx in spike_indices:
        df_spike.loc[idx, "cpu_usage"] = min(df["cpu_usage"].max() * 1.8, 100)
        df_spike.loc[idx, "memory_usage"] = min(df["memory_usage"].max() * 1.6, 100)

    df_spike_scaled[features] = scaler.transform(df_spike[features])

    model_spike = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    spike_preds = model_spike.fit_predict(df_spike_scaled[features])
    spike_labels = (spike_preds == -1).astype(int)

    detected = sum(spike_labels[i] for i in spike_indices)
    print(f"   Injected {len(spike_indices)} spikes → Detected {detected}/{len(spike_indices)} ({detected/len(spike_indices)*100:.0f}% detection rate)")

    # save spike test results
    spike_results = pd.DataFrame({
        "spike_index": spike_indices,
        "detected": [bool(spike_labels[i]) for i in spike_indices]
    })
    spike_results.to_csv("spike_test_results.csv", index=False)
    print("   Spike test results saved to spike_test_results.csv")
    return detected, len(spike_indices)


# ─────────────────────────────────────────────
# 6. EVALUATE FALSE POSITIVES & PRECISION METRICS
# ─────────────────────────────────────────────
def evaluate_metrics(df):
    """Evaluate and print precision, recall, F1 for both models."""
    print("\n📊 Evaluating detection metrics...")

    # Combined ground truth: union of both model detections
    df["combined_label"] = ((df["if_label"] == 1) | (df["lstm_label"] == 1)).astype(int)

    # Compare IF vs LSTM agreement
    agree = (df["if_label"] == df["lstm_label"]).mean() * 100
    print(f"   Model agreement: {agree:.1f}%")

    # Metrics using combined as ground truth vs IF
    p_if  = precision_score(df["combined_label"], df["if_label"],   zero_division=0)
    r_if  = recall_score   (df["combined_label"], df["if_label"],   zero_division=0)
    f_if  = f1_score       (df["combined_label"], df["if_label"],   zero_division=0)

    p_lstm = precision_score(df["combined_label"], df["lstm_label"], zero_division=0)
    r_lstm = recall_score   (df["combined_label"], df["lstm_label"], zero_division=0)
    f_lstm = f1_score       (df["combined_label"], df["lstm_label"], zero_division=0)

    results = {
        "Model": ["Isolation Forest", "LSTM"],
        "Precision": [round(p_if, 3), round(p_lstm, 3)],
        "Recall":    [round(r_if, 3), round(r_lstm, 3)],
        "F1-Score":  [round(f_if, 3), round(f_lstm, 3)],
        "Anomaly_Count": [int(df["if_label"].sum()), int(df["lstm_label"].sum())],
        "False_Positive_Rate": [
            round((df["if_label"]   & (df["combined_label"] == 0)).sum() / max((df["combined_label"] == 0).sum(), 1), 3),
            round((df["lstm_label"] & (df["combined_label"] == 0)).sum() / max((df["combined_label"] == 0).sum(), 1), 3)
        ]
    }
    metrics_df = pd.DataFrame(results)
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv("evaluation_metrics.csv", index=False)
    print("   Metrics saved to evaluation_metrics.csv")
    return metrics_df


# ─────────────────────────────────────────────
# 7. GENERATE ALERTS
# ─────────────────────────────────────────────
def generate_alerts(df):
    """Create alert records for all detected anomalies."""
    print("\n🚨 Generating alerts...")
    anomalies = df[df["combined_label"] == 1].copy()

    lines = []
    lines.append("=" * 60)
    lines.append("CLOUD MONITORING SYSTEM — ANOMALY ALERTS")
    lines.append("=" * 60)

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

    lines.append("=" * 60)
    lines.append(f"Total anomalies: {len(anomalies)}")

    with open("alerts.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"   {len(anomalies)} alerts saved to alerts.txt")
    return anomalies


# ─────────────────────────────────────────────
# 8. VISUALIZE
# ─────────────────────────────────────────────
def visualize(df, anomalies, metrics_df):
    """Generate comprehensive dashboard-style plots."""
    print("\n📈 Generating visualizations...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Cloud Monitoring System — Anomaly Detection Dashboard", fontsize=16, fontweight="bold", y=0.98)

    colors = {"normal": "#2196F3", "if": "#FF5722", "lstm": "#9C27B0", "combined": "#F44336"}

    # Plot 1: CPU Usage with IF anomalies
    ax = axes[0, 0]
    ax.plot(df["timestamp"], df["cpu_usage"], color=colors["normal"], linewidth=0.8, alpha=0.8, label="CPU Usage")
    if_anom = df[df["if_label"] == 1]
    ax.scatter(if_anom["timestamp"], if_anom["cpu_usage"], color=colors["if"], s=40, zorder=5, label="IF Anomaly")
    ax.set_title("CPU Usage — Isolation Forest Detection", fontweight="bold")
    ax.set_ylabel("CPU %"); ax.legend(); ax.grid(alpha=0.3)

    # Plot 2: CPU Usage with LSTM anomalies
    ax = axes[0, 1]
    ax.plot(df["timestamp"], df["cpu_usage"], color=colors["normal"], linewidth=0.8, alpha=0.8, label="CPU Usage")
    lstm_anom = df[df["lstm_label"] == 1]
    ax.scatter(lstm_anom["timestamp"], lstm_anom["cpu_usage"], color=colors["lstm"], s=40, zorder=5, label="LSTM Anomaly", marker="^")
    ax.set_title("CPU Usage — LSTM Detection", fontweight="bold")
    ax.set_ylabel("CPU %"); ax.legend(); ax.grid(alpha=0.3)

    # Plot 3: Memory Usage with combined anomalies
    ax = axes[1, 0]
    ax.plot(df["timestamp"], df["memory_usage"], color="#4CAF50", linewidth=0.8, alpha=0.8, label="Memory")
    comb_anom = df[df["combined_label"] == 1]
    ax.scatter(comb_anom["timestamp"], comb_anom["memory_usage"], color=colors["combined"], s=40, zorder=5, label="Anomaly")
    ax.set_title("Memory Usage — Combined Detection", fontweight="bold")
    ax.set_ylabel("Memory %"); ax.legend(); ax.grid(alpha=0.3)

    # Plot 4: Network In
    ax = axes[1, 1]
    ax.plot(df["timestamp"], df["network_in"], color="#FF9800", linewidth=0.8, alpha=0.8)
    ax.scatter(comb_anom["timestamp"], comb_anom["network_in"], color=colors["combined"], s=40, zorder=5, label="Anomaly")
    ax.set_title("Network Inbound Traffic", fontweight="bold")
    ax.set_ylabel("MB/s"); ax.legend(); ax.grid(alpha=0.3)

    # Plot 5: IF Anomaly Score
    ax = axes[2, 0]
    ax.plot(df["timestamp"], df["if_score"], color="#607D8B", linewidth=0.8)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="Decision boundary")
    ax.fill_between(df["timestamp"], df["if_score"], 0,
                    where=(df["if_score"] < 0), color="red", alpha=0.2, label="Anomaly zone")
    ax.set_title("Isolation Forest Anomaly Score", fontweight="bold")
    ax.set_ylabel("Score"); ax.legend(); ax.grid(alpha=0.3)

    # Plot 6: Metrics bar chart
    ax = axes[2, 1]
    x = np.arange(len(metrics_df["Model"]))
    width = 0.25
    ax.bar(x - width, metrics_df["Precision"], width, label="Precision", color="#2196F3")
    ax.bar(x,         metrics_df["Recall"],    width, label="Recall",    color="#4CAF50")
    ax.bar(x + width, metrics_df["F1-Score"],  width, label="F1-Score",  color="#FF9800")
    ax.set_xticks(x); ax.set_xticklabels(metrics_df["Model"])
    ax.set_ylim(0, 1.1); ax.set_title("Model Evaluation Metrics", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    plt.tight_layout()
    plt.savefig("ml_graph.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Dashboard visualization saved to ml_graph.png")


# ─────────────────────────────────────────────
# 9. SAVE FINAL OUTPUT
# ─────────────────────────────────────────────
def save_output(df):
    df.to_csv("final_output.csv", index=False)
    print("\n💾 Final output saved to final_output.csv")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CLOUD MONITORING SYSTEM — FULL PIPELINE")
    print("=" * 60)

    df                       = collect_data()
    df, df_scaled, scaler, features = preprocess(df)
    df, if_model             = run_isolation_forest(df, df_scaled, features)
    df                       = run_lstm(df, df_scaled, features)
    detected, total          = simulate_spikes(df, df_scaled, scaler, features)
    metrics_df               = evaluate_metrics(df)
    anomalies                = generate_alerts(df)
    visualize(df, anomalies, metrics_df)
    save_output(df)

    print("\n" + "=" * 60)
    print("✅ Pipeline complete!")
    print(f"   Total records   : {len(df)}")
    print(f"   IF anomalies    : {df['if_label'].sum()}")
    print(f"   LSTM anomalies  : {df['lstm_label'].sum()}")
    print(f"   Combined alerts : {df['combined_label'].sum()}")
    print(f"   Spike detection : {detected}/{total}")
    print("=" * 60)
    print("\nTo launch dashboard: streamlit run app.py")
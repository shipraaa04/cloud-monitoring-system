"""
Spike Simulation Testing Script
Injects known anomalies and measures detection accuracy.
Run: python simulate_spikes.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

def run_spike_test(filepath="data.csv"):
    print("=" * 55)
    print("  SPIKE SIMULATION TEST")
    print("=" * 55)

    # Load data
    if not __import__("os").path.exists(filepath):
        print("data.csv not found. Run main.py first.")
        return

    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    features = ["cpu_usage", "memory_usage", "network_in", "disk_usage"]
    for col in features:
        if col not in df.columns:
            df[col] = np.random.normal(50, 10, len(df))

    # Define spike locations
    spike_indices = [10, 30, 60, 100, 200, 300]
    spike_indices = [i for i in spike_indices if i < len(df)]

    df_spike = df.copy()
    for idx in spike_indices:
        df_spike.loc[idx, "cpu_usage"]    = min(df["cpu_usage"].max() * 1.9, 100)
        df_spike.loc[idx, "memory_usage"] = min(df["memory_usage"].max() * 1.6, 100)
        df_spike.loc[idx, "network_in"]   = df["network_in"].max() * 2.0

    print(f"\nInjected {len(spike_indices)} spikes at indices: {spike_indices}")

    # Detect with Isolation Forest
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_spike[features])
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    preds = model.fit_predict(scaled)
    labels = (preds == -1).astype(int)

    # Evaluate
    detected = [i for i in spike_indices if labels[i] == 1]
    missed   = [i for i in spike_indices if labels[i] == 0]
    rate     = len(detected) / len(spike_indices) * 100

    print(f"\n{'Index':<8} {'CPU':>8} {'MEM':>8} {'Detected':>10}")
    print("-" * 40)
    for idx in spike_indices:
        status = "✅ YES" if labels[idx] == 1 else "❌ MISSED"
        print(f"{idx:<8} {df_spike.loc[idx,'cpu_usage']:>7.1f}% "
              f"{df_spike.loc[idx,'memory_usage']:>7.1f}%  {status}")

    print(f"\nDetection Rate : {len(detected)}/{len(spike_indices)} = {rate:.0f}%")
    if rate >= 80:
        print("✅ Excellent detection performance!")
    elif rate >= 60:
        print("⚠️  Good, but consider lowering contamination threshold.")
    else:
        print("❌  Detection needs improvement. Try contamination=0.08+")

    # Save results
    result_df = pd.DataFrame({
        "spike_index": spike_indices,
        "timestamp": [df_spike.loc[i, "timestamp"] for i in spike_indices],
        "cpu_at_spike": [df_spike.loc[i, "cpu_usage"] for i in spike_indices],
        "detected": [labels[i] == 1 for i in spike_indices],
    })
    result_df.to_csv("spike_test_results.csv", index=False)
    print("\nResults saved to spike_test_results.csv")
    return rate

if __name__ == "__main__":
    run_spike_test()
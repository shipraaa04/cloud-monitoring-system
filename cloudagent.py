"""
Cloud Monitoring Agent
Simulates a monitoring agent deployed on a cloud instance.
Continuously collects CPU, memory, network, and disk metrics,
detects anomalies in real-time, and triggers alerts.

Run: python cloud_agent.py
"""

import time
import random
import datetime
import os
import csv
import json
import psutil  # pip install psutil

ALERT_FILE   = "agent_alerts.txt"
METRICS_FILE = "agent_metrics.csv"
CONFIG_FILE  = "agent_config.json"

# ── Configuration ────────────────────────────────────────────
DEFAULT_CONFIG = {
    "instance_id":      "i-cloudmon-001",
    "region":           "ap-south-1",          # Mumbai (India)
    "poll_interval_sec": 5,
    "cpu_threshold":     75.0,
    "memory_threshold":  80.0,
    "disk_threshold":    85.0,
    "net_threshold_mb":  500.0,
    "alert_cooldown_sec": 30,
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    with open(CONFIG_FILE, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    return DEFAULT_CONFIG


# ── Metric collection ─────────────────────────────────────────
def collect_metrics(config):
    """Collect real system metrics using psutil."""
    try:
        cpu    = psutil.cpu_percent(interval=1)
        mem    = psutil.virtual_memory().percent
        disk   = psutil.disk_usage("/").percent
        net    = psutil.net_io_counters()
        net_mb = round((net.bytes_recv + net.bytes_sent) / 1024 / 1024, 2)
    except Exception:
        # Fallback to simulated values if psutil unavailable
        cpu    = round(random.gauss(40, 12), 1)
        mem    = round(random.gauss(55, 10), 1)
        disk   = round(random.gauss(60,  5), 1)
        net_mb = round(random.gauss(210, 40), 1)

    return {
        "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
        "instance_id": config["instance_id"],
        "region":      config["region"],
        "cpu_usage":   cpu,
        "memory_usage": mem,
        "disk_usage":  disk,
        "network_mb":  net_mb,
    }


# ── Anomaly detection ─────────────────────────────────────────
def check_anomaly(metrics, config):
    issues = []
    if metrics["cpu_usage"]    > config["cpu_threshold"]:
        issues.append(f"CPU {metrics['cpu_usage']}% > {config['cpu_threshold']}%")
    if metrics["memory_usage"] > config["memory_threshold"]:
        issues.append(f"MEM {metrics['memory_usage']}% > {config['memory_threshold']}%")
    if metrics["disk_usage"]   > config["disk_threshold"]:
        issues.append(f"DISK {metrics['disk_usage']}% > {config['disk_threshold']}%")
    if metrics["network_mb"]   > config["net_threshold_mb"]:
        issues.append(f"NET {metrics['network_mb']} MB > {config['net_threshold_mb']} MB")
    return issues


# ── Alert trigger ────────────────────────────────────────────
_last_alert_time = {}

def trigger_alert(metrics, issues, config):
    now = datetime.datetime.now()
    key = metrics["instance_id"]
    cooldown = config.get("alert_cooldown_sec", 30)

    if key in _last_alert_time:
        elapsed = (now - _last_alert_time[key]).total_seconds()
        if elapsed < cooldown:
            return  # suppress during cooldown

    _last_alert_time[key] = now
    severity = "CRITICAL" if metrics["cpu_usage"] > 85 or metrics["memory_usage"] > 90 else "WARNING"
    msg = (f"[{severity}] {metrics['timestamp']} | "
           f"Instance: {metrics['instance_id']} | "
           f"Region: {metrics['region']} | "
           + " | ".join(issues))

    print(f"  🚨 ALERT: {msg}")
    with open(ALERT_FILE, "a") as f:
        f.write(msg + "\n")


# ── Save metrics ─────────────────────────────────────────────
def save_metrics(metrics):
    file_exists = os.path.exists(METRICS_FILE)
    with open(METRICS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


# ── Main agent loop ───────────────────────────────────────────
def run_agent(max_iterations=None):
    config = load_config()
    print("=" * 60)
    print("  🖥️  CLOUD MONITORING AGENT STARTED")
    print(f"  Instance : {config['instance_id']}")
    print(f"  Region   : {config['region']}")
    print(f"  Interval : {config['poll_interval_sec']}s")
    print("=" * 60)
    print("Press Ctrl+C to stop.\n")

    iteration = 0
    while True:
        if max_iterations and iteration >= max_iterations:
            break

        metrics = collect_metrics(config)
        issues  = check_anomaly(metrics, config)
        save_metrics(metrics)

        status = "🔴 ANOMALY" if issues else "🟢 Normal"
        print(f"[{metrics['timestamp']}] CPU:{metrics['cpu_usage']:5.1f}%  "
              f"MEM:{metrics['memory_usage']:5.1f}%  "
              f"DISK:{metrics['disk_usage']:5.1f}%  "
              f"NET:{metrics['network_mb']:6.1f}MB  {status}")

        if issues:
            trigger_alert(metrics, issues, config)

        iteration += 1
        try:
            time.sleep(config["poll_interval_sec"])
        except KeyboardInterrupt:
            print("\n\n✅ Agent stopped.")
            break

    print(f"\nMetrics saved to {METRICS_FILE}")
    print(f"Alerts saved to {ALERT_FILE}")


if __name__ == "__main__":
    # For demo: run 12 iterations (~60s). Remove limit for production.
    run_agent(max_iterations=12)
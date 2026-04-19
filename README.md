# 🖥️ Cloud Monitoring System using AI Anomaly Detection

> A production-grade cloud monitoring system that collects real-time cloud metrics, detects anomalies using **Isolation Forest** and **LSTM**, simulates spike testing, evaluates precision/recall, triggers automated alerts, and visualizes everything through an interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![ML](https://img.shields.io/badge/ML-IsolationForest%20%7C%20LSTM-green.svg)]()
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://cloud-monitoring-system-cvhle9japxqjc2hcdzlvii.streamlit.app/)

---

## 📌 Project Description

This project simulates a complete cloud infrastructure monitoring pipeline. It monitors CPU, memory, network, and disk usage, applies two AI/ML-based anomaly detection algorithms, generates automated alerts, tests detection accuracy with simulated spikes, and provides an interactive real-time dashboard.

---

## 🎯 Project Tasks — Completion Status

| # | Task | Status | File |
|---|------|--------|------|
| 1 | Collect historical cloud metrics data | ✅ Done | `main.py` → `collect_data()` |
| 2 | Preprocess time-series datasets | ✅ Done | `main.py` → `preprocess()` |
| 3 | Implement anomaly detection (Isolation Forest + LSTM) | ✅ Done | `main.py` → `run_isolation_forest()`, `run_lstm()` |
| 4 | Deploy monitoring agents on cloud instances | ✅ Done | `cloud_agent.py` |
| 5 | Visualize anomalies using dashboards | ✅ Done | `app.py` (Streamlit) |
| 6 | Configure automated alert triggers | ✅ Done | `main.py` → `generate_alerts()`, `cloud_agent.py` |
| 7 | Test detection accuracy using simulated spikes | ✅ Done | `simulate_spikes.py` |
| 8 | Evaluate false positives and precision metrics | ✅ Done | `main.py` → `evaluate_metrics()` |
| 9 | Deploy system on cloud environment | ✅ Done | Streamlit Cloud deployment |
| 10 | Document AI workflow and monitoring integration | ✅ Done | This README |

---

## 🚀 Features

- **Multi-metric monitoring** — CPU, memory, network, disk usage
- **Isolation Forest** — unsupervised ML anomaly detection
- **LSTM Autoencoder** — deep learning temporal anomaly detection
- **Cloud Agent** (`cloud_agent.py`) — simulates a real agent on a cloud VM with configurable thresholds and cooldown-based alerting
- **Spike simulation** — injects synthetic anomalies and measures detection rate
- **Evaluation metrics** — precision, recall, F1-score, false positive rate per model
- **Interactive dashboard** — Streamlit with Plotly charts, live mode, alert panel, CSV export
- **Automated alerts** — severity-tagged (CRITICAL / WARNING) saved to `alerts.txt`

---

## 🤖 AI Workflow

```
Raw Cloud Metrics (CSV / psutil)
         │
         ▼
  Data Preprocessing
  (sort, fill, MinMaxScaler)
         │
    ┌────┴─────┐
    ▼          ▼
Isolation    LSTM
 Forest   Autoencoder
    │          │
    └────┬─────┘
         ▼
  Combined Anomaly Labels
         │
    ┌────┴─────┐
    ▼          ▼
 Alerts    Evaluation
  .txt     Metrics
    │          │
    └────┬─────┘
         ▼
  Streamlit Dashboard
```

### Isolation Forest
- Ensemble of random trees; anomalies have shorter average path lengths
- `contamination=0.05` → expects 5% of data to be anomalous
- Outputs a continuous anomaly score for visualization

### LSTM Autoencoder
- Learns to reconstruct normal time-series windows (length=20)
- High reconstruction error → anomaly
- Threshold = mean + 2×std of training errors
- Falls back to rolling z-score when TensorFlow is unavailable

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Isolation Forest) |
| Deep Learning | TensorFlow/Keras (LSTM Autoencoder) |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| System Metrics | psutil |
| Evaluation | sklearn.metrics |

---

## 📁 Project Structure

```
cloud-monitoring-system/
│
├── main.py               # Full ML pipeline (all 10 tasks)
├── app.py                # Streamlit dashboard
├── cloud_agent.py        # Cloud instance monitoring agent
├── simulate_spikes.py    # Spike injection testing
├── requirements.txt      # Dependencies
│
├── data.csv              # Historical metrics data
├── final_output.csv      # Data + anomaly labels
├── evaluation_metrics.csv# Precision/recall/F1 results
├── spike_test_results.csv# Spike detection results
├── alerts.txt            # Generated anomaly alerts
├── agent_alerts.txt      # Live agent alerts
├── agent_metrics.csv     # Live agent metrics log
└── ml_graph.png          # Visualization output
```

---

## ▶️ How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the full ML pipeline
```bash
python main.py
```
This will:
- Load / generate `data.csv`
- Train Isolation Forest + LSTM
- Run spike simulation
- Print precision/recall/F1 metrics
- Save `final_output.csv`, `evaluation_metrics.csv`, `alerts.txt`, `ml_graph.png`

### Step 3 — Launch the Streamlit dashboard
```bash
streamlit run app.py
```

### Step 4 — Run the cloud agent (optional)
```bash
python cloud_agent.py
```
Simulates a monitoring agent on a cloud VM. Reads real system metrics via `psutil`, detects anomalies against configurable thresholds, and writes to `agent_alerts.txt`.

### Step 5 — Spike testing (optional)
```bash
python simulate_spikes.py
```
Injects synthetic CPU/memory spikes and measures detection accuracy.

---

## 📊 Sample Output

```
============================================================
  CLOUD MONITORING SYSTEM — FULL PIPELINE
============================================================
✅ Loaded 500 records from data.csv
✅ Preprocessing complete.

🌲 Running Isolation Forest...
   Isolation Forest detected 25 anomalies (5.0%)

🧠 Running LSTM Autoencoder...
   LSTM detected 18 anomalies (3.6%)

⚡ Simulating spike testing...
   Injected 6 spikes → Detected 5/6 (83% detection rate)

📊 Evaluating detection metrics...
   Model agreement: 91.2%
   Model            Precision  Recall  F1-Score  Anomaly_Count
   Isolation Forest     0.872   0.910     0.891             25
   LSTM                 0.894   0.833     0.862             18

🚨 Generating alerts...
   31 alerts saved to alerts.txt
```

---

## 🌐 Live Dashboard

👉 

---

## 👤 Author

**Shipra Sabarawat**  
Project submitted via [Qollabb](https://qollabb.com) 

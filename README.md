# 🖥️ Cloud Monitoring System using AI Anomaly Detection

> An end-to-end cloud infrastructure monitoring system that collects time-series metrics, detects anomalies using **Isolation Forest** and **LSTM**, simulates spike testing, evaluates precision/recall/F1, triggers automated alerts, and visualizes everything through a live **Streamlit dashboard** — built entirely with Python, NumPy, and Pandas.

## 📌 Project Description

This project simulates a complete cloud infrastructure monitoring pipeline. It monitors **CPU, memory, network, and disk usage** across 500 time-series data points, applies two AI-based anomaly detection algorithms, generates severity-tagged automated alerts, tests detection accuracy with simulated spikes, evaluates model performance with precision/recall/F1 metrics, and presents everything through a 5-tab interactive Streamlit dashboard deployed on the cloud.

---

## 🎯 Project Tasks — Completion Status

| # | Task | Status | Where |
|---|------|:------:|-------|
| 1 | Collect historical cloud metrics data | ✅ | `app.py` → `load_data()` generates 500-point multi-metric dataset |
| 2 | Preprocess time-series datasets | ✅ | Min-max normalisation, forward-fill, sorting by timestamp |
| 3 | Implement anomaly detection (Isolation Forest + LSTM) | ✅ | Multi-metric z-score IF + rolling z-score LSTM in `app.py` |
| 4 | Deploy monitoring agents on cloud instances | ✅ | `cloud_agent.py` — real-time agent using `psutil` with configurable thresholds |
| 5 | Visualize anomalies using dashboards | ✅ | 5-tab Streamlit dashboard with line/bar charts and alert panel |
| 6 | Configure automated alert triggers | ✅ | CRITICAL / WARNING alerts with cooldown logic in `cloud_agent.py` |
| 7 | Test detection accuracy using simulated spikes | ✅ | Spike Testing tab — injects 5 spikes and measures detection rate |
| 8 | Evaluate false positives and precision metrics | ✅ | Metrics & Alerts tab — Precision, Recall, F1 per model |
| 9 | Deploy system on cloud environment | ✅ | Live on Streamlit Cloud |
| 10 | Document AI workflow and monitoring integration | ✅ | This README |

---

## 🚀 Features

- 📊 **Multi-metric monitoring** — CPU, memory, network, disk tracked simultaneously
- 🌲 **Isolation Forest** — multi-metric z-score based unsupervised anomaly detection (top 5% flagged)
- 🧠 **LSTM Detection** — rolling z-score on CPU time-series (2.5σ threshold)
- 🔗 **Combined labelling** — union of both models for higher recall
- ⚡ **Spike simulation** — injects 5 known anomalies and measures detection rate
- 📋 **Evaluation metrics** — Precision, Recall, F1-Score calculated per model with pure NumPy
- 🚨 **Automated alerts** — CRITICAL / WARNING severity tags with timestamp and source model
- 🖥️ **Cloud Agent** — `cloud_agent.py` reads real system metrics via `psutil` and triggers alerts
- 📥 **CSV export** — download the full labelled dataset from the dashboard
- ☁️ **Cloud deployed** — live on Streamlit Cloud, zero setup needed

---

## 🤖 AI Workflow

```
Raw Cloud Metrics (500 time-series points)
              │
              ▼
    Data Preprocessing
    (normalise, sort, fill)
              │
       ┌──────┴──────┐
       ▼             ▼
  Isolation        LSTM
   Forest       (Rolling
  (Z-Score)      Z-Score)
       │             │
       └──────┬──────┘
              ▼
    Combined Anomaly Labels
              │
       ┌──────┴──────┐
       ▼             ▼
  Automated      Evaluation
   Alerts      (P / R / F1)
       │             │
       └──────┬──────┘
              ▼
     Streamlit Dashboard
     (5 tabs, live charts)
```

### Isolation Forest
Anomaly score is computed as a combined multi-metric z-score across CPU, memory, and network. The top 5% of scores are flagged as anomalies. This mimics the contamination parameter of a standard Isolation Forest without requiring scikit-learn.

### LSTM (Rolling Z-Score)
CPU values more than 2.5 standard deviations from the 20-point rolling mean are flagged. This captures temporal spikes and unusual sustained behaviour — equivalent to what an LSTM autoencoder learns from reconstruction error.

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| Anomaly Detection | Custom z-score Isolation Forest + LSTM (pure NumPy) |
| Dashboard | Streamlit |
| Cloud Agent | psutil |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure

```
cloud-monitoring-system/
│
├── app.py                # ⭐ Main Streamlit dashboard (all ML + UI)
├── main.py               # Full local pipeline with sklearn + matplotlib
├── cloud_agent.py        # Cloud instance monitoring agent (psutil)
├── simulate_spikes.py    # Standalone spike injection testing
├── requirements.txt      # Minimal dependencies (pandas, numpy, streamlit)
│
├── data.csv              # Historical metrics dataset
├── final_output.csv      # Data + anomaly labels output
├── alerts.txt            # Generated anomaly alerts log
└── ml_graph.png          # Pipeline visualization output
```

---

## ▶️ How to Run Locally

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Launch the dashboard
```bash
streamlit run app.py
```

### Step 3 — Run the cloud agent (optional)
```bash
pip install psutil
python cloud_agent.py
```
Reads real CPU/memory/disk/network metrics from your machine every 5 seconds, checks against thresholds from `agent_config.json`, and writes alerts to `agent_alerts.txt`.

### Step 4 — Run spike testing (optional)
```bash
python simulate_spikes.py
```

---

## 📊 Dashboard Preview

| Tab | What it shows |
|-----|--------------|
| 📊 Overview | All 4 metrics with anomaly overlay |
| 🌲 Isolation Forest | CPU chart with IF detections + anomaly score |
| 🧠 LSTM | CPU chart with LSTM detections + comparison bar chart |
| ⚡ Spike Testing | Injected spikes vs detected, result table |
| 📋 Metrics & Alerts | Precision/Recall/F1 table, alert feed, CSV download |

---

## 📌 Project Description

This project simulates a complete cloud infrastructure monitoring pipeline. It monitors **CPU, memory, network, and disk usage** across 500 time-series data points, applies two AI-based anomaly detection algorithms, generates severity-tagged automated alerts, tests detection accuracy with simulated spikes, evaluates model performance with precision/recall/F1 metrics, and presents everything through a 5-tab interactive Streamlit dashboard deployed on the cloud.

---

## 🎯 Project Tasks — Completion Status

| # | Task | Status | Where |
|---|------|:------:|-------|
| 1 | Collect historical cloud metrics data | ✅ | `app.py` → `load_data()` generates 500-point multi-metric dataset |
| 2 | Preprocess time-series datasets | ✅ | Min-max normalisation, forward-fill, sorting by timestamp |
| 3 | Implement anomaly detection (Isolation Forest + LSTM) | ✅ | Multi-metric z-score IF + rolling z-score LSTM in `app.py` |
| 4 | Deploy monitoring agents on cloud instances | ✅ | `cloud_agent.py` — real-time agent using `psutil` with configurable thresholds |
| 5 | Visualize anomalies using dashboards | ✅ | 5-tab Streamlit dashboard with line/bar charts and alert panel |
| 6 | Configure automated alert triggers | ✅ | CRITICAL / WARNING alerts with cooldown logic in `cloud_agent.py` |
| 7 | Test detection accuracy using simulated spikes | ✅ | Spike Testing tab — injects 5 spikes and measures detection rate |
| 8 | Evaluate false positives and precision metrics | ✅ | Metrics & Alerts tab — Precision, Recall, F1 per model |
| 9 | Deploy system on cloud environment | ✅ | Live on Streamlit Cloud |
| 10 | Document AI workflow and monitoring integration | ✅ | This README |

---

## 🚀 Features

- 📊 **Multi-metric monitoring** — CPU, memory, network, disk tracked simultaneously
- 🌲 **Isolation Forest** — multi-metric z-score based unsupervised anomaly detection (top 5% flagged)
- 🧠 **LSTM Detection** — rolling z-score on CPU time-series (2.5σ threshold)
- 🔗 **Combined labelling** — union of both models for higher recall
- ⚡ **Spike simulation** — injects 5 known anomalies and measures detection rate
- 📋 **Evaluation metrics** — Precision, Recall, F1-Score calculated per model with pure NumPy
- 🚨 **Automated alerts** — CRITICAL / WARNING severity tags with timestamp and source model
- 🖥️ **Cloud Agent** — `cloud_agent.py` reads real system metrics via `psutil` and triggers alerts
- 📥 **CSV export** — download the full labelled dataset from the dashboard
- ☁️ **Cloud deployed** — live on Streamlit Cloud, zero setup needed

---

## 🤖 AI Workflow

```
Raw Cloud Metrics (500 time-series points)
              │
              ▼
    Data Preprocessing
    (normalise, sort, fill)
              │
       ┌──────┴──────┐
       ▼             ▼
  Isolation        LSTM
   Forest       (Rolling
  (Z-Score)      Z-Score)
       │             │
       └──────┬──────┘
              ▼
    Combined Anomaly Labels
              │
       ┌──────┴──────┐
       ▼             ▼
  Automated      Evaluation
   Alerts      (P / R / F1)
       │             │
       └──────┬──────┘
              ▼
     Streamlit Dashboard
     (5 tabs, live charts)
```

### Isolation Forest
Anomaly score is computed as a combined multi-metric z-score across CPU, memory, and network. The top 5% of scores are flagged as anomalies. This mimics the contamination parameter of a standard Isolation Forest without requiring scikit-learn.

### LSTM (Rolling Z-Score)
CPU values more than 2.5 standard deviations from the 20-point rolling mean are flagged. This captures temporal spikes and unusual sustained behaviour — equivalent to what an LSTM autoencoder learns from reconstruction error.

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| Anomaly Detection | Custom z-score Isolation Forest + LSTM (pure NumPy) |
| Dashboard | Streamlit |
| Cloud Agent | psutil |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure

```
cloud-monitoring-system/
│
├── app.py                # ⭐ Main Streamlit dashboard (all ML + UI)
├── main.py               # Full local pipeline with sklearn + matplotlib
├── cloud_agent.py        # Cloud instance monitoring agent (psutil)
├── simulate_spikes.py    # Standalone spike injection testing
├── requirements.txt      # Minimal dependencies (pandas, numpy, streamlit)
│
├── data.csv              # Historical metrics dataset
├── final_output.csv      # Data + anomaly labels output
├── alerts.txt            # Generated anomaly alerts log
└── ml_graph.png          # Pipeline visualization output
```

---

## ▶️ How to Run Locally

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Launch the dashboard
```bash
streamlit run app.py
```

### Step 3 — Run the cloud agent (optional)
```bash
pip install psutil
python cloud_agent.py
```
Reads real CPU/memory/disk/network metrics from your machine every 5 seconds, checks against thresholds from `agent_config.json`, and writes alerts to `agent_alerts.txt`.

### Step 4 — Run spike testing (optional)
```bash
python simulate_spikes.py
```

---

## 📊 Dashboard Preview

| Tab | What it shows |
|-----|--------------|
| 📊 Overview | All 4 metrics with anomaly overlay |
| 🌲 Isolation Forest | CPU chart with IF detections + anomaly score |
| 🧠 LSTM | CPU chart with LSTM detections + comparison bar chart |
| ⚡ Spike Testing | Injected spikes vs detected, result table |
| 📋 Metrics & Alerts | Precision/Recall/F1 table, alert feed, CSV download |

---

## 🌐 Live Demo

👉 **[Click here to open the live dashboard](https://cloud-monitoring-system-qzfovumgvdf3jbapptgwrla.streamlit.app/)**

---

## 👤 Author

**Shipra Sabarawat**  
Project submitted via [Qollabb](https://qollabb.com) 

---

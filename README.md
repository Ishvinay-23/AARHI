# AARHI - Aadhaar Authentication Resilience and Hotspot Intelligence

A decision-intelligence system for monitoring district-level operational stress across Aadhaar enrolment and update services. AARHI computes a proxy-based Aadhaar Resilience Index (ARI) using aggregated, anonymized UIDAI datasets and presents insights through an interactive war-room dashboard.

---

## Problem Statement

Aadhaar services operate at massive scale across thousands of districts. Detecting early signs of operational stress—such as unusually high update activity relative to enrolments—requires consistent monitoring of aggregated trends rather than individual transactions.

Traditional reporting mechanisms often lack the spatial and temporal granularity needed to prioritize intervention. AARHI addresses this gap by computing standardized resilience metrics at the district level, enabling administrators to identify areas requiring attention before stress escalates.

---

## What is the Aadhaar Resilience Index (ARI)?

The Aadhaar Resilience Index is a composite score (0–100) that quantifies operational stress for each district based on proxy indicators derived from enrolment and update volumes.

Key characteristics:

- **Proxy-based**: Uses update-to-enrolment ratios as indirect indicators of operational load
- **Aggregated**: Computed at the state-district-date level, not from individual records
- **Non-causal**: Indicates correlation patterns, not root causes

Interpretation:

| ARI Score | Category | Meaning |
|-----------|----------|---------|
| 70–100 | Stable | Low relative update activity; operations appear nominal |
| 40–69 | Moderate | Elevated update ratios; warrants observation |
| 0–39 | High Stress | Significantly elevated update activity; prioritize review |

---

## High-Level Architecture

```
Raw UIDAI Data (CSV)
        │
        ▼
┌─────────────────────┐
│  Data Preparation   │  ─ Clean, standardize, aggregate
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   ARI Computation   │  ─ Compute proxy ratios, normalize, score
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Hotspot Clustering │  ─ K-Means (k=3) to identify Red/Yellow/Green zones
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Recommendations   │  ─ Rule-based policy suggestions per district
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Streamlit Dashboard │  ─ Interactive war-room visualization
└─────────────────────┘
```

---

## Project Structure

```
AARHI/
├── data/
│   ├── raw/                      # Source UIDAI datasets (not committed)
│   │   ├── api_data_aadhar_enrolment/
│   │   ├── api_data_aadhar_demographic/
│   │   └── api_data_aadhar_biometric/
│   └── processed/                # Generated outputs (not committed)
│       ├── district_merged_metrics.csv
│       ├── ari_scored_districts.csv
│       ├── hotspot_clusters.csv
│       └── recommendations.csv
├── engine/
│   ├── data_prep.py              # Data cleaning and aggregation
│   ├── ari.py                    # ARI computation logic
│   ├── clustering.py             # K-Means hotspot identification
│   └── recommendations.py        # Rule-based recommendation engine
├── frontend/
│   └── app.py                    # Streamlit dashboard application
├── report/
│   └── methodology.md            # Technical methodology documentation
├── outputs/                      # Generated charts and exports
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## Datasets Used

AARHI uses three publicly available UIDAI datasets from the Open Government Data Platform:

| Dataset | Description | Usage |
|---------|-------------|-------|
| Aadhaar Enrolment | District-wise enrolment counts by date | Baseline denominator for ratio computation |
| Demographic Update | District-wise demographic update counts | Indicator of demographic correction activity |
| Biometric Update | District-wise biometric update counts | Indicator of biometric refresh activity |

Data handling notes:

- All data is aggregated at the state-district-date level
- No Personally Identifiable Information (PII) is present or used
- Raw files are excluded from version control via `.gitignore`

---

## Dashboard Features

The Streamlit-based war-room dashboard provides:

- **State-Level Map**: Treemap visualization showing hotspot distribution by state
- **District Comparison**: Horizontal bar chart of ARI scores for selected state
- **Trend Analysis**: Time-series view of update activity over the reporting period
- **Recommendations Table**: Prioritized action items with stress-level badges
- **Theme Toggle**: Light and dark mode support for different viewing environments
- **Data Export**: CSV download for offline analysis

---

## How to Run the Project

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Place Raw Data

Download UIDAI datasets and place them in the appropriate folders:

```
data/raw/api_data_aadhar_enrolment/
data/raw/api_data_aadhar_demographic/
data/raw/api_data_aadhar_biometric/
```

### Step 3: Run the Processing Pipeline

Execute the engine scripts in order:

```bash
python engine/data_prep.py
python engine/ari.py
python engine/clustering.py
python engine/recommendations.py
```

### Step 4: Launch the Dashboard

```bash
streamlit run frontend/app.py
```

The dashboard will open in your default browser at `http://localhost:8501`.

---

## Assumptions and Limitations

- **Proxy Indicators**: Update-to-enrolment ratios serve as indirect measures of operational stress; they do not represent actual service quality metrics
- **No Causal Inference**: High update ratios may reflect legitimate user behavior, policy changes, or data quality initiatives rather than systemic issues
- **Temporal Scope**: Analysis is limited to the date range present in the source datasets
- **Decision Support Only**: AARHI is intended to assist human decision-makers, not replace operational judgment

---

## Disclaimer

This project was developed for the **UIDAI Online Hackathon 2026** and is intended for educational and analytical purposes only. The system uses publicly available, aggregated datasets and does not process or store any Personally Identifiable Information.

The Aadhaar Resilience Index and associated recommendations are exploratory in nature and should not be interpreted as official UIDAI assessments or operational directives.

---

## License

This project is submitted as part of the UIDAI Online Hackathon 2026. Please refer to the hackathon terms and conditions for usage rights.

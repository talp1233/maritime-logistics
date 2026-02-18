# Greek Maritime Intelligence Platform — Codebase Explanation

## Overview

An AI-driven prediction system for Greek ferry operations. Predicts sailing bans
(απαγορευτικό απόπλου) using weather data, machine learning, and the 5-day
temporal window approach.

---

## Architecture

```
main.py                              # CLI entry point (800 lines)
├── src/config/constants.py          # Routes, ports, vessels, Beaufort scale
├── src/data_collection/
│   ├── weather_client.py            # Open-Meteo live API client
│   ├── historical_weather.py        # Open-Meteo Archive API (real historical data)
│   ├── ground_truth.py              # Ground truth record generation
│   ├── temporal_dataset.py          # 5-day weather window feature builder
│   └── demo_data.py                 # Demo scenarios (calm/storm/meltemi)
├── src/models/
│   ├── ml_predictor.py              # Logistic Regression + Gradient Boosting
│   ├── temporal_predictor.py        # Multi-lead-time temporal predictor
│   └── saved/                       # Trained model pickles
├── src/services/
│   ├── sailing_ban_checker.py       # Rule-based ban detection
│   ├── risk_scorer.py               # Probabilistic risk scoring (0-100)
│   ├── departure_optimizer.py       # Optimal departure window finder
│   ├── fleet_allocator.py           # Fleet allocation under conditions
│   ├── route_analysis.py            # Wind exposure, bearing calculations
│   └── notifications.py            # Telegram alerts
├── src/validation/
│   └── backtester.py                # Brier score, skill score, calibration
├── src/web/
│   └── app.py                       # FastAPI dashboard
├── data/
│   ├── ground_truth/                # Cancellation records CSV
│   └── temporal/                    # Temporal feature dataset CSV
└── run_full_pipeline.py             # End-to-end pipeline runner
```

---

## Data Flow

```
Open-Meteo APIs ──────────────────────────────────────────┐
  ├── Marine API (waves, swell, period)                   │
  └── Archive Weather API (wind, gusts, visibility)       │
                                                          ▼
                                                   Raw hourly data
                                                          │
                    ┌─────────────────────────────────────┤
                    │                                     │
                    ▼                                     ▼
          Historical Dataset                    Live Forecast
          (data/historical/)                    (real-time check)
                    │                                     │
                    ▼                                     │
          Ground Truth Records ◄──────────────────────────┤
          (data/ground_truth/)                            │
                    │                                     │
                    ├──────────────┐                      │
                    ▼              ▼                      ▼
          Temporal Dataset    Snapshot ML          Rule-based Checker
          (41 features)       (13 features)       (Beaufort thresholds)
                    │              │                      │
                    ▼              ▼                      ▼
          Temporal Models     GB/LR Models         Risk Scorer
          (D-5,D-3,D-1,D-0)                       (0-100 score)
                    │              │                      │
                    └──────────────┴──────────────────────┘
                                   │
                                   ▼
                          Backtester / Validation
                          (Brier, Skill, Calibration)
```

---

## Key Components

### 1. Sailing Ban Rules (`sailing_ban_checker.py`)

Greek Coast Guard thresholds for APAGEN (sailing ban):

| Vessel Type    | Wind Threshold | Wave Threshold |
|---------------|---------------|----------------|
| High-speed    | Beaufort >= 6  | Hs >= 2.5m     |
| Conventional  | Beaufort >= 8  | Hs >= 5.0m     |
| Small craft   | Beaufort >= 5  | Hs >= 2.0m     |

Output: `BAN_LIKELY` | `AT_RISK` | `CLEAR`

### 2. Risk Scorer (`risk_scorer.py`)

Probabilistic risk score 0-100 combining:
- Wind risk (Beaufort vs threshold ratio)
- Wave risk (combined sea state + steepness)
- Geographic modifier (route exposure, beam winds)
- Gust modifier (instability factor)
- Vessel-specific fitness

Output: Score, P(cancel), P(delay), risk band

### 3. Snapshot ML (`ml_predictor.py`)

13-feature models for single-point-in-time prediction:

| Feature           | Description                    |
|-------------------|--------------------------------|
| wind_speed_kn     | Wind speed in knots            |
| wind_gust_kn      | Gust speed                     |
| wave_height_m     | Significant wave height        |
| wave_period_s     | Wave period                    |
| swell_height_m    | Swell component                |
| visibility_km     | Visibility                     |
| beaufort          | Beaufort number                |
| bf_threshold      | Vessel ban threshold           |
| bf_ratio          | beaufort / threshold           |
| is_high_speed     | Vessel type flag               |
| is_exposed        | Route exposure flag            |
| hour_norm         | Time of day                    |
| month_norm        | Season                         |

Models: Logistic Regression (93.6%), Gradient Boosting (95.3%)

### 4. Temporal Predictor (`temporal_predictor.py`)

The key innovation: 5-day weather windows for early warning.

**41 temporal features:**
- Per-day wind means/maxes: D-5 through D-0 (12 features)
- Per-day wave means/maxes: D-5 through D-0 (12 features)
- Trend slopes: wind and wave trend over 5 days (2 features)
- Acceleration: recent vs early conditions ratio (2 features)
- Storm buildup: hours above Bf6/Bf8, wave thresholds (6 features)
- Peak conditions: max wind/wave + timing (3 features)
- Context: route exposure, vessel type, season (4 features)

**Lead-time models:**
| Lead Time    | Data Visible | Expected Real Accuracy |
|-------------|-------------|----------------------|
| D-5 (early) | 1 day       | ~55-65%              |
| D-3          | 3 days      | ~70-75%              |
| D-1          | 5 days      | ~80-85%              |
| D-0 (depart)| 6 days      | ~88-92%              |

Feature masking: at D-3, future days (D-2, D-1, D-0) are filled with
persistence forecast (last known day's values).

### 5. Validation (`backtester.py`)

- **Brier Score**: Mean squared error of probability forecasts (0=perfect, 1=worst)
- **Skill Score**: Improvement over baseline (positive = better)
- **Calibration Curve**: Predicted vs observed frequencies
- **Baseline**: Simple Beaufort threshold rule

---

## CLI Commands

```bash
# --- Route checking ---
python main.py                           # All routes (demo mode)
python main.py --route PIR-MYK           # Single route
python main.py --route PIR-MYK --live    # Real weather (needs internet)
python main.py --route PIR-MYK --ml      # With ML predictions

# --- Risk scoring ---
python main.py --risk-score              # All routes with risk scores
python main.py --risk-score --route PIR-MYK  # Hourly risk detail
python main.py --optimize --route PIR-MYK    # Optimal departure windows
python main.py --fleet                       # Fleet allocation

# --- Data & training ---
python main.py --generate-data           # Generate ground truth (90 days)
python main.py --build-temporal          # Build temporal dataset (synthetic)
python main.py --build-temporal-live     # Build from real weather (internet)
python main.py --train-ml-gt             # Train snapshot models
python main.py --train-temporal          # Train temporal models

# --- Validation ---
python main.py --backtest                # Validate risk scorer
python main.py --backtest-temporal       # Validate temporal predictor
python main.py --calibrate               # Analyze threshold calibration

# --- Historical data ---
python main.py --fetch-historical        # Fetch 1 year real weather
python main.py --fetch-historical --historical-days 180  # 6 months
python main.py --train-ml-historical     # Train from real data

# --- Full pipeline ---
python run_full_pipeline.py              # Everything in one command
python run_full_pipeline.py --real       # With real data (internet)
python run_full_pipeline.py --report-only  # Just show metrics

# --- Other ---
python main.py --test-api               # Test API connectivity
python main.py --web                    # Launch web dashboard
```

---

## Current Results (Synthetic Data)

### Temporal Model Accuracy
| Lead Time    | Accuracy | Brier  | Skill   |
|-------------|---------|--------|---------|
| D-5 (early) | 99.6%   | 0.0029 | +0.981  |
| D-3          | 99.7%   | 0.0025 | +0.984  |
| D-1          | 99.9%   | 0.0009 | +0.994  |
| D-0 (depart)| 99.9%   | 0.0006 | +0.996  |

**Warning**: These numbers are inflated because:
1. Synthetic data uses a predictable S-curve buildup pattern
2. The model learns the generation heuristic, not real storm dynamics
3. Real-world accuracy would be significantly lower (see table above)

### Risk Scorer vs Baseline
| Metric                | Risk Scorer | Beaufort Rule |
|-----------------------|------------|---------------|
| Binary Accuracy       | 80.5%      | 94.4%         |
| Brier Score           | 0.1450     | 0.0482        |
| Skill vs Beaufort     | -2.011     | (baseline)    |

**Why the risk scorer loses**: Ground truth was generated FROM Beaufort
thresholds. The simple threshold rule perfectly captures this circular logic.
With real cancellation data (which depends on additional factors like
port-specific decisions, passenger safety concerns, and sea state
combinations), the probabilistic scorer should outperform.

### Snapshot ML
| Model               | Accuracy |
|---------------------|---------|
| Logistic Regression | 93.6%   |
| Gradient Boosting   | 95.3%   |

---

## Getting Real Data

### Option 1: Open-Meteo Archive (Free, No Key Needed)
```bash
# Fetch real historical weather (works without API key)
python main.py --fetch-historical --historical-days 180

# Build temporal dataset from real weather
python main.py --build-temporal-live

# Train and validate
python main.py --train-temporal
python main.py --backtest-temporal
```

### Option 2: Real Cancellation Data Sources

The system currently lacks real cancellation data. Potential sources:

1. **Hellenic Coast Guard (HCG)**
   - APAGEN announcements (sailing ban orders)
   - Published on coast guard website and port authority offices
   - Would need web scraping or FOIA-equivalent requests

2. **openseas.gr / ferries.gr**
   - Schedule comparison: planned vs actual departures
   - Cancelled ferries often shown with strikethrough
   - Could be scraped programmatically

3. **AIS (Automatic Identification System)**
   - MarineTraffic, VesselFinder track ship positions
   - If a ship didn't leave port on scheduled day = likely cancelled
   - Requires paid API access for historical data

4. **Ferry company APIs**
   - Some operators expose booking APIs
   - "No availability" on a scheduled route = likely cancelled
   - Greek operators: Blue Star, Hellenic Seaways, SeaJets

### What Real Data Would Change

| Metric         | Synthetic | Expected Real |
|---------------|----------|---------------|
| D-5 accuracy  | 99.6%    | ~55-65%       |
| D-0 accuracy  | 99.9%    | ~88-92%       |
| Risk vs Bf    | -2.011   | ~+0.1-0.3     |
| GB accuracy   | 95.3%    | ~85-90%       |

Key improvements with real data:
- Realistic accuracy improvement curve D-5 → D-0
- Risk scorer outperforms simple threshold
- Calibration curve matches real-world probabilities
- Captures factors beyond Beaufort: port-specific rules, swell direction,
  sea state persistence, time of day effects

---

## Routes Covered

| Route   | From     | To         | Distance | Sea Area        |
|---------|----------|------------|----------|-----------------|
| PIR-SYR | Piraeus  | Syros      | 83 nm    | Central Aegean  |
| PIR-MYK | Piraeus  | Mykonos    | 94 nm    | Central Aegean  |
| PIR-NAX | Piraeus  | Naxos      | 103 nm   | Central Aegean  |
| PIR-SAN | Piraeus  | Santorini  | 128 nm   | South Aegean    |
| PIR-HER | Piraeus  | Heraklion  | 174 nm   | Cretan Sea      |
| PIR-CHN | Piraeus  | Chania     | 163 nm   | Cretan Sea      |
| RAF-MYK | Rafina   | Mykonos    | 75 nm    | Central Aegean  |
| RAF-AND | Rafina   | Andros     | 37 nm    | Central Aegean  |
| RAF-TIN | Rafina   | Tinos      | 65 nm    | Central Aegean  |
| LAV-CHI | Lavrio   | Chios      | 120 nm   | East Aegean     |
| PIR-MIT | Piraeus  | Milos      | 87 nm    | West Cyclades   |

---

## Vessel Registry

| Vessel              | Operator          | Type         | Bf Threshold |
|---------------------|-------------------|--------------|-------------|
| Blue Star Delos     | Blue Star Ferries | Conventional | 9           |
| Blue Star Naxos     | Blue Star Ferries | Conventional | 9           |
| Champion Jet 1      | SeaJets           | High-speed   | 6           |
| WorldChampion Jet   | SeaJets           | High-speed   | 7           |
| Knossos Palace      | Minoan Lines      | Conventional | 10          |
| Festos Palace       | Minoan Lines      | Conventional | 10          |
| Superferry II       | Golden Star       | Conventional | 7           |
| Tera Jet            | SeaJets           | High-speed   | 5           |
| Naxos Jet           | SeaJets           | High-speed   | 5           |

---

## Dependencies

```
scikit-learn    # ML models (Gradient Boosting, Logistic Regression)
fastapi         # Web dashboard
uvicorn         # ASGI server
pytest          # Testing
```

No API keys required — uses free Open-Meteo APIs.

# Phase 1 — Data Sources & Ecosystem Mapping

## Project: Greek Maritime Intelligence Platform
## Goal: Predict ferry cancellations from sea-state and wind conditions

---

## 1. Weather & Sea-State Data

### 1.1 Open-Meteo Marine Forecast API (Primary)
- **URL**: https://marine-api.open-meteo.com/v1/marine
- **Cost**: Free, no API key required
- **Coverage**: Global ocean/sea data, excellent Aegean coverage
- **Resolution**: 0.25° (~25 km) spatial, hourly temporal
- **Forecast range**: Up to 7 days ahead + 3 months historical
- **Key variables**:
  - `wave_height` — significant wave height (m)
  - `wave_direction` — dominant wave direction (°)
  - `wave_period` — dominant wave period (s)
  - `wind_wave_height` — wind-driven wave height (m)
  - `swell_wave_height` — swell wave height (m)
- **Rate limit**: ~10,000 requests/day (fair use)

### 1.2 Open-Meteo Weather Forecast API (Complementary)
- **URL**: https://api.open-meteo.com/v1/forecast
- **Cost**: Free, no API key required
- **Key variables**:
  - `wind_speed_10m` — wind at 10m height (km/h)
  - `wind_direction_10m` — wind direction (°)
  - `wind_gusts_10m` — gust speed (km/h)
  - `visibility` — horizontal visibility (m)
  - `precipitation` — rainfall (mm)

### 1.3 Open-Meteo Historical Weather API
- **URL**: https://archive-api.open-meteo.com/v1/archive
- **Cost**: Free for non-commercial use
- **Range**: 1940–present (ERA5 reanalysis)
- **Use case**: Training ML models on historical wind/wave patterns

### 1.4 EMODnet (European Marine Observation and Data Network)
- **URL**: https://emodnet.ec.europa.eu
- **Cost**: Free
- **Key datasets**: Bathymetry, sea surface temperature, wave climate
- **Use case**: Reference data for route exposure analysis

### 1.5 Copernicus Marine Service (CMEMS)
- **URL**: https://marine.copernicus.eu
- **Cost**: Free (registration required)
- **Key products**:
  - Mediterranean Sea Waves (MED-WAV): 1/24° resolution
  - Mediterranean Sea Wind (MED-WIND)
- **Use case**: High-resolution historical reanalysis for model training

---

## 2. Ferry Schedule & Cancellation Data

### 2.1 Vessel Tracking — MarineTraffic / VesselFinder
- **Source**: AIS (Automatic Identification System)
- **Approach**: Track vessel positions to detect actual sailings vs. scheduled
- **Use case**: Ground truth for cancellation detection
- **Note**: Free tier has limited historical data; premium required for bulk

### 2.2 Greek Ferry Booking Sites
- **Sources**: directferries.com, ferries.gr, openseas.gr
- **Data available**: Routes, schedules, vessel names, operators
- **Approach**: Periodic checks for schedule changes / "cancelled" flags
- **Legal**: Public information; respect robots.txt and rate limits

### 2.3 Greek Coast Guard Announcements
- **Source**: Hellenic Coast Guard (Λιμενικό Σώμα)
- **Key data**: Sailing ban announcements (απαγορευτικό απόπλου)
- **Frequency**: Issued per port when wind exceeds thresholds
- **Approach**: Monitor official announcements feeds

---

## 3. Port & Geographic Data

### 3.1 Port Coordinates & Metadata
- **Status**: ✅ Already captured in `src/config/constants.py`
- **Ports covered**: Piraeus, Rafina, Heraklion, Santorini, Mykonos, Naxos, Syros, Chania

### 3.2 Route Definitions
- **Status**: ✅ Already captured in `src/config/constants.py`
- **Routes covered**: 11 major Cyclades + Crete routes with distances, sea areas, exposure flags

---

## 4. Data Collection Priority Matrix

| Source | Priority | Difficulty | Value |
|--------|----------|------------|-------|
| Open-Meteo Marine API | **P0** | Easy | High — direct wave/wind data |
| Open-Meteo Weather API | **P0** | Easy | High — wind speed/gusts |
| Open-Meteo Historical | **P1** | Easy | High — ML training data |
| Coast Guard bans | **P1** | Medium | High — ground truth labels |
| Ferry schedule sites | **P2** | Medium | Medium — schedule context |
| Copernicus CMEMS | **P2** | Medium | Medium — high-res reanalysis |
| MarineTraffic AIS | **P3** | Hard | Medium — vessel tracking |

---

## 5. Phase 1 Architecture Decision

### Approach: Weather-first prediction
We focus on **Open-Meteo APIs** as the primary data source because:
1. Free, no auth required — fast iteration
2. Both forecast and historical data available
3. Marine-specific variables (wave height, period, swell)
4. Hourly resolution matches ferry schedules

### Prediction logic (v1 — rule-based):
```
IF wind_speed_knots >= SAILING_BAN_THRESHOLDS[vessel_type]:
    prediction = "LIKELY_CANCELLED"
ELIF wave_height >= 3.0m AND vessel_type == "high_speed":
    prediction = "AT_RISK"
ELSE:
    prediction = "EXPECTED_SAILING"
```

This rule-based baseline will later be replaced/augmented by ML models trained on historical weather + cancellation data.

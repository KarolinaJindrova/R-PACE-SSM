# R-PACE-SSM
# F1 Lap Time Decomposition: Data Preparation and Race Event Detection

This repository contains core preprocessing scripts used in the working paper:

> **Modeling Race Dynamics in Formula One: A State-Space Approach for Lap Time Decomposition**  
> Karolína Jindrová (2025)

The code prepares high-resolution race data from the FastF1 API and constructs key explanatory variables that capture situational on-track events. The final dataset is suitable for advanced econometric modeling, including Kalman filtering and state-space decomposition.

---

## 📁 Contents

### `01_prepare_race_data.py`

Performs basic race data acquisition and preprocessing for a selected session:
- Downloads laps, telemetry, weather, and car data using the FastF1 API.
- Cleans and extends lap-level data with:
  - Estimated fuel mass (based on lap number)
  - Weather conditions (merged with lap intervals)
  - Cumulative race progress
  - Safety Car (SC) and Virtual Safety Car (VSC) flags (manually specified)
  - Pit entry/exit flags and tyre codes
- Splits the dataset into individual driver CSVs and saves all files to disk.

> ✅ This script works for any race with only minor manual input required (e.g., SC/VSC lap ranges).

---

### `02_detect_race_events.py`

Implements two novel algorithms to construct situational variables that capture race dynamics:
1. **Blue Flag Disruptions:**  
   Identifies laps where a driver was lapped (blue-flagged) or overtaking slower traffic under blue flags.
   - Output: `blue_flag`, `traffic` (driver-level binary indicators)

2. **Position Battles:**  
   Detects on-track attacking/defending manoeuvres based on actual position changes between consecutive laps.
   - Filters out position gains due to other drivers' pit stops using timestamp overlap logic.
   - Output: `battle`, `attacking`, `defending`, `PositionChange`

All variables are appended to the extended lap-level dataset produced in Step 1.

---

## 🔧 Requirements

- Python ≥ 3.9
- [FastF1](https://theoehrly.github.io/Fast-F1/) ≥ 3.1
- `pandas`, `numpy`, `os`

Install via pip:
```bash
pip install fastf1 pandas numpy

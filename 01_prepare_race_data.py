# -*- coding: utf-8 -*-
"""
Race Data Preparation Script

This script uses the FastF1 API to download and process Formula 1 race data 
for a selected session. It performs the following steps:

- Loads laps, telemetry, weather, and car data for all drivers
- Merges and extends FastF1 data into a unified driver-level DataFrame
- Adds manually defined SC and VSC flags
- Computes estimated fuel mass and lap-by-lap fuel usage
- Prepares a clean dataset for modeling (e.g., lap time decomposition)
- Saves the processed dataset to disk for further use

Note:
This script performs only the basic data preparation steps. 
Additional situational variables (e.g., blue flag events, passing manoeuvres) 
are constructed in separate scripts using custom algorithms.

Works universally for any race, but can be customized for a specific event if needed.
"""

# %% --- Load required packages ---
import fastf1
import pandas as pd



# %% --- Importing the data ---


# --- USER INPUT: Define race session details ---
YEAR = 2023
LOCATION = 'Miami'
SESSION_TYPE = 'R'  # e.g., 'FP1', 'Q', 'R'


# --- Load session and extract laps ---
session = fastf1.get_session(YEAR, LOCATION, SESSION_TYPE)
session.load()

# --- Copy lap data ---
laps = session.laps.copy()


# %% --- Data information ---

# --- Display column names in the DataFrame ---
print("Available columns in laps DataFrame:")
print(laps.columns.tolist())


# --- HANDLE MISSING LAP TIMES ---
# Make a deep copy to fully isolate the cleaned dataset
laps_clean = laps.copy(deep=True)

# Identify missing LapTime values
missing_laptimes = laps_clean['LapTime'].isna()

# Compute LapTime from sector times where missing
laps_clean.loc[missing_laptimes, 'LapTime'] = (
    laps_clean.loc[missing_laptimes, 'Sector1Time'] +
    laps_clean.loc[missing_laptimes, 'Sector2Time'] +
    laps_clean.loc[missing_laptimes, 'Sector3Time']
)

# Mark which LapTimes were computed manually
laps_clean['IsComputed'] = missing_laptimes

# --- DISPLAY FOR VERIFICATION ---
print(f"Number of LapTime values computed from sectors: {missing_laptimes.sum()}")


# --- COMPOUND DISTRIBUTION REPORT ---
# Ensure all compounds are shown, even if not used in the race
compounds = ['SOFT', 'MEDIUM', 'HARD']
compound_counts = laps_clean['Compound'].value_counts().reindex(compounds, fill_value=0)

print("Number of laps by compound:")
print(compound_counts)


# %% --- Extend Lap Data with Fuel Estimates, Weather Conditions, and Race Progress ---

# --- FUNCTION: EXTEND LAPS WITH FUEL, WEATHER, AND RACE PROGRESS ---
def extend_laps_with_fuel_weather_progress(
    laps: pd.DataFrame,
    weather_data: pd.DataFrame,
    initial_fuel: float = 110.0
) -> pd.DataFrame:
    """
    Enhances the lap-level dataset with:
    - Estimated fuel level per lap
    - Merged weather conditions from FastF1 session
    - Cumulative race progress up to each lap (across all drivers)

    Parameters:
        laps (pd.DataFrame): Cleaned lap-level FastF1 data
        weather_data (pd.DataFrame): FastF1 weather data
        initial_fuel (float): Estimated starting fuel load in kg (default = 110.0)

    Returns:
        pd.DataFrame: Extended lap data
    """
    # Deep copy to preserve original data
    laps_extended = laps.copy(deep=True)

    # 1. --- FUEL LEVEL ESTIMATION ---
    max_laps = laps_extended['LapNumber'].max()
    fuel_per_lap = initial_fuel / max_laps
    laps_extended['fuel_amount'] = initial_fuel - ((laps_extended['LapNumber'] - 1) * fuel_per_lap)

    # 2. --- WEATHER MERGE (Lap-level) ---
    laps_extended['LapEndTime'] = laps_extended['LapStartTime'] + laps_extended['LapTime']
    weather_cols = ["AirTemp", "Humidity", "Pressure", "Rainfall", "TrackTemp", "WindDirection", "WindSpeed"]

    for i, lap in laps_extended.iterrows():
        weather_window = weather_data[
            (weather_data['Time'] >= lap['LapStartTime']) &
            (weather_data['Time'] <= lap['LapEndTime'])
        ]
        if not weather_window.empty:
            closest_weather = weather_window.iloc[0]
            for col in weather_cols:
                laps_extended.at[i, col] = closest_weather[col]
        else:
            for col in weather_cols:
                laps_extended.at[i, col] = pd.NA

    # 3. --- RACE PROGRESS (Cumulative Laps Started) ---
    laps_extended = laps_extended.sort_values('LapStartTime').reset_index(drop=True)
    laps_extended['cumulative_laps_completed'] = 0
    lap_start_times = laps_extended['LapStartTime']

    for i in range(len(laps_extended)):
        current_time = lap_start_times[i]
        laps_extended.at[i, 'cumulative_laps_completed'] = (lap_start_times < current_time).sum()

    return laps_extended

# --- APPLY TRANSFORMATION TO CLEAN DATASET ---
laps_extended = extend_laps_with_fuel_weather_progress(
    laps=laps_clean,
    weather_data=session.weather_data
)


# %% --- Enrich Lap Data with Time in Seconds, Pit Flags, and Tyre Code ---
def enrich_laps_with_auxiliary_info(laps: pd.DataFrame) -> None:
    """
    Adds auxiliary variables to the laps_extended DataFrame in-place:
    - LapTimeSeconds: Lap time in seconds
    - InPit: True if lap ends in pit (PitInTime not NA)
    - OutPit: True if lap starts from pit (PitOutTime not NA)
    - Tyre: Categorical tyre compound mapped to numeric code (SOFT=1, MEDIUM=2, HARD=3)
    
    Parameters:
        laps (pd.DataFrame): The laps_extended DataFrame to be modified
    """
    # --- LAP TIME IN SECONDS ---
    laps['LapTimeSeconds'] = pd.to_timedelta(laps['LapTime']).dt.total_seconds()

    # --- PIT ENTRY/EXIT FLAGS ---
    laps['InPit'] = laps['PitInTime'].notna()
    laps['OutPit'] = laps['PitOutTime'].notna()

    # --- NUMERICAL TYRE ENCODING ---
    tyre_map = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3}
    laps['Tyre'] = laps['Compound'].map(tyre_map)

# Apply the function directly to laps_extended
enrich_laps_with_auxiliary_info(laps_extended)


# %% --- Add Manual Flags for Safety Car (SC) and Virtual Safety Car (VSC) Periods ---
def flag_sc_vsc_periods(
    laps: pd.DataFrame,
    sc_periods: list[tuple[int, int]] = None,
    vsc_periods: list[tuple[int, int]] = None
) -> None:
    """
    Adds binary flags to laps DataFrame for SC and VSC periods based on manually specified lap ranges.
    Modifies the input DataFrame in-place by adding:
    - 'IsSC': True if lap falls within any SC period (inclusive)
    - 'IsVSC': True if lap falls within any VSC period (inclusive)

    Parameters:
        laps (pd.DataFrame): The laps_extended DataFrame to modify.
        sc_periods (list of tuples): Each tuple is (start_lap, end_lap) for a Safety Car phase.
        vsc_periods (list of tuples): Each tuple is (start_lap, end_lap) for a Virtual Safety Car phase.
    """
    # Default example if no input provided
    # This race (e.g. Miami 2023) had no SC/VSC periods, so we pass empty lists: sc_periods=[], vsc_periods=[]
    # If a race includes SC/VSC, you must manually specify the lap ranges using (start_lap, end_lap) tuples.
    # No default values are used — all SC/VSC flags must be defined explicitly.

    if sc_periods is None:
#        sc_periods = [(10, 12), (34, 35)]  # Example SC periods
        sc_periods = []  
    if vsc_periods is None:
#        vsc_periods = [(20, 21)]           # Example VSC period
        vsc_periods = []           

    # Initialize with False
    laps['IsSC'] = False
    laps['IsVSC'] = False

    # Apply SC flags
    for start, end in sc_periods:
        laps.loc[(laps['LapNumber'] >= start) & (laps['LapNumber'] <= end), 'IsSC'] = True

    # Apply VSC flags
    for start, end in vsc_periods:
        laps.loc[(laps['LapNumber'] >= start) & (laps['LapNumber'] <= end), 'IsVSC'] = True

# --- Apply manually defined SC/VSC flags ---
flag_sc_vsc_periods(laps_extended)


# %% --- Split Lap Data by Driver and Summarize Race Results ---
def split_laps_by_driver(laps: pd.DataFrame, session_results: pd.DataFrame) -> dict:
    """
    Splits the main laps_extended DataFrame into a dictionary of DataFrames by driver abbreviation.
    Also prints a race summary table with driver finish status and position.

    Parameters:
        laps (pd.DataFrame): The main lap-level dataset (laps_extended).
        session_results (pd.DataFrame): The FastF1 session results table (session.results).

    Returns:
        dict: Keys are driver abbreviations, values are DataFrames with that driver’s lap data.
    """
    # --- GROUP BY DRIVER ---
    drivers = laps['Driver'].unique()
    laps_by_driver = {driver: laps[laps['Driver'] == driver].copy() for driver in drivers}

    # --- MERGE RESULTS: Abbreviation → Driver, Status, Position ---
    result_summary = session_results[['Abbreviation', 'Status', 'Position']].copy()
    result_summary = result_summary.sort_values('Position', key=pd.to_numeric, na_position='last')

    print("=== Race Result Summary ===")
    print(result_summary.to_string(index=False))

    return laps_by_driver

# --- APPLY FUNCTION ---
laps_by_driver = split_laps_by_driver(laps_extended, session.results)


# %% --- Save Processed Laps Data (Full and by Driver) ---

import os

def save_processed_lap_data(laps_full: pd.DataFrame, laps_by_driver: dict, year: int, location: str, session_type: str) -> None:
    """
    Saves the processed lap data to disk:
    - laps_full is saved as a single CSV
    - laps_by_driver is saved as separate CSVs in a dedicated folder

    Parameters:
        laps_full (pd.DataFrame): The full laps_extended dataset.
        laps_by_driver (dict): Dictionary of {driver_abbreviation: driver_laps_df}.
        year (int): Race year (e.g., 2023).
        location (str): Circuit name (e.g., "Miami").
        session_type (str): Session type (e.g., "R", "Q").
    """
    # Format session info for filenames
    session_label = f"{year}_{location}_{session_type}".replace(" ", "")
    
    # Define paths
    output_dir = f"laps_output/{session_label}"
    os.makedirs(output_dir, exist_ok=True)

    # Save full extended lap dataset
    full_path = os.path.join(output_dir, f"laps_extended_{session_label}.csv")
    laps_full.to_csv(full_path, index=False)
    print(f"Saved full laps data to: {full_path}")

    # Save each driver's laps separately
    for driver, df in laps_by_driver.items():
        driver_path = os.path.join(output_dir, f"{driver}_{session_label}.csv")
        df.to_csv(driver_path, index=False)

    print(f"Saved individual driver datasets to: {output_dir}/")

# --- Call the saving function ---
save_processed_lap_data(laps_extended, laps_by_driver, YEAR, LOCATION, SESSION_TYPE)


# -*- coding: utf-8 -*-
"""
Race Event Detection Script

This script processes a selected Formula 1 race session using FastF1 data 
to identify in-race events and construct situational variables for econometric modeling. 
It implements two custom heuristic algorithms central to the paper:

- Blue Flag Detection: identifies laps where a driver received a blue flag 
  and laps where another driver was passing under blue flag conditions
- Position Battle Detection: identifies laps where a driver gained or lost 
  position through genuine on-track racing rather than pit stop timing

The script performs the following steps:

- Loads race control messages and lap-by-lap data for all drivers
- Detects blue flag moments and filters passing opportunities
- Computes position changes per driver per lap
- Filters out pit stop–induced gains to isolate real battles
- Flags laps involving blue flags, traffic, attacking, and defending
- Outputs a clean, extended dataset with new binary variables

Note:
This script implements the event detection logic described in the paper 
and is intended for direct use in regression-based modeling.

Works universally for any race, but can be adapted for focused case studies.
"""

# %% --- Importing the data ---
import fastf1
import pandas as pd

# --- USER INPUT: Define race session details ---
YEAR = 2023
LOCATION = 'Miami'
SESSION_TYPE = 'R'  # e.g., 'FP1', 'Q', 'R'

# --- Load session and extract laps ---
session = fastf1.get_session(YEAR, LOCATION, SESSION_TYPE)
session.load()

# --- Copy lap data ---
laps = session.laps.copy(deep=True)

# %% --- Identify Blue Flag Moments (Deep-Copied Session) ---
"""
This block extracts all raw blue flag events from Race Control messages.
Each event is stored with a timestamp, the flagged driver's number, and the leader's lap at the time.
These events form the basis for constructing situational variables (e.g. BlueFlagSlow) later in the analysis.
"""

import fastf1
import pandas as pd

# --- Deep copy the session so the original isn’t modified ---
session_bfe = fastf1.get_session(YEAR, LOCATION, SESSION_TYPE)
session_bfe.load()

# Make a deep copy of the race control messages DataFrame
rcm_bfe = session_bfe.race_control_messages.copy(deep=True)

# --- STEP 1: Extract raw blue flag events from Race Control messages ---
# Each event is stored as a dictionary with:
#   - Time_blueflag:       UTC timestamp of the blue flag
#   - BlueFlaggedDriver:   Driver number who received the blue flag
#   - Lap_number_leader:   Lap number of the race leader when the flag was shown
BlueFlagMoments = []

# Convert 'Time' to pandas datetime for accurate comparisons
rcm_bfe['Time'] = pd.to_datetime(rcm_bfe['Time'])

# Iterate through Race Control messages and extract all BLUE flags
for _, row in rcm_bfe.iterrows():
    if row['Flag'] == 'BLUE':
        Time_blueflag     = row['Time']
        BlueFlaggedDriver = int(row['RacingNumber'])
        Lap_number_leader = int(row['Lap'])

        BlueFlagMoments.append({
            'Time_blueflag':     Time_blueflag,
            'BlueFlaggedDriver': BlueFlaggedDriver,
            'Lap_number_leader': Lap_number_leader
        })

# Count how many blue flag events were detected
N_BFE = len(BlueFlagMoments)

# --- VERIFICATION OUTPUT ---
print(f"Identified {N_BFE} Blue Flag event(s):")
for evt in BlueFlagMoments:
    print(
        f"  • Time: {evt['Time_blueflag']}   | "
        f"Flagged Car #: {evt['BlueFlaggedDriver']}   | "
        f"Leader’s Lap: {evt['Lap_number_leader']}"
    )
    
# %% --- Identify Pit Stop Windows (Deep‐Copied Laps) ---
"""
This block extracts structured pit stop windows (in and out times) for each driver
by identifying laps with pit entries and pairing them with the following lap's exit time.
The result is a list of dictionaries used later to filter out non-race conditions
(e.g., for blue flag or position battle analysis).
"""

import pandas as pd

# --- Deep copy the laps DataFrame to avoid modifying the original session data ---
laps_pit = session_bfe.laps.copy(deep=True)

# Initialize list to store pit stop window info
PitStopWindows = []

# Track the number of pit stops for each driver using their DriverNumber as key in laps_pit
pit_counts = {}

# Group laps by driver and iterate through each group
for driver_num, driver_laps in laps_pit.groupby('DriverNumber'):
    # driver_num is the same type as laps_pit['DriverNumber'] (often a string, e.g. '1', '11', etc.)
    
    # Sort the laps chronologically by lap number
    driver_laps_sorted = driver_laps.sort_values('LapNumber')
    
    # Initialize this driver's pit stop counter
    pit_counts[driver_num] = 0
    
    # Look for laps where a pit entry occurred (PitInTime is not missing)
    for _, lap_row in driver_laps_sorted[driver_laps_sorted['PitInTime'].notna()].iterrows():
        # Keep driver_num as the key type (string or int, whatever laps_pit uses)
        LapNumberIn  = lap_row['LapNumber']    # numeric
        PitInTime    = lap_row['PitInTime']    # Timestamp
        
        # Find the next lap (LapNumberIn + 1) for this driver
        next_lap_mask = (
            (laps_pit['DriverNumber'] == driver_num) &
            (laps_pit['LapNumber'] == (LapNumberIn + 1))
        )
        next_lap = laps_pit.loc[next_lap_mask]
        
        # Extract PitOutTime if next lap exists, otherwise NaT
        if not next_lap.empty:
            PitOutTime = next_lap.iloc[0]['PitOutTime']
        else:
            PitOutTime = pd.NaT
        
        # Update pit stop counter and record the pit stop window
        pit_counts[driver_num] += 1
        PitStopNumber = pit_counts[driver_num]
        
        # Append info to PitStopWindows, converting DriverNumber to int if desired
        # (but keeping it consistent with the laps_pit type)
        PitStopWindows.append({
            'DriverNumber':   driver_num,
            'PitStopNumber':  PitStopNumber,
            'PitInTime':      PitInTime,
            'PitOutTime':     PitOutTime
        })

# --- VERIFICATION OUTPUT ---
print(f"Identified {len(PitStopWindows)} pit stop window(s):")
for ps in PitStopWindows:
    print(
        f"  • Driver #{ps['DriverNumber']} | PitStop #{ps['PitStopNumber']} | "
        f"In: {ps['PitInTime']} | Out: {ps['PitOutTime']}"
    )

# %% --- Algorithm 1 – Detection of Blue Flag Passing Events ---
"""
This block implements the first custom algorithm described in the paper:
*Algorithm 1 – Detection of Blue Flag Passing Events*.

The procedure processes each blue flag moment and identifies possible passing drivers.
It builds exclusion sets (e.g. pit lane, recent passers, cars behind), evaluates position and distance proximity,
and logs a list of detected passing drivers. The result is a list of events (PassingEvents) used for modeling
race interruptions caused by lapping.

This algorithm requires:
    – BlueFlagMoments (from Race Control messages)
    – PitStopWindows (from session data)
    – FastF1 session with positional telemetry loaded (session_bfe.pos_data)
"""


import math
import pandas as pd
from collections import defaultdict

# --- Prepare deep copies ---
laps_bfe   = session_bfe.laps.copy(deep=True)
pos_data   = session_bfe.pos_data

# Assumes you already have:
#  - BlueFlagMoments: list of {'Time_blueflag','BlueFlaggedDriver','Lap_number_leader'}
#  - PitStopWindows: list of {'DriverNumber','PitInTime','PitOutTime'} (pd.Timedelta)

# --- PARAMETERS ---
TOP_N = 5 # Number of top candidates (closest in race position) to consider
THRESHOLD_FACTOR = 1.5 # Distance ratio to accept multiple passers

# --- Storage structures ---
passed_registry = defaultdict(dict) # Track recent passes to avoid duplicates
PassingEvents   = [] # Final list of detected passing events

# --- MAIN LOOP: Process each Blue Flag event ---
for idx, evt in enumerate(BlueFlagMoments, start=1):
    print(f"\n=== Blue Flag Event {idx} ===")
    
    # --- Unpack current event ---
    bf_time   = pd.to_datetime(evt['Time_blueflag'])
    bf_driver = str(evt['BlueFlaggedDriver'])
    bf_lap    = evt['Lap_number_leader']
    print(f"Event time: {bf_time}, BF car: #{bf_driver}, Leader lap: {bf_lap}")

    # === Step 1: Get the BF driver’s current lap and position at event time ===
    df_bf = laps_bfe[laps_bfe['DriverNumber'].astype(str)==bf_driver].copy()
    df_bf['LapStartDate'] = pd.to_datetime(df_bf['LapStartDate'])
    laps_before = df_bf[df_bf['LapStartDate'] <= bf_time]
    if not laps_before.empty:
        last = laps_before.loc[laps_before['LapStartDate'].idxmax()]
        bf_position = int(last['Position'])
        bf_lap_used = int(last['LapNumber'])
    else:
        bf_position = bf_lap_used = None
    print(f"[Snippet1] BF was in P{bf_position} on lap {bf_lap_used}")

    # === Step 2a: Convert bf_time to session-relative time; detect drivers in pit ===
    leader_lap1 = laps_bfe[(laps_bfe['LapNumber']==1)&(laps_bfe['Position']==1)].iloc[0]
    session_start_abs = (pd.to_datetime(leader_lap1['LapStartDate']) - leader_lap1['LapStartTime'])
    bf_rel = bf_time - session_start_abs
    print(f"[Snippet2a] session_start_abs = {session_start_abs}, bf_rel = {bf_rel}")

    # Build set of cars in pit lane at the moment of the blue flag
    set_A_pit = set()
    for driver in laps_bfe['DriverNumber'].astype(str).unique():
        wins = [
            w for w in PitStopWindows
            if str(w['DriverNumber'])==driver and w['PitInTime'] <= bf_rel
        ]
        if wins:
            last_win = max(wins, key=lambda w: w['PitInTime'])
            if last_win['PitOutTime'] >= bf_rel:
                set_A_pit.add(driver)
    print(f"[Snippet2a] In pit lane: {sorted(set_A_pit)}")

    # === Step 2c: Get lap and position at bf_time for all drivers ===
    current_status = {}
    for driver in laps_bfe['DriverNumber'].astype(str).unique():
        df = laps_bfe[laps_bfe['DriverNumber'].astype(str)==driver].copy()
        df['LapStartDate'] = pd.to_datetime(df['LapStartDate'])
        past = df[df['LapStartDate'] <= bf_time]
        if past.empty:
            current_status[driver] = {'LapNumber': None, 'Position': None}
        else:
            row = past.loc[past['LapStartDate'].idxmax()]
            current_status[driver] = {
                'LapNumber': int(row['LapNumber']),
                'Position':  int(row['Position'])
            }
    sample = dict(list(current_status.items())[:5])
    print(f"[Snippet2c] Sample status: {sample}")

    # === Step 2b + 2d: Apply exclusions to build Set A ===
    set_A = set_A_pit.copy()
    candidates = []
    for driver, stat in current_status.items():
        if driver in set_A:
            continue
        
        # Exclude if passed the BF car within the last 3 laps
        last_pass = passed_registry[bf_driver].get(driver)
        if last_pass is not None and (bf_lap - last_pass) <= 3:
            set_A.add(driver)
            continue
        # Exclude if no current position info
        curr_pos = stat['Position']
        if curr_pos is None:
            set_A.add(driver)
            continue
        # Exclude if behind the BF driver
        if bf_position is not None and curr_pos > bf_position:
            set_A.add(driver)
            continue
        # Eligible candidate for passing
        candidates.append((driver, curr_pos))
    print(f"[Step2] Candidates after exclusions: {candidates}")

    
    # === Step 3: Keep top-N candidates (Set B), finalize Set A and Set C ===
    candidates.sort(key=lambda x: x[1]) # Sort by position (lower = ahead)
    set_B = {d for d,_ in candidates[:TOP_N]} # Closest N drivers ahead
    set_A.update(d for d,_ in candidates[TOP_N:]) # Others -> exclude
    # Set for the blue‐flagged car
    set_C = {bf_driver} # The blue-flagged car
    # Ensure BF driver isn’t in A or B
    set_A.discard(bf_driver)
    set_B.discard(bf_driver)
    print(f"[Step3] Set A: {sorted(set_A)}")
    print(f"[Step3] Set B: {sorted(set_B)}")
    print(f"[Step3] Set C (BF): {set_C}")

    
    # === Step 4: Compute spatial distances between BF car and Set B drivers ===
    df_bf_tel = pos_data[bf_driver].copy()
    df_bf_tel['Date']  = pd.to_datetime(df_bf_tel['Date'])
    df_bf_tel['Delta'] = (df_bf_tel['Date'] - bf_time).abs()
    bf_idx = df_bf_tel['Delta'].idxmin()
    x_bf, y_bf = df_bf_tel.loc[bf_idx, ['X','Y']]

    dists = []
    for driver in set_B:
        df_drv_tel = pos_data[driver].copy()
        df_drv_tel['Date']  = pd.to_datetime(df_drv_tel['Date'])
        df_drv_tel['Delta'] = (df_drv_tel['Date'] - bf_time).abs()
        idx = df_drv_tel['Delta'].idxmin()
        x_i, y_i = df_drv_tel.loc[idx, ['X','Y']]
        dist = math.hypot(x_i - x_bf, y_i - y_bf)
        dists.append((driver, dist))
    dists.sort(key=lambda x: x[1])
    distances = {d: dist for d, dist in dists}
    print(f"[Step4] Distances: {distances}")

    # === Step 5: Final decision – who passed the BF driver? ===
    passing = []
    false_positive = False
    if dists:
        closest, d0 = dists[0]
        passing.append(closest)
        top_cand = candidates[0][0] if candidates else None
        if closest != top_cand:
            false_positive = True
        if len(dists) > 1 and dists[1][1] <= THRESHOLD_FACTOR * d0:
            passing.append(dists[1][0])
    print(f"[Step5] Passing: {passing}, FalsePositive: {false_positive}")

    # === Step 6 + 7: Log and update passed registry ===
    PassingEvents.append({
        'Time_blueflag':      bf_time,
        'BlueFlaggedDriver':  bf_driver,
        'Lap_number_leader':  bf_lap,
        'SetA_excluded':      sorted(set_A),
        'SetB_potentials':    sorted(set_B),
        'SetC_flagged':       sorted(set_C),
        'Distances':          distances,
        'PassingDrivers':     passing,
        'FalsePositive':      false_positive,
        'BF_position':        bf_position
    })
    for drv in passing:
        passed_registry[bf_driver][drv] = bf_lap


# === Final output: Summary of all detected passing events ===
print("\n=== All Passing Events ===")
for ev in PassingEvents:
    print(ev)


# %% --- Plot: Blue Flag Event Positions (Universal Style) ---
"""
This visualization shows the spatial positions of all cars on track at the moment
a selected BLUE flag event occurred. The goal is to display:

    - The blue-flagged driver (Set C, shown in red),
    - Potential passing candidates (Set B, in black),
    - Excluded drivers (Set A, in gold),
    - Verstappen's full positional trace for context (light gray).

The plot uses positional telemetry (X, Y coordinates) from FastF1 and dynamically
retrieves driver abbreviations to annotate each point. This tool is useful for
manual verification of Algorithm 1 outputs and inspecting relative positions.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# --- ASSUMPTIONS ---
# - PassingEvents, session_bfe, pos_data, and results_df are preloaded

# --- Extract First Event (manually chosen by index) ---
event_index = 6 # Change this index to visualize a different BLUE flag event
evt = PassingEvents[event_index]

# Extract basic info from selected event
bf_time = evt['Time_blueflag']
bf_driver = evt['BlueFlaggedDriver']
set_A_excluded = set(evt['SetA_excluded'])
set_B_potentials = set(evt['SetB_potentials'])
set_C_flagged = set(evt['SetC_flagged'])

# Deep copy of session results to map abbreviations to driver numbers
results_df   = session_bfe.results.copy(deep=True)

# --- Dynamic Event Number (1-based) ---
event_number = event_index + 1

# --- Driver Abbreviation Mapping ---
# Dynamically detect abbreviation column (depends on FastF1 version)
if 'Abbreviation' in results_df.columns:
    abbrev_col = 'Abbreviation'
elif 'Abbrev' in results_df.columns:
    abbrev_col = 'Abbrev'
else:
    raise RuntimeError("No abbreviation column found.")

# Build mapping: DriverNumber -> Abbreviation (e.g. '1' -> 'VER')
driver_abbrev = {
    str(int(r['DriverNumber'])): r[abbrev_col]
    for _, r in results_df.iterrows()
}

# --- Verstappen’s Trace for Contextual Background ---
pos_VER = pos_data['1'].copy()
pos_VER = pos_VER[(pos_VER['X'] != 0) | (pos_VER['Y'] != 0)] # Remove zero-points

# --- Prepare all driver coordinates at the time of BLUE flag ---
points = []
for driver, df_tel in pos_data.items():
    df = df_tel.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Delta'] = (df['Date'] - bf_time).abs() # Find timestamp closest to bf_time
    idx = df['Delta'].idxmin()
    sample = df.loc[idx]
    points.append((driver, sample['X'], sample['Y']))

# --- Define known bounds for the selected track (can be adjusted per circuit) ---
min_x, max_x = -5000, 11500
min_y, max_y = -5500, 2000

# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
plt.style.use('default')

# Plot Verstappen’s full trace as light gray background path
ax.plot(
    pos_VER['X'],
    pos_VER['Y'],
    color='lightgray',
    linewidth=0.6,
    alpha=0.5,
    label="Verstappen Trace"
)

# Plot driver positions by category with colors + text labels
for driver, x, y in points:
    if driver in set_C_flagged:
        color = 'red'
    elif driver in set_B_potentials:
        color = 'black'
    else:
        color = '#FFD700'  # Gold (Set A: excluded)
    ax.scatter(
        x, y,
        color=color,
        edgecolors='k',
        s=80,
        alpha=0.9
    )
    abbrev = driver_abbrev.get(driver, driver)
    ax.text(
        x + 100, y + 100,
        abbrev,
        fontsize=10,
        fontweight='bold',
        color='black',
        va='center',
        ha='left'
    )

# --- Axes and Layout ---
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)
ax.set_aspect('equal', 'box')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

# Titles and axis labels
ax.set_title(
    f"Driver Positions at BLUE Flag Event #{event_number}\n"
#    f"(Red = BF Driver {bf_driver}, Black = Set B, Yellow = Set A)\n"
#    f"{bf_time.strftime('%Y-%m-%d %H:%M:%S')} UTC – {LOCATION} {YEAR}",
    f"{LOCATION} {YEAR}",
    fontsize=18, pad=15
)
ax.set_xlabel("Track X Coordinate (arbitrary units)", fontsize=14, labelpad=10)
ax.set_ylabel("Track Y Coordinate (arbitrary units)", fontsize=14, labelpad=10)
ax.tick_params(axis='both', labelsize=12)

# --- Build and Add Legend ---
legend_elements = [
    Line2D([0], [0], color='lightgray', linewidth=0.6, alpha=0.5, label='Verstappen Trace'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='red', markeredgecolor='k',
           markersize=8, linestyle='', label='Blue-Flagged Driver'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='black', markeredgecolor='k',
           markersize=8, linestyle='', label='Set B (Potential Passers)'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='#FFD700', markeredgecolor='k',
           markersize=8, linestyle='', label='Set A (Excluded Drivers)')
]
ax.legend(
    handles=legend_elements,
    title='Legend',
    fontsize=11, title_fontsize=12,
    loc='lower right',
    frameon=True,
    framealpha=0.95
)

plt.tight_layout()

# --- Save or show the plot ---
#plt.savefig(f"BFE_{YEAR}_{LOCATION}_event_{event_number}.pdf", bbox_inches='tight')
plt.show()

# %% --- Calculate Position Changes and Prepare Battle Dataset ---
"""
This block constructs a dataset of all lap-to-lap position changes for each driver.
It compares each lap with the previous one to identify overtaking or being overtaken,
and stores metadata such as lap timing, position, and pit info. The resulting `battle_df`
is later used for identifying potential position battles during the race.
"""

import pandas as pd

# --- Deep copy the laps dataset from the session to avoid modifying original ---
laps_battle = session.laps.copy(deep=True)

# Initialize a list to store all position change events
position_changes = []

# Loop through each driver individually
for driver_num, driver_laps in laps_battle.groupby('DriverNumber'):
    # Ensure laps are sorted in time (LapNumber) for accurate comparison
    driver_laps_sorted = driver_laps.sort_values('LapNumber').reset_index(drop=True)
    
    # Compare each lap to the previous one (start from Lap 2)
    for i in range(1, len(driver_laps_sorted)):
        current_lap = driver_laps_sorted.iloc[i]
        prev_lap = driver_laps_sorted.iloc[i - 1]
        
        # Calculate the position change between two consecutive laps
        pos_change = current_lap['Position'] - prev_lap['Position']
        
        # Only log this lap if a position change occurred
        if pos_change != 0:
            # Record relevant race data
            record = {
                'DriverNumber': driver_num,
                'LapNumber': current_lap['LapNumber'],
                'LapStartTime': current_lap['LapStartTime'],  # Already in timedelta format
                'Position': current_lap['Position'],
                'PositionChange': pos_change, # Positive = lost places, Negative = gained
                'LapTime': current_lap['LapTime'],
                'PitInTime': current_lap.get('PitInTime', pd.NaT),
                'PitOutTime': current_lap.get('PitOutTime', pd.NaT)
            }
            position_changes.append(record)

# Convert the list of dictionaries into a clean DataFrame
battle_df = pd.DataFrame(position_changes)

# Compute the end time of each lap from its start time and duration
battle_df['LapEndTime'] = battle_df['LapStartTime'] + pd.to_timedelta(battle_df['LapTime'])

# --- VERIFICATION OUTPUT ---
print(f"Constructed position change dataset with {len(battle_df)} records (position changes only).")
print(battle_df.head(10))

# %% --- Filter Out Position Changes Influenced by the Driver's Own Pit Stops ---
"""
This step filters the previously constructed dataset (`battle_df`) to exclude
any position changes that occurred during laps when the driver entered or exited
the pit lane. Such changes are typically caused by strategy, not direct on-track battles.
Resulting in DataFrame (`battle_df_filtered`).
"""

# Drop all rows where the driver either entered or exited the pit during the lap
# (PitInTime or PitOutTime is not NA -> lap was influenced by a pit stop)
battle_df_filtered = battle_df[
    battle_df['PitInTime'].isna() & battle_df['PitOutTime'].isna()
].copy()

# --- VERIFICATION OUTPUT ---
# Print the number of position change events that are unaffected by own pit stops
print(f"Filtered battle dataset: {len(battle_df_filtered)} records remain after excluding laps with pit stops.")
print(battle_df_filtered.head(10))

# %% --- Filter Out Position Gains Influenced by Other Drivers' Pit Stops ---
"""
This block removes cases where a driver appears to have gained positions,
but those gains are likely explained by other drivers entering the pit lane.
Only true on-track battles are retained — i.e., where the number of cars
in the pits during the lap is insufficient to explain the observed position gain.
"""

# Importing the required package
import numpy as np

# Deep copy to avoid modifying the original DataFrame
battle_df_pit_filtered = battle_df_filtered.copy(deep=True)

# Initialize a list to store rows that are potential on-track battles
filtered_rows = []

# Loop through each row in the dataset with position gains (PositionChange < 0)
for idx, row in battle_df_pit_filtered.iterrows():
    driver_num = row['DriverNumber']
    lap_start = row['LapStartTime']
    lap_end = row['LapEndTime']
    pos_change = row['PositionChange']
    
    
    # Focus only on laps with position gains (negative change means moving forward)
    if pos_change < 0:
        pit_count = 0 # Count how many other drivers were in the pit during this lap
        for pit in PitStopWindows:
            pit_driver = str(pit['DriverNumber'])
            # Skip own pit windows (already filtered before)
            if pit_driver == str(driver_num):
                continue
            # Check overlap between pit window and lap window
            pit_in = pit['PitInTime']
            pit_out = pit['PitOutTime']
            # Count overlap between the lap window and the pit window
            if pd.notna(pit_in) and pd.notna(pit_out):
                # Check for overlap
                if (pit_in <= lap_end) and (pit_out >= lap_start):
                    pit_count += 1
        
        # Check the filtering logic
        # If the number of drivers in the pit is greater than or equal to the absolute position gain,
        # it's likely a pit-induced position change and not an on-track battle.
        if abs(pos_change) > pit_count:
            # Keep this row as a potential battle
            filtered_rows.append(row)
    else:
        # Position losses or no change are kept as is
        filtered_rows.append(row)

# Create final filtered dataset
battle_df_final = pd.DataFrame(filtered_rows)

# --- VERIFICATION OUTPUT ---
print(f"Filtered battle dataset: {len(battle_df_final)} records remain after excluding pit-influenced position gains.")
print(battle_df_final.head(10))


# %% --- Load Saved Laps Data for Variable Creation ---
#
# This block loads the previously saved output from the lap-level preprocessing script.
# It includes:
# - The full dataset of all drivers' extended laps (`laps_extended`)
# - Individual lap datasets for each driver stored in separate CSV files
#


import os
import pandas as pd

# --- USER INPUT: Match with processing script ---
YEAR = 2023
LOCATION = 'Miami'
SESSION_TYPE = 'R'  # 'FP1', 'FP2', 'Q', 'R'

# --- Construct file path suffix ---
session_label = f"{YEAR}_{LOCATION}_{SESSION_TYPE}".replace(" ", "")
base_path = f"laps_output/{session_label}"

# --- Load full laps_extended dataset ---
laps_extended = pd.read_csv(os.path.join(base_path, f"laps_extended_{session_label}.csv"))

# --- Load individual driver datasets ---
laps_by_driver = {}
for filename in os.listdir(base_path):
    if filename.endswith(".csv") and filename.startswith("laps_extended") is False:
        driver_abbr = filename.split("_")[0]
        df = pd.read_csv(os.path.join(base_path, filename))
        laps_by_driver[driver_abbr] = df

print(f"Loaded full dataset and {len(laps_by_driver)} driver datasets from: {base_path}")


# %% --- Create blue_flag and traffic variables from PassingEvents ---
"""
This block constructs two binary variables based on previously detected blue flag events:
  - 'blue_flag': True if the driver was shown a blue flag during the lap
  - 'traffic':   True if the driver was the one passing a blue-flagged car on that lap

The variables are added to the `laps_extended` dataset and used later in modeling.
"""

# Ensure LapStartDate is datetime
laps_extended['LapStartDate'] = pd.to_datetime(laps_extended['LapStartDate'])
laps_extended['LapEndTime'] = pd.to_timedelta(laps_extended['LapTime']) + laps_extended['LapStartDate']

# Initialize new columns with False
laps_extended['blue_flag'] = False
laps_extended['traffic'] = False

# Ensure datetime
laps_extended['LapStartDate'] = pd.to_datetime(laps_extended['LapStartDate'])

# Iterate over all detected blue flag events
for evt in PassingEvents:
    bf_time = pd.to_datetime(evt['Time_blueflag'])
    bf_driver = str(evt['BlueFlaggedDriver'])
    passing_drivers = [str(d) for d in evt['PassingDrivers']]

    # FLAG 1: Mark lap of the blue-flagged driver with 'blue_flag = True'
    mask_bf = (
        (laps_extended['DriverNumber'].astype(str) == bf_driver) &
        (laps_extended['LapStartDate'] <= bf_time)
    )
    if mask_bf.any():
        last_idx = laps_extended[mask_bf]['LapStartDate'].idxmax()
        laps_extended.at[last_idx, 'blue_flag'] = True

    # FLAG 2: Mark lap of the passing drivers with 'traffic = True'
    for p_driver in passing_drivers:
        mask_p = (
            (laps_extended['DriverNumber'].astype(str) == p_driver) &
            (laps_extended['LapStartDate'] <= bf_time)
        )
        if mask_p.any():
            last_idx_p = laps_extended[mask_p]['LapStartDate'].idxmax()
            laps_extended.at[last_idx_p, 'traffic'] = True

# --- VERIFICATION ---
print("Number of blue-flagged laps:", laps_extended['blue_flag'].sum())
print("Number of traffic laps (passing under blue flag):", laps_extended['traffic'].sum())


# %% --- Add Battle, Attacking, Defending, and PositionChange Variables ---
"""
This block merges the battle dataset (position change data) with the main `laps_extended` dataset.
It creates new flags to indicate:
   - 'battle': whether a position change occurred
   - 'attacking': position gained (negative change)
   - 'defending': position lost (positive change)
These labels help capture racing dynamics for later modeling.
"""

# Ensure DriverNumber is of the same type (string) in both DataFrames
battle_df_final['DriverNumber'] = battle_df_final['DriverNumber'].astype(str)
laps_extended['DriverNumber'] = laps_extended['DriverNumber'].astype(str)

# Create merge keys for both datasets: "DriverNumber_LapNumber"
battle_df_final['merge_key'] = battle_df_final['DriverNumber'] + "_" + battle_df_final['LapNumber'].astype(str)
laps_extended['merge_key'] = laps_extended['DriverNumber'] + "_" + laps_extended['LapNumber'].astype(str)

# Create a dictionary mapping each merge key to its corresponding PositionChange
position_change_map = battle_df_final.set_index('merge_key')['PositionChange'].to_dict()

# Map PositionChange values into laps_extended using the merge key
# Missing values (non-battle laps) are set to 0
laps_extended['PositionChange'] = laps_extended['merge_key'].map(position_change_map).fillna(0).astype(int)

# Derive binary battle flags from PositionChange
laps_extended['battle'] = laps_extended['PositionChange'] != 0
laps_extended['attacking'] = laps_extended['PositionChange'] < 0  # gained position
laps_extended['defending'] = laps_extended['PositionChange'] > 0  # lost position

# Clean up temporary merge key
laps_extended.drop(columns=['merge_key'], inplace=True)

# --- VERIFICATION ---
print("Number of battle laps:", laps_extended['battle'].sum())
print("Attacking laps:", laps_extended['attacking'].sum())
print("Defending laps:", laps_extended['defending'].sum())


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

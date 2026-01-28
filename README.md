# Cricket T20I Match Predictor

ML-based predictor for T20 International cricket matches using historical match data.

## Quick Start

```bash
python predictor.py
```

## Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn

Install dependencies:
```bash
pip install pandas numpy scikit-learn
```

## Data

Thanks to https://cricsheet.org/ for collecting and formatting data. 

## Features Used

| Feature | Description |
|---------|-------------|
| `team1_win_pct` | Team 1's overall win % (time-decayed) |
| `team2_win_pct` | Team 2's overall win % (time-decayed) |
| `strength_diff` | Difference in team strengths |
| `team1_recent_form` | Team 1's win % in last 10 matches |
| `team2_recent_form` | Team 2's win % in last 10 matches |
| `h2h_team1_win_pct` | Head-to-head win % for Team 1 |
| `h2h_total_matches` | Total H2H matches played |
| `h2h_last_winner` | Who won last H2H encounter |
| `venue_team1_win_pct` | Team 1's win % at venue |
| `venue_team2_win_pct` | Team 2's win % at venue |
| `venue_avg_1st_score` | Average 1st innings score at venue |
| `venue_bat_first_win_pct` | Batting first win % at venue |
| `is_home_team1` | 1 if Team 1 is home team |
| `is_home_team2` | 1 if Team 2 is home team |
| `is_neutral` | 1 if neutral venue |
| `toss_winner_is_team1` | 1 if Team 1 won toss |
| `elected_to_bat` | 1 if toss winner chose to bat |

## Assumptions

1. **Data is pre-filtered**: All JSON files in the data directory are valid T20I matches (no league matches like IPL/BBL)

2. **Time-decay weighting**: Recent matches weighted more heavily than old ones using exponential decay (factor: 0.95)

3. **Minimum history**: Matches are skipped if either team has fewer than 5 prior matches

4. **No data leakage**: Features are computed using only matches that occurred before the match being predicted

5. **Chronological train/test split**: 80/20 split based on date, not random

6. **Home team detection**: Based on venue name keywords (e.g., "Lahore" → Pakistan home)

7. **Excluded matches**: Ties, no-results, and DLS-affected matches are excluded

## Model

- **Algorithm**: Logistic Regression

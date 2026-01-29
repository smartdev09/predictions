# Cricket T20I Match Predictor

ML-based predictor for T20 International cricket matches using historical match data.

```bash
pip install pandas numpy scikit-learn
python predictor.py
```

Thanks to https://cricsheet.org/ for collecting and formatting data. 

## Features

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

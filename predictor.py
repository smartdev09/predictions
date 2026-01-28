import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from constants import (
    DATA_PATH,
    DECAY_FACTOR,
    MIN_PRIOR_MATCHES,
    RECENT_FORM_WINDOW,
    VENUE_HOME_TEAMS,
)


def load_match_data(filepath: Path) -> dict | None:
    """Load and parse a single match JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def extract_match_info(match_data: dict) -> dict | None:
    """Extract relevant match information for feature building."""
    info = match_data.get("info", {})
    
    teams = info.get("teams", [])
    if len(teams) != 2:
        return None
    
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    
    # Skip matches with no result, ties, or DLS
    if not winner or "result" in outcome:
        return None
    
    dates = info.get("dates", [])
    if not dates:
        return None
    
    toss = info.get("toss", {})
    
    # Calculate first innings score
    innings = match_data.get("innings", [])
    first_innings_score = 0
    if innings:
        first_innings = innings[0]
        for over in first_innings.get("overs", []):
            for delivery in over.get("deliveries", []):
                first_innings_score += delivery.get("runs", {}).get("total", 0)
    
    # Determine who batted first
    batting_first = innings[0].get("team") if innings else None
    
    return {
        "date": dates[0],
        "team1": teams[0],
        "team2": teams[1],
        "winner": winner,
        "team1_won": 1 if winner == teams[0] else 0,
        "venue": info.get("venue", "Unknown"),
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "first_innings_score": first_innings_score,
        "batting_first": batting_first,
        "batting_first_won": 1 if winner == batting_first else 0
    }


def load_all_matches() -> pd.DataFrame:
    """Load all T20I matches from JSON files."""
    matches = []
    
    json_files = list(DATA_PATH.glob("*.json"))
    
    for filepath in json_files:
        match_data = load_match_data(filepath)
        if match_data:
            match_info = extract_match_info(match_data)
            if match_info:
                match_info["file_id"] = filepath.stem
                matches.append(match_info)
    
    df = pd.DataFrame(matches)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    return df


class FeatureBuilder:
    """Build features for each match using only prior match history."""
    
    def __init__(self, decay_factor: float = DECAY_FACTOR):
        self.decay_factor = decay_factor
        
        # Historical trackers
        self.team_matches = defaultdict(list)  # team -> list of (date, won, opponent)
        self.h2h_matches = defaultdict(list)   # (team1, team2) sorted -> list of (date, winner)
        self.venue_matches = defaultdict(list)  # venue -> list of (date, team1, team2, winner, bat_first, first_score, bat_first_won)
    
    def _get_decayed_win_pct(self, matches: list, team: str, as_of_date) -> float:
        """Calculate time-decayed win percentage."""
        if not matches:
            return 0.5
        
        # Filter matches before the given date
        prior_matches = [(m["date"], m["won"]) for m in matches if m["date"] < as_of_date]
        if not prior_matches:
            return 0.5
        
        # Sort by date (most recent first) and apply decay
        prior_matches.sort(key=lambda x: x[0], reverse=True)
        
        weighted_wins = 0
        total_weight = 0
        for i, (_, won) in enumerate(prior_matches):
            weight = self.decay_factor ** i
            weighted_wins += won * weight
            total_weight += weight
        
        return weighted_wins / total_weight if total_weight > 0 else 0.5
    
    def _get_recent_form(self, matches: list, as_of_date, window: int = RECENT_FORM_WINDOW) -> float:
        """Calculate recent form (last N matches with decay)."""
        prior_matches = [(m["date"], m["won"]) for m in matches if m["date"] < as_of_date]
        if not prior_matches:
            return 0.5
        
        # Sort by date and take last N
        prior_matches.sort(key=lambda x: x[0], reverse=True)
        recent = prior_matches[:window]
        
        weighted_wins = 0
        total_weight = 0
        for i, (_, won) in enumerate(recent):
            weight = self.decay_factor ** i
            weighted_wins += won * weight
            total_weight += weight
        
        return weighted_wins / total_weight if total_weight > 0 else 0.5
    
    def _get_h2h_stats(self, team1: str, team2: str, as_of_date) -> dict:
        """Get head-to-head statistics."""
        # Use sorted key for consistent lookup
        key = tuple(sorted([team1, team2]))
        matches = self.h2h_matches[key]
        
        prior = [m for m in matches if m["date"] < as_of_date]
        if not prior:
            return {
                "h2h_team1_win_pct": 0.5,
                "h2h_total_matches": 0,
                "h2h_last_winner": 0.5
            }
        
        # Calculate decayed win pct for team1
        prior.sort(key=lambda x: x["date"], reverse=True)
        
        weighted_wins = 0
        total_weight = 0
        for i, m in enumerate(prior):
            weight = self.decay_factor ** i
            if m["winner"] == team1:
                weighted_wins += weight
            total_weight += weight
        
        h2h_win_pct = weighted_wins / total_weight if total_weight > 0 else 0.5
        
        # Last winner
        last_winner = prior[0]["winner"]
        h2h_last = 1 if last_winner == team1 else 0
        
        return {
            "h2h_team1_win_pct": h2h_win_pct,
            "h2h_total_matches": len(prior),
            "h2h_last_winner": h2h_last
        }
    
    def _get_venue_stats(self, venue: str, team1: str, team2: str, as_of_date) -> dict:
        """Get venue statistics."""
        matches = self.venue_matches[venue]
        prior = [m for m in matches if m["date"] < as_of_date]
        
        if not prior:
            return {
                "venue_team1_win_pct": 0.5,
                "venue_team2_win_pct": 0.5,
                "venue_avg_1st_score": 150,
                "venue_bat_first_win_pct": 0.5
            }
        
        # Team1 win % at venue
        team1_venue = [m for m in prior if team1 in (m["team1"], m["team2"])]
        if team1_venue:
            team1_wins = sum(1 for m in team1_venue if m["winner"] == team1)
            team1_venue_pct = team1_wins / len(team1_venue)
        else:
            team1_venue_pct = 0.5
        
        # Team2 win % at venue
        team2_venue = [m for m in prior if team2 in (m["team1"], m["team2"])]
        if team2_venue:
            team2_wins = sum(1 for m in team2_venue if m["winner"] == team2)
            team2_venue_pct = team2_wins / len(team2_venue)
        else:
            team2_venue_pct = 0.5
        
        # Average first innings score
        scores = [m["first_innings_score"] for m in prior if m["first_innings_score"] > 0]
        avg_score = np.mean(scores) if scores else 150
        
        # Batting first win %
        bat_first_wins = sum(m["bat_first_won"] for m in prior)
        bat_first_pct = bat_first_wins / len(prior)
        
        return {
            "venue_team1_win_pct": team1_venue_pct,
            "venue_team2_win_pct": team2_venue_pct,
            "venue_avg_1st_score": avg_score,
            "venue_bat_first_win_pct": bat_first_pct
        }
    
    def _determine_home_team(self, venue: str, team1: str, team2: str) -> dict:
        """Determine home/away/neutral status based on venue."""
        venue_lower = venue.lower()
        
        home_team = None
        for keyword, team in VENUE_HOME_TEAMS.items():
            if keyword in venue_lower:
                home_team = team
                break
        
        is_home_team1 = 1 if home_team == team1 else 0
        is_home_team2 = 1 if home_team == team2 else 0
        is_neutral = 1 if home_team is None or (home_team != team1 and home_team != team2) else 0
        
        return {
            "is_home_team1": is_home_team1,
            "is_home_team2": is_home_team2,
            "is_neutral": is_neutral
        }
    
    def build_features(self, match: pd.Series) -> dict | None:
        """Build all features for a single match using prior history."""
        team1 = match["team1"]
        team2 = match["team2"]
        date = match["date"]
        venue = match["venue"]
        
        # Check minimum history requirement
        team1_prior = len([m for m in self.team_matches[team1] if m["date"] < date])
        team2_prior = len([m for m in self.team_matches[team2] if m["date"] < date])
        
        if team1_prior < MIN_PRIOR_MATCHES or team2_prior < MIN_PRIOR_MATCHES:
            return None
        
        features = {}
        
        # Team strength features
        features["team1_win_pct"] = self._get_decayed_win_pct(self.team_matches[team1], team1, date)
        features["team2_win_pct"] = self._get_decayed_win_pct(self.team_matches[team2], team2, date)
        features["strength_diff"] = features["team1_win_pct"] - features["team2_win_pct"]
        
        features["team1_recent_form"] = self._get_recent_form(self.team_matches[team1], date)
        features["team2_recent_form"] = self._get_recent_form(self.team_matches[team2], date)
        features["form_diff"] = features["team1_recent_form"] - features["team2_recent_form"]
        
        # H2H features
        h2h_stats = self._get_h2h_stats(team1, team2, date)
        features.update(h2h_stats)
        
        # Venue features
        venue_stats = self._get_venue_stats(venue, team1, team2, date)
        features.update(venue_stats)
        
        # Home/away features
        home_stats = self._determine_home_team(venue, team1, team2)
        features.update(home_stats)
        
        # Toss features
        features["toss_winner_is_team1"] = 1 if match["toss_winner"] == team1 else 0
        features["elected_to_bat"] = 1 if match["toss_decision"] == "bat" else 0
        
        # Target variable
        features["team1_won"] = match["team1_won"]
        
        # Metadata (not for training)
        features["_team1"] = team1
        features["_team2"] = team2
        features["_date"] = date
        features["_venue"] = venue
        
        return features
    
    def update_history(self, match: pd.Series):
        """Update historical trackers after processing a match."""
        team1 = match["team1"]
        team2 = match["team2"]
        date = match["date"]
        winner = match["winner"]
        venue = match["venue"]
        
        # Update team matches
        self.team_matches[team1].append({
            "date": date,
            "won": 1 if winner == team1 else 0,
            "opponent": team2
        })
        self.team_matches[team2].append({
            "date": date,
            "won": 1 if winner == team2 else 0,
            "opponent": team1
        })
        
        # Update H2H
        key = tuple(sorted([team1, team2]))
        self.h2h_matches[key].append({
            "date": date,
            "winner": winner
        })
        
        # Update venue
        self.venue_matches[venue].append({
            "date": date,
            "team1": team1,
            "team2": team2,
            "winner": winner,
            "bat_first": match["batting_first"],
            "first_innings_score": match["first_innings_score"],
            "bat_first_won": match["batting_first_won"]
        })


def build_training_data(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from all matches."""
    builder = FeatureBuilder()
    features_list = []
    
    for idx, match in matches_df.iterrows():
        # Build features using only prior history
        features = builder.build_features(match)
        if features:
            features_list.append(features)
        
        # Update history for next iteration
        builder.update_history(match)
    
    features_df = pd.DataFrame(features_list)
    
    return features_df, builder


def train_model(features_df: pd.DataFrame):
    """Train the prediction model."""
    feature_cols = [c for c in features_df.columns if not c.startswith("_") and c != "team1_won"]
    
    X = features_df[feature_cols].values
    y = features_df["team1_won"].values
    
    # Chronological split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, feature_cols


def predict_match(model, scaler, feature_cols: list, builder: FeatureBuilder,
                  team1: str, team2: str, venue: str, 
                  toss_winner: str = None, toss_decision: str = None) -> dict:
    """Predict outcome of a specific match."""
    
    # Use current date as reference
    current_date = pd.Timestamp.now()
    
    # Build features
    features = {}
    
    # Team strength features
    features["team1_win_pct"] = builder._get_decayed_win_pct(builder.team_matches[team1], team1, current_date)
    features["team2_win_pct"] = builder._get_decayed_win_pct(builder.team_matches[team2], team2, current_date)
    features["strength_diff"] = features["team1_win_pct"] - features["team2_win_pct"]
    
    features["team1_recent_form"] = builder._get_recent_form(builder.team_matches[team1], current_date)
    features["team2_recent_form"] = builder._get_recent_form(builder.team_matches[team2], current_date)
    features["form_diff"] = features["team1_recent_form"] - features["team2_recent_form"]
    
    # H2H features
    h2h_stats = builder._get_h2h_stats(team1, team2, current_date)
    features.update(h2h_stats)
    
    # Venue features
    venue_stats = builder._get_venue_stats(venue, team1, team2, current_date)
    features.update(venue_stats)
    
    # Home/away features
    home_stats = builder._determine_home_team(venue, team1, team2)
    features.update(home_stats)
    
    # Toss features (use defaults if not specified)
    if toss_winner:
        features["toss_winner_is_team1"] = 1 if toss_winner == team1 else 0
    else:
        features["toss_winner_is_team1"] = 0.5  # Unknown
    
    if toss_decision:
        features["elected_to_bat"] = 1 if toss_decision == "bat" else 0
    else:
        features["elected_to_bat"] = 0.5  # Unknown
    
    # Create feature vector
    X = np.array([[features[col] for col in feature_cols]])
    X_scaled = scaler.transform(X)
    
    # Predict
    prob = model.predict_proba(X_scaled)[0]
    
    return {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "team1_win_prob": prob[1],
        "team2_win_prob": prob[0],
        "predicted_winner": team1 if prob[1] > 0.5 else team2,
        "confidence": max(prob),
        "features": features
    }


def main():
    """Main execution function."""
    matches_df = load_all_matches()
    features_df, builder = build_training_data(matches_df)
    model, scaler, feature_cols = train_model(features_df)
    
    pred = predict_match(
        model, scaler, feature_cols, builder,
        team1="Pakistan",
        team2="Australia", 
        venue="Gaddafi Stadium, Lahore"
    )
    
    print(f"Pakistan win probability: {pred['team1_win_prob']:.1%}")
    print(f"Australia win probability: {pred['team2_win_prob']:.1%}")


if __name__ == "__main__":
    main()

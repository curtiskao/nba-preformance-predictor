# process_data.py

"""
Processes player game logs into ML-ready features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import centralized NBA API client
import nba_api_client

from nba_api.stats.endpoints import (
    playergamelog,
    leaguedashteamstats
)
from nba_api.stats.static import players


# -----------------------------------------------------
# Determine NBA season
# -----------------------------------------------------
def get_current_season():
    year = datetime.now().year
    month = datetime.now().month
    if month >= 10:
        return f"{year}-{str(year+1)[-2:]}"
    return f"{year-1}-{str(year)[-2:]}"

def get_last_nba_season():
    """
    Returns the previous NBA season as a string in 'YYYY-YY' format.
    Example: If current season is 2025-26, returns '2024-25'
    """
    now = datetime.now()
    year = now.year
    month = now.month

    # NBA season typically starts in October
    if month >= 10:
        start_year = year - 1
        end_year = year
    else:
        start_year = year - 2
        end_year = year - 1

    return f"{start_year}-{str(end_year)[-2:]}"


# -----------------------------------------------------
# Load Team DEF Rating + Pace
# -----------------------------------------------------
def load_team_stats(season: str):
    print("Loading team advanced stats...")

    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Advanced"
    ).get_data_frames()[0]

    result = stats[["TEAM_ID", "TEAM_NAME", "DEF_RATING", "PACE"]]
    return result


# -----------------------------------------------------
# Fetch player logs
# -----------------------------------------------------
def fetch_player_logs(player_name, season):
    matches = players.find_players_by_full_name(player_name)
    if not matches:
        raise ValueError(f"No player found with name {player_name}")

    player_id = matches[0]["id"]

    print(f"Fetching game logs for {player_name}...")

    df = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season
    ).get_data_frames()[0]

    print(f"Loaded {len(df)} games.")
    return df


# -----------------------------------------------------
# Feature engineering
# -----------------------------------------------------
def engineer_features(df, team_stats):
    df = df.copy()

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE")

    df = df.merge(
        team_stats,
        left_on="MATCHUP_OPPONENT",
        right_on="TEAM_NAME",
        how="left"
    )

    df.rename(columns={
        "DEF_RATING": "OPP_DEF_RATING",
        "PACE": "OPP_PACE"
    }, inplace=True)

    df.drop(columns=["TEAM_NAME"], inplace=True)

    df["PTS_last3"] = df["PTS"].rolling(3).mean()
    df["REB_last3"] = df["REB"].rolling(3).mean()
    df["AST_last3"] = df["AST"].rolling(3).mean()

    df["Is_Home"] = df["MATCHUP"].str.contains("vs").astype(int)

    df["Prev_Date"] = df["GAME_DATE"].shift(1)
    df["Days_Rest"] = (df["GAME_DATE"] - df["Prev_Date"]).dt.days
    df["Is_BackToBack"] = (df["Days_Rest"] == 1).astype(int)

    df.drop(columns=["Prev_Date"], inplace=True)

    return df


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def process_player_data(player_name, season):
    team_stats = load_team_stats(season)
    logs = fetch_player_logs(player_name, season)

    logs["MATCHUP_OPPONENT"] = logs["MATCHUP"].apply(lambda x: x.split(" ")[-1])

    df = engineer_features(logs, team_stats)

    out = f"output/{player_name}_processed.csv"
    df.to_csv(out, index=False)

    print(f"Saved {out}")


if __name__ == "__main__":
    season = get_current_season()
    player = "Devin Booker"

    print(f"Processing data for {player} ({season})...")
    process_player_data(player, season)
    print("Done.")
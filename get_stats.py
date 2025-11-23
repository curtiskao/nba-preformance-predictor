# get_stats.py
"""
Fetches NBA data from the API - player logs and team stats.
"""

import pandas as pd
from datetime import datetime

# Import patched NBA API client
import nba_api_client

from nba_api.stats.endpoints import (
    playergamelog,
    playergamelogs,
    leaguedashteamstats
)
from nba_api.stats.static import players


# -----------------------------------------------------
# Season helpers
# -----------------------------------------------------
def get_current_season():
    """Returns current NBA season as 'YYYY-YY' format."""
    year = datetime.now().year
    month = datetime.now().month
    if month >= 10:
        return f"{year}-{str(year+1)[-2:]}"
    return f"{year-1}-{str(year)[-2:]}"


def get_last_season():
    """Returns previous NBA season as 'YYYY-YY' format."""
    now = datetime.now()
    year = now.year
    month = now.month
    
    if month >= 10:
        start_year = year - 1
        end_year = year
    else:
        start_year = year - 2
        end_year = year - 1
    
    return f"{start_year}-{str(end_year)[-2:]}"


# -----------------------------------------------------
# Fetch all player logs for a season
# -----------------------------------------------------
def fetch_all_player_logs(season=None):
    """Fetch game logs for all players in a season."""
    if season is None:
        season = get_current_season()
    
    print(f"Fetching all player logs for season: {season}")
    
    logs = playergamelogs.PlayerGameLogs(
        season_nullable=season
    ).get_data_frames()[0]
    
    print(f"Retrieved {len(logs)} game logs.")
    return logs


# -----------------------------------------------------
# Fetch logs for a specific player
# -----------------------------------------------------
def fetch_player_logs(player_name, season=None):
    """Fetch game logs for a specific player."""
    if season is None:
        season = get_current_season()
    
    print(f"Fetching logs for {player_name} ({season})...")
    
    # Find player
    matches = players.find_players_by_full_name(player_name)
    if not matches:
        raise ValueError(f"Player not found: {player_name}")
    
    player_id = matches[0]["id"]
    
    # Fetch logs
    df = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season
    ).get_data_frames()[0]
    
    print(f"Retrieved {len(df)} games for {player_name}")
    return df


# -----------------------------------------------------
# Fetch team advanced stats (UPDATED)
# -----------------------------------------------------
def fetch_team_stats(season=None):
    """Fetch defensive rating, offensive rating, net rating, and pace for all teams."""
    if season is None:
        season = get_current_season()
    
    print(f"Fetching team stats for season: {season}")
    
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Advanced"  # This gets advanced stats
    ).get_data_frames()[0]
    
    # Keep only needed columns - now including OFF_RATING and NET_RATING
    required_cols = ["TEAM_ID", "TEAM_NAME", "DEF_RATING", "OFF_RATING", "NET_RATING", "PACE"]
    
    # Check which columns exist
    available_cols = [col for col in required_cols if col in stats.columns]
    missing_cols = [col for col in required_cols if col not in stats.columns]
    
    if missing_cols:
        print(f"⚠️  Warning: Missing columns: {missing_cols}")
        print(f"Available columns: {stats.columns.tolist()}")
    
    result = stats[available_cols].copy()
    
    print(f"Retrieved stats for {len(result)} teams with columns: {available_cols}")
    return result


# -----------------------------------------------------
# Main - for testing
# -----------------------------------------------------
if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    
    season = get_current_season()
    print(f"Current season: {season}\n")
    
    # Test: Fetch all logs
    all_logs = fetch_all_player_logs(season)
    all_logs.to_csv("output/nba_game_logs.csv", index=False)
    print(f"Saved all logs: {len(all_logs)} rows\n")
    
    # Test: Fetch specific player
    player = "Devin Booker"
    player_logs = fetch_player_logs(player, season)
    player_logs.to_csv(f"output/{player}_raw.csv", index=False)
    print(f"Saved {player} logs: {len(player_logs)} games\n")
    
    # Test: Fetch team stats
    team_stats = fetch_team_stats(season)
    team_stats.to_csv("output/team_stats.csv", index=False)
    print(f"\nTeam stats columns: {team_stats.columns.tolist()}")
    print(f"Sample data:\n{team_stats.head()}\n")
    
    print("Done.")
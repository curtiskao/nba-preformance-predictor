# get_stats.py
"""
Fetches NBA data from the API - player logs and team stats.
Multi-season support is built-in for better model training.
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
# Fetch logs for a specific player (single season)
# -----------------------------------------------------
def fetch_player_logs(player_name, season=None):
    """Fetch game logs for a specific player in one season."""
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
# Fetch logs for multiple seasons (DEFAULT METHOD)
# -----------------------------------------------------
def fetch_player_logs_multi_season(player_name, num_seasons=3):
    """
    Fetch game logs for a player across multiple seasons.
    This is the recommended way to fetch data for better model accuracy.
    
    Args:
        player_name (str): Full player name
        num_seasons (int): Number of recent seasons to fetch (default: 3)
    
    Returns:
        pd.DataFrame: Combined game logs from all seasons
    """
    # Auto-generate last N seasons
    seasons = []
    current = get_current_season()
    current_year = int(current.split('-')[0])
    
    for i in range(num_seasons):
        year = current_year - i
        season_str = f"{year}-{str(year+1)[-2:]}"
        seasons.append(season_str)
    
    seasons.reverse()  # Chronological order
    
    print(f"\n{'='*60}")
    print(f"Fetching {num_seasons}-season data for {player_name}")
    print(f"Seasons: {', '.join(seasons)}")
    print(f"{'='*60}")
    
    all_logs = []
    
    for season in seasons:
        try:
            logs = fetch_player_logs(player_name, season)
            if len(logs) > 0:
                all_logs.append(logs)
                print(f"  ✓ {season}: {len(logs)} games")
            else:
                print(f"  ⚠ {season}: No games found")
        except Exception as e:
            print(f"  ✗ {season}: Error - {str(e)}")
    
    if not all_logs:
        raise ValueError(f"No data found for {player_name} in any season")
    
    # Combine all seasons
    combined = pd.concat(all_logs, ignore_index=True)
    
    # Sort by date
    combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"])
    combined = combined.sort_values("GAME_DATE").reset_index(drop=True)
    
    print(f"{'='*60}")
    print(f"✓ Total games: {len(combined)}")
    print(f"  Date range: {combined['GAME_DATE'].min().date()} to {combined['GAME_DATE'].max().date()}")
    print(f"{'='*60}\n")
    
    return combined


# -----------------------------------------------------
# Fetch team advanced stats
# -----------------------------------------------------
def fetch_team_stats(season=None):
    """Fetch defensive rating, offensive rating, net rating, and pace for all teams."""
    if season is None:
        season = get_current_season()
    
    print(f"Fetching team stats for season: {season}")
    
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Advanced"
    ).get_data_frames()[0]
    
    # Keep only needed columns
    required_cols = ["TEAM_ID", "TEAM_NAME", "DEF_RATING", "OFF_RATING", "NET_RATING", "PACE"]
    
    # Check which columns exist
    available_cols = [col for col in required_cols if col in stats.columns]
    missing_cols = [col for col in required_cols if col not in stats.columns]
    
    if missing_cols:
        print(f"⚠️  Warning: Missing columns: {missing_cols}")
    
    result = stats[available_cols].copy()
    
    print(f"Retrieved stats for {len(result)} teams")
    return result


# -----------------------------------------------------
# Main - for testing
# -----------------------------------------------------
if __name__ == "__main__":
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch NBA data")
    parser.add_argument("--player", type=str, default="Devin Booker")
    parser.add_argument("--seasons", type=int, default=3)
    args = parser.parse_args()
    
    os.makedirs("output", exist_ok=True)
    
    # Fetch multi-season data
    player_logs = fetch_player_logs_multi_season(args.player, num_seasons=args.seasons)
    player_logs.to_csv(f"output/{args.player}_raw.csv", index=False)
    
    # Fetch team stats
    team_stats = fetch_team_stats()
    team_stats.to_csv("output/team_stats.csv", index=False)
    
    print(f"\n✓ Saved data to output/ folder")
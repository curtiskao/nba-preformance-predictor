# get_stats.py
"""
Pulls NBA player game logs with a centralized, reliable API client.
"""

import time
import requests
import pandas as pd
from datetime import datetime

# Import auto-patched NBA API client
import nba_api_client

from nba_api.stats.endpoints import playergamelogs, playergamelog
from nba_api.stats.static import players


# -----------------------------------------------------
# Helper: Determine NBA season string, e.g., "2025-26"
# -----------------------------------------------------
def get_current_season():
    year = datetime.now().year
    month = datetime.now().month
    if month >= 10:
        return f"{year}-{str(year+1)[-2:]}"
    else:
        return f"{year-1}-{str(year)[-2:]}"


# -----------------------------------------------------
# Fetch ALL player logs
# -----------------------------------------------------
def fetch_current_season_logs():
    season_str = get_current_season()
    print(f"Fetching NBA game logs for season: {season_str}")

    logs = playergamelogs.PlayerGameLogs(
        season_nullable=season_str
    ).get_data_frames()[0]

    logs.to_csv("output/nba_game_logs.csv", index=False)
    print(f"Saved nba_game_logs.csv with {len(logs)} rows.")
    return logs


# -----------------------------------------------------
# Fetch logs for one player
# -----------------------------------------------------
def fetch_player_logs(player_name):
    print(f"Fetching logs for: {player_name}")

    matches = players.find_players_by_full_name(player_name)
    if not matches:
        raise ValueError(f"Player not found: {player_name}")

    player_id = matches[0]["id"]

    attempts = 0
    while attempts < 6:
        try:
            df = playergamelog.PlayerGameLog(
                player_id=player_id
            ).get_data_frames()[0]
            print(f"Retrieved {len(df)} games for {player_name}")
            return df
        except Exception as e:
            attempts += 1
            print(f"Error (attempt {attempts}/6): {e}")
            time.sleep(1.3)

    raise Exception("Failed to fetch player logs after retries.")


if __name__ == "__main__":
    logs = fetch_current_season_logs()

    player = "Devin Booker"
    try:
        df = fetch_player_logs(player)
        df.to_csv(f"output/{player}.csv", index=False)
        print("Saved player logs.")
    except Exception as e:
        print("Error fetching player logs:", e)

    print("Done.")
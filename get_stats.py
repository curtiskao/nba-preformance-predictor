"""
get_stats.py
-------------
Pulls full NBA player game logs for the current season
and safely fetches individual player logs using nba_api.

Handles NBA Stats API rate limits and JSONDecodeError with retries.
"""

import time
import requests
import pandas as pd
from datetime import datetime

from nba_api.stats.endpoints import playergamelogs, playergamelog
from nba_api.stats.static import players
from nba_api.stats.library.http import NBAStatsHTTP


# -----------------------------------------------------
# PATCH: Add retry logic + user-agent header
# -----------------------------------------------------
class PatchedHTTP(NBAStatsHTTP):
    def send_request(self):
        # Spoof browser header to avoid blocks
        self.session.headers.update({
            "User-Agent":
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        })

        for attempt in range(8):  # up to 8 retries
            try:
                return super().send_request()
            except requests.exceptions.RequestException:
                print(f"Request failed (attempt {attempt + 1}/8) — retrying...")
                time.sleep(1.2)

        raise Exception("Failed after multiple retries.")


# Apply monkey patch globally
NBAStatsHTTP.send_request = PatchedHTTP.send_request


# -----------------------------------------------------
# Helper: Determine NBA season string, e.g. "2025-26"
# -----------------------------------------------------
def get_current_season():
    year = datetime.now().year
    month = datetime.now().month

    if month >= 10:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"


# -----------------------------------------------------
# Fetch ALL player game logs for season
# -----------------------------------------------------
def fetch_current_season_logs():
    season_str = get_current_season()
    print(f"Fetching NBA game logs for season: {season_str}")

    # Safe fetch
    logs = playergamelogs.PlayerGameLogs(
        season_nullable=season_str
    ).get_data_frames()[0]

    # Save to CSV
    logs.to_csv("output/nba_game_logs.csv", index=False)
    print(f"Saved nba_game_logs.csv with {len(logs)} rows.")

    return logs


# -----------------------------------------------------
# Fetch game logs for a single player by name
# -----------------------------------------------------
def fetch_player_logs(player_name):
    print(f"Fetching logs for: {player_name}")

    # Find player ID
    matches = players.find_players_by_full_name(player_name)

    if len(matches) == 0:
        raise ValueError(f"Player not found: {player_name}")

    player_id = matches[0]["id"]

    # Attempt request with retries
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
            print(f"rror fetching logs (attempt {attempts}/6): {e}")
            time.sleep(1.2)

    raise Exception("Failed to fetch player logs after retries.")


if __name__ == "__main__":
    # Pull full season logs
    logs = fetch_current_season_logs()

    player_name = "Devin Booker"
    try:
        player_log = fetch_player_logs(player_name)
        player_log.to_csv("output/"+player_name +".csv", index=False)
        print("Saved book.csv")
    except Exception as e:
        print("Could not fetch individual player logs:", e)

    print("Done.")
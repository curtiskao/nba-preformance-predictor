# nba_api_client.py
import time
from nba_api.stats.library.http import NBAStatsHTTP

class ReliableNBAHTTP(NBAStatsHTTP):
    """
    A more reliable HTTP client that retries failed requests automatically.
    """

    def send_api_request(self, *args, **kwargs):
        max_retries = 5
        delay = 1

        for attempt in range(max_retries):
            try:
                return super().send_api_request(*args, **kwargs)
            except Exception as e:
                print(f"[NBA_API] Error: {e} (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
                delay *= 2  # exponential backoff

        raise RuntimeError("Failed to reach NBA Stats API after retries.")

# --- PATCH THE CLIENT USED BY ALL nba_api ENDPOINTS ---
from nba_api.stats.library import http
http.NBAStatsHTTP = ReliableNBAHTTP

print("[NBA_API] Patched HTTP client initialized.")
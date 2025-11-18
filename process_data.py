"""
process_data.py
----------------
Loads an individual player's game log CSV and performs simple
feature engineering to prepare data for model training.
"""

import pandas as pd
import os


# -----------------------------------------------------
# Clean and engineer features for a single player
# -----------------------------------------------------
def process_player_log(player_name, window=5):
    """
    Loads `output/<player_name>.csv`, cleans it,
    and adds simple rolling features.

    Args:
        player_name (str): Player full name, e.g. "Devin Booker"
        window (int): Rolling window size for averages (default 5)
    """
    file_path = f"output/{player_name}.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"No CSV found for {player_name}. "
            f"Run get_stats.py first."
        )

    # Load raw data
    df = pd.read_csv(file_path)

    # Sort by game date (ascending)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # Basic features from playergamelog:
    # PTS, REB, AST, STL, BLK, TOV, FGM, FGA, FG3M, FG3A, FT%, etc.
    basic_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV", 
                  "FGM", "FGA", "FG3M", "FG3A", "FTA", "FTM", "OREB", "DREB"]

    # Keep only relevant columns + date
    df_basic = df[["GAME_DATE"] + basic_cols].copy()

    # Rolling averages
    for col in basic_cols:
        df_basic[f"{col}_avg_last_{window}"] = df_basic[col].rolling(window).mean()

    # Target variable (next game’s points)
    df_basic["PTS_next_game"] = df_basic["PTS"].shift(-1)

    # Drop final row (no next game available)
    df_clean = df_basic.dropna().reset_index(drop=True)

    # Save processed dataset
    out_path = f"output/processed_{player_name}.csv"
    df_clean.to_csv(out_path, index=False)

    print(f"Processed file saved to: {out_path}")
    print(f"Rows: {len(df_clean)}")
    return df_clean


if __name__ == "__main__":
    player_name = "Devin Booker"
    process_player_log(player_name)
# process_data.py
"""
Processes raw NBA data into ML-ready features.
Assumes data has already been fetched via get_stats.py
"""

import pandas as pd
import numpy as np
import argparse
import os
import util

# -----------------------------------------------------
# Feature engineering pipeline
# -----------------------------------------------------
def engineer_features(player_logs, team_stats):
    """Enhanced feature engineering with more predictive features."""
    df = player_logs.copy()
    
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    
    # Opponent matching
    df["OPP_ABBREV"] = df["MATCHUP"].apply(lambda x: x.split(" ")[-1])
    df["OPP_TEAM_NAME"] = df["OPP_ABBREV"].map(util.NBA_TEAM_ABBREV_TO_NAME)
    
    df = df.merge(team_stats, left_on="OPP_TEAM_NAME", right_on="TEAM_NAME", how="left")
    df.rename(columns={
        "DEF_RATING": "OPP_DEF_RATING",
        "OFF_RATING": "OPP_OFF_RATING",
        "NET_RATING": "OPP_NET_RATING",
        "PACE": "OPP_PACE"
    }, inplace=True)
    df.drop(columns=["TEAM_NAME", "OPP_ABBREV"], inplace=True, errors="ignore")
    
    # ========================================
    # ROLLING AVERAGES (with min_periods=1 to avoid NaN)
    # ========================================
    # Use only 3-game window for limited data
    df["PTS_last3"] = df["PTS"].shift(1).rolling(3, min_periods=1).mean()
    df["MIN_last3"] = df["MIN"].shift(1).rolling(3, min_periods=1).mean()
    df["FGA_last3"] = df["FGA"].shift(1).rolling(3, min_periods=1).mean()
    df["REB_last3"] = df["REB"].shift(1).rolling(3, min_periods=1).mean()
    df["AST_last3"] = df["AST"].shift(1).rolling(3, min_periods=1).mean()
    df["FG_PCT_last3"] = df["FG_PCT"].shift(1).rolling(3, min_periods=1).mean()
    
    # Fill first game with player's first game stats as baseline
    for col in ["PTS_last3", "MIN_last3", "FGA_last3", "REB_last3", "AST_last3", "FG_PCT_last3"]:
        base_col = col.replace("_last3", "")
        df[col] = df[col].fillna(df[base_col])
    
    # ========================================
    # HOME/AWAY
    # ========================================
    df["Is_Home"] = df["MATCHUP"].str.contains("vs").astype(int)
    
    # ========================================
    # REST DAYS
    # ========================================
    df["Prev_Date"] = df["GAME_DATE"].shift(1)
    df["Days_Rest"] = (df["GAME_DATE"] - df["Prev_Date"]).dt.days
    # Fill first game with 3 days rest (typical)
    df["Days_Rest"] = df["Days_Rest"].fillna(3)
    df["Is_BackToBack"] = (df["Days_Rest"] == 1).astype(int)
    
    # Cleanup
    df.drop(columns=["Prev_Date"], inplace=True, errors="ignore")
    
    return df


# -----------------------------------------------------
# Complete processing pipeline
# -----------------------------------------------------
def process_player_data(player_name, player_logs, team_stats):
    """
    Complete processing pipeline for a player.
    Returns processed dataframe ready for ML.
    """
    print(f"Processing data for {player_name}...")
    print(f"Input: {len(player_logs)} games")
    
    df = engineer_features(player_logs, team_stats)
    
    # Show stats before dropping
    print(f"\nFeature completeness:")
    print(f"  - Valid OPP_DEF_RATING: {df['OPP_DEF_RATING'].notna().sum()}/{len(df)}")
    print(f"  - Valid PTS_last3: {df['PTS_last3'].notna().sum()}/{len(df)}")
    
    # Only drop rows missing opponent stats (critical for predictions)
    initial_rows = len(df)
    df = df.dropna(subset=["OPP_DEF_RATING"])
    
    print(f"\nOutput: {len(df)} games (removed {initial_rows - len(df)} incomplete rows)")
    
    return df


# -----------------------------------------------------
# Main - for testing
# -----------------------------------------------------
if __name__ == "__main__":
    from get_stats import fetch_player_logs, fetch_team_stats, get_current_season
    
    parser = argparse.ArgumentParser(
        description="Process NBA player data for ML training"
    )
    parser.add_argument(
        "--player",
        type=str,
        default="Devin Booker",
        help="Player's full name (e.g., 'LeBron James', 'Stephen Curry')"
    )
    
    args = parser.parse_args()
    
    os.makedirs("output", exist_ok=True)
    
    season = get_current_season()
    player = args.player
    
    # Fetch data
    print(f"Fetching data for {player}...\n")
    player_logs = fetch_player_logs(player, season)
    team_stats = fetch_team_stats(season)
    
    print(f"\nTeam stats loaded: {len(team_stats)} teams")
    print(f"Player logs loaded: {len(player_logs)} games")
    
    # Process data
    print("\n" + "="*60)
    print("PROCESSING DATA")
    print("="*60)
    processed_df = process_player_data(player, player_logs, team_stats)
    
    # Save
    output_path = f"output/{player}_processed.csv"
    processed_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
    
    print(f"\nNext step: python train_model.py --player \"{player}\"")
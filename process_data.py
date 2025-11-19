# process_data.py
"""
Processes raw NBA data into ML-ready features.
Assumes data has already been fetched via get_stats.py
"""

import pandas as pd
import numpy as np
import util

# -----------------------------------------------------
# Feature engineering pipeline
# -----------------------------------------------------
def engineer_features(player_logs, team_stats):
    """
    Adds features to player game logs:
    - Rolling averages (PTS, REB, AST)
    - Opponent defensive rating and pace
    - Home/away indicator
    - Days rest and back-to-back indicator
    """
    df = player_logs.copy()
    
    # Convert and sort by date
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    
    # Extract opponent team abbreviation from MATCHUP
    # MATCHUP format: "PHX vs. LAL" or "PHX @ BOS"
    df["OPP_ABBREV"] = df["MATCHUP"].apply(lambda x: x.split(" ")[-1])
    
    # Map abbreviation to full team name
    df["OPP_TEAM_NAME"] = df["OPP_ABBREV"].map(util.NBA_TEAM_ABBREV_TO_NAME)
    
    # Debug: Check if mapping worked
    unmapped = df[df["OPP_TEAM_NAME"].isna()]["OPP_ABBREV"].unique()
    if len(unmapped) > 0:
        print(f"⚠️  Warning: Could not map these team abbreviations: {unmapped}")
    
    # Merge with team stats to get opponent metrics
    df = df.merge(
        team_stats,
        left_on="OPP_TEAM_NAME",
        right_on="TEAM_NAME",
        how="left"
    )
    
    # Debug: Check merge success
    missing_stats = df["DEF_RATING"].isna().sum()
    if missing_stats > 0:
        print(f"⚠️  Warning: {missing_stats} games missing opponent stats")
        print("Sample missing opponents:")
        print(df[df["DEF_RATING"].isna()][["GAME_DATE", "MATCHUP", "OPP_TEAM_NAME"]].head())
    
    # Rename to make it clear these are opponent stats
    df.rename(columns={
        "DEF_RATING": "OPP_DEF_RATING",
        "PACE": "OPP_PACE"
    }, inplace=True)
    
    # Drop unnecessary columns
    df.drop(columns=["TEAM_NAME", "OPP_ABBREV"], inplace=True, errors="ignore")
    
    # Rolling averages (last 3 games)
    df["PTS_last3"] = df["PTS"].rolling(3, min_periods=1).mean()
    df["REB_last3"] = df["REB"].rolling(3, min_periods=1).mean()
    df["AST_last3"] = df["AST"].rolling(3, min_periods=1).mean()
    df["FG_PCT_last3"] = df["FG_PCT"].rolling(3, min_periods=1).mean()
    df["MIN_last3"] = df["MIN"].rolling(3, min_periods=1).mean()
    
    # Home vs away
    df["Is_Home"] = df["MATCHUP"].str.contains("vs").astype(int)
    
    # Days rest and back-to-back
    df["Prev_Date"] = df["GAME_DATE"].shift(1)
    df["Days_Rest"] = (df["GAME_DATE"] - df["Prev_Date"]).dt.days
    df["Is_BackToBack"] = (df["Days_Rest"] == 1).astype(int)
    
    # Clean up temporary columns
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
    print(f"  - Valid Days_Rest: {df['Days_Rest'].notna().sum()}/{len(df)}")
    
    # Remove rows with NaN in critical features
    initial_rows = len(df)
    df = df.dropna(subset=["PTS_last3", "OPP_DEF_RATING", "Days_Rest"])
    
    print(f"\nOutput: {len(df)} games (removed {initial_rows - len(df)} incomplete rows)")
    
    return df


# -----------------------------------------------------
# Main - for testing
# -----------------------------------------------------
if __name__ == "__main__":
    import os
    from get_stats import fetch_player_logs, fetch_team_stats, get_current_season
    
    os.makedirs("output", exist_ok=True)
    
    season = get_current_season()
    player = "Devin Booker"
    
    # Fetch data
    print("Fetching data...\n")
    player_logs = fetch_player_logs(player, season)
    team_stats = fetch_team_stats(season)
    
    print(f"\nTeam stats loaded: {len(team_stats)} teams")
    print("Sample team names:", team_stats["TEAM_NAME"].head(5).tolist())
    
    print(f"\nPlayer logs loaded: {len(player_logs)} games")
    print("Sample matchups:", player_logs["MATCHUP"].head(5).tolist())
    
    # Process data
    print("\n" + "="*60)
    print("PROCESSING DATA")
    print("="*60)
    processed_df = process_player_data(player, player_logs, team_stats)
    
    # Save
    output_path = f"output/{player}_processed.csv"
    processed_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
    
    # Show sample
    print("\nSample features:")
    feature_cols = ["GAME_DATE", "MATCHUP", "PTS", "PTS_last3", "OPP_DEF_RATING", 
                    "OPP_PACE", "Is_Home", "Is_BackToBack"]
    print(processed_df[feature_cols].head(10).to_string())
    
    print("\nDone.")
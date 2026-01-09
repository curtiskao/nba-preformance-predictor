# process_data.py
"""
Processes raw NBA data into ML-ready features.
Includes improved features for better XGBoost performance.
Automatically fetches 3 seasons by default.
"""

import pandas as pd
import numpy as np
import argparse
import os
import util

from get_stats import fetch_player_logs_multi_season, fetch_team_stats, get_current_season


# -----------------------------------------------------
# Feature engineering with improved features
# -----------------------------------------------------
def engineer_features(player_logs, team_stats):
    """
    Comprehensive feature engineering for XGBoost.
    Includes multiple rolling windows, consistency metrics, efficiency features.
    """
    df = player_logs.copy()
    
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    
    print("  → Mapping opponent teams...")
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
    # MULTIPLE ROLLING WINDOWS (L3, L5, L10)
    # ========================================
    print("  → Adding rolling averages (3, 5, 10 game windows)...")
    
    # Short-term (last 3 games) - recent form
    df["PTS_L3"] = df["PTS"].shift(1).rolling(3, min_periods=1).mean()
    df["MIN_L3"] = df["MIN"].shift(1).rolling(3, min_periods=1).mean()
    df["FGA_L3"] = df["FGA"].shift(1).rolling(3, min_periods=1).mean()
    df["FG_PCT_L3"] = df["FG_PCT"].shift(1).rolling(3, min_periods=1).mean()
    
    # Medium-term (last 5 games) - stable form
    df["PTS_L5"] = df["PTS"].shift(1).rolling(5, min_periods=1).mean()
    df["MIN_L5"] = df["MIN"].shift(1).rolling(5, min_periods=1).mean()
    df["FGA_L5"] = df["FGA"].shift(1).rolling(5, min_periods=1).mean()
    df["FG_PCT_L5"] = df["FG_PCT"].shift(1).rolling(5, min_periods=1).mean()
    
    # Long-term (last 10 games) - season form
    df["PTS_L10"] = df["PTS"].shift(1).rolling(10, min_periods=1).mean()
    df["MIN_L10"] = df["MIN"].shift(1).rolling(10, min_periods=1).mean()
    
    # ========================================
    # CONSISTENCY METRICS
    # ========================================
    print("  → Adding consistency metrics...")
    
    df["PTS_STD_L10"] = df["PTS"].shift(1).rolling(10, min_periods=2).std()
    df["PTS_STD_L10"] = df["PTS_STD_L10"].fillna(0)
    
    # ========================================
    # EFFICIENCY METRICS
    # ========================================
    print("  → Adding efficiency metrics...")
    
    # Points per minute
    df["PTS_PER_MIN_L5"] = df["PTS_L5"] / df["MIN_L5"].replace(0, 1)
    
    # True shooting percentage (accounts for 3s and FTs)
    if "FTA" in df.columns:
        df["TS_PCT_L5"] = df["PTS"].shift(1).rolling(5, min_periods=1).sum() / (
            2 * (df["FGA"].shift(1).rolling(5, min_periods=1).sum() + 
                 0.44 * df["FTA"].shift(1).rolling(5, min_periods=1).sum())
        )
        df["TS_PCT_L5"] = df["TS_PCT_L5"].fillna(0.5).replace([np.inf, -np.inf], 0.5)
    else:
        df["TS_PCT_L5"] = df["FG_PCT_L5"]
    
    # ========================================
    # 3-POINT FEATURES
    # ========================================
    print("  → Adding 3-point features...")
    
    # 3-pointers MADE (not just attempts)
    if "FG3M" in df.columns:
        df["FG3M_L5"] = df["FG3M"].shift(1).rolling(5, min_periods=1).mean()
    else:
        df["FG3M_L5"] = 0
    
    df["FG3A_L5"] = df["FG3A"].shift(1).rolling(5, min_periods=1).mean()
    df["FG3_PCT_L5"] = df["FG3_PCT"].shift(1).rolling(5, min_periods=1).mean()
    df["FG3_PCT_L5"] = df["FG3_PCT_L5"].fillna(0).replace([np.inf, -np.inf], 0)
    
    # ========================================
    # USAGE PROXY
    # ========================================
    print("  → Adding usage proxy...")
    
    if "TOV" in df.columns and "FTA" in df.columns:
        df["USAGE_L5"] = (
            df["FGA"].shift(1).rolling(5, min_periods=1).mean() +
            0.44 * df["FTA"].shift(1).rolling(5, min_periods=1).mean() +
            df["TOV"].shift(1).rolling(5, min_periods=1).mean()
        )
    else:
        df["USAGE_L5"] = df["FGA_L5"]
    
    # ========================================
    # GAME CONTEXT
    # ========================================
    print("  → Adding game context features...")
    
    df["Is_Home"] = df["MATCHUP"].str.contains("vs").astype(int)
    
    # Rest days
    df["Prev_Date"] = df["GAME_DATE"].shift(1)
    df["Days_Rest"] = (df["GAME_DATE"] - df["Prev_Date"]).dt.days
    df["Days_Rest"] = df["Days_Rest"].fillna(3)
    df["Is_BackToBack"] = (df["Days_Rest"] == 1).astype(int)
    
    # ========================================
    # TREND FEATURES
    # ========================================
    print("  → Adding trend features...")
    
    df["PTS_TREND"] = df["PTS_L3"] - df["PTS_L10"]
    df["MIN_TREND"] = df["MIN_L3"] - df["MIN_L10"]
    
    # Cleanup
    df.drop(columns=["Prev_Date"], inplace=True, errors="ignore")
    
    # Fill any remaining NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
        # Replace inf values
        df[col] = df[col].replace([np.inf, -np.inf], df[col].median())
    
    print("  ✓ Feature engineering complete")
    
    return df


# -----------------------------------------------------
# Complete processing pipeline
# -----------------------------------------------------
def process_player_data(player_name, player_logs, team_stats):
    """Complete processing pipeline."""
    print(f"\nProcessing data for {player_name}...")
    print(f"Input: {len(player_logs)} games")
    
    df = engineer_features(player_logs, team_stats)
    
    # Only drop rows missing opponent stats
    initial_rows = len(df)
    df = df.dropna(subset=["OPP_DEF_RATING"])
    
    print(f"\nOutput: {len(df)} games (removed {initial_rows - len(df)} incomplete rows)")
    
    return df


# -----------------------------------------------------
# Main
# -----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process NBA player data with improved features for XGBoost"
    )
    parser.add_argument(
        "--player",
        type=str,
        default="Devin Booker",
        help="Player's full name"
    )
    parser.add_argument(
        "--seasons",
        type=int,
        default=3,
        help="Number of seasons to fetch (default: 3)"
    )
    
    args = parser.parse_args()
    
    os.makedirs("output", exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"NBA DATA PROCESSING - {args.player}")
    print(f"{'='*70}\n")
    
    # Fetch data (automatically multi-season)
    player_logs = fetch_player_logs_multi_season(args.player, num_seasons=args.seasons)
    team_stats = fetch_team_stats(get_current_season())
    
    # Process
    processed_df = process_player_data(args.player, player_logs, team_stats)
    
    # Save
    output_path = f"output/{args.player}_processed.csv"
    processed_df.to_csv(output_path, index=False)
    
    # Show summary
    feature_cols = [col for col in processed_df.columns if 
                   any(x in col for x in ['_L3', '_L5', '_L10', 'OPP_', 'Is_', 'TREND', 'STD', 'PER_MIN', 'USAGE', 'TS_'])]
    
    print(f"\n✓ Saved to {output_path}")
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"  - Rolling averages: {len([f for f in feature_cols if '_L' in f])}")
    print(f"  - Opponent features: {len([f for f in feature_cols if 'OPP_' in f])}")
    print(f"  - Context features: {len([f for f in feature_cols if 'Is_' in f or 'Days_Rest' in f])}")
    print(f"  - Efficiency/Trend: {len([f for f in feature_cols if any(x in f for x in ['TREND', 'STD', 'PER_MIN', 'TS_'])])}")
    
    print(f"\nNext step: python train_model.py --player \"{args.player}\"")
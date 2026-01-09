# simple_predict.py
"""
Enhanced simple prediction with outlier filtering and better adjustments.
Often more accurate than ML for single-player predictions.
"""

import pandas as pd
import numpy as np
import argparse


def simple_prediction(player_name, opponent_name, is_home, days_rest):
    """
    Simple but effective prediction using recent averages.
    Filters outliers for more stable predictions.
    """
    
    # Load data
    df = pd.read_csv(f"output/{player_name}_processed.csv")
    df = df.sort_values("GAME_DATE")
    
    team_stats = pd.read_csv("output/team_stats.csv")
    
    print(f"\n{'='*60}")
    print(f"SIMPLE PREDICTION FOR: {player_name}")
    print(f"{'='*60}\n")
    
    # ========================================
    # BASE PREDICTION WITH OUTLIER FILTERING
    # ========================================
    
    # Get last 10 games for outlier detection
    last_10 = df.tail(10)["PTS"]
    
    # Calculate bounds (remove extreme outliers)
    q1 = last_10.quantile(0.25)
    q3 = last_10.quantile(0.75)
    iqr = q3 - q1
    lower_bound = max(10, q1 - 1.5 * iqr)  # At least 10 points
    upper_bound = min(50, q3 + 1.5 * iqr)  # At most 50 points
    
    # Filter last 5 games (remove outliers)
    last_5 = df.tail(5)["PTS"]
    filtered_5 = last_5[(last_5 >= lower_bound) & (last_5 <= upper_bound)]
    
    if len(filtered_5) >= 3:
        recent_pts = filtered_5.mean()
        outliers_removed = 5 - len(filtered_5)
        print(f"Base: Last 5 games average")
        print(f"  Raw average: {last_5.mean():.1f} pts")
        print(f"  Filtered average: {recent_pts:.1f} pts")
        if outliers_removed > 0:
            print(f"  (Filtered out {outliers_removed} outlier(s))")
    else:
        # Not enough valid games after filtering
        recent_pts = last_5.mean()
        print(f"Base: Last 5 games average")
        print(f"  Average: {recent_pts:.1f} pts")
        print(f"  (No outlier filtering applied)")
    
    # Also calculate season average for reference
    season_avg = df["PTS"].mean()
    print(f"  Season average: {season_avg:.1f} pts\n")
    
    # ========================================
    # OPPONENT DEFENSE ADJUSTMENT
    # ========================================
    
    opp_stats = team_stats[team_stats["TEAM_NAME"] == opponent_name]
    if not opp_stats.empty:
        opp_def = opp_stats.iloc[0]["DEF_RATING"]
        league_avg_def = 110.0
        
        # Worse defense = more points (scale: 0.15 pts per rating point)
        def_adjustment = (league_avg_def - opp_def) * 0.15
        
        print(f"Opponent: {opponent_name}")
        print(f"  Defensive Rating: {opp_def:.1f}")
        print(f"  League Average: {league_avg_def:.1f}")
        
        if opp_def < 108:
            print(f"  → Elite defense")
        elif opp_def < 110:
            print(f"  → Good defense")
        elif opp_def < 112:
            print(f"  → Average defense")
        elif opp_def < 114:
            print(f"  → Below average defense")
        else:
            print(f"  → Poor defense")
        
        print(f"  Adjustment: {def_adjustment:+.1f} pts\n")
    else:
        def_adjustment = 0
        print(f"Opponent: {opponent_name}")
        print(f"  ⚠️  Team not found in stats")
        print(f"  Adjustment: 0.0 pts\n")
    
    # ========================================
    # HOME COURT ADVANTAGE
    # ========================================
    
    # Check actual home/away split for this player
    if 'Is_Home' in df.columns:
        home_games = df[df['Is_Home'] == 1]
        away_games = df[df['Is_Home'] == 0]
        
        if len(home_games) > 10 and len(away_games) > 10:
            actual_home_adv = home_games['PTS'].mean() - away_games['PTS'].mean()
            # Use player's actual home advantage (capped at 1-4 points)
            home_bonus = min(4.0, max(1.0, actual_home_adv)) if is_home else 0
            
            print(f"Location: {'Home' if is_home else 'Away'}")
            print(f"  Player's home avg: {home_games['PTS'].mean():.1f} pts")
            print(f"  Player's away avg: {away_games['PTS'].mean():.1f} pts")
            print(f"  Actual home advantage: {actual_home_adv:+.1f} pts")
            print(f"  Adjustment: {home_bonus:+.1f} pts\n")
        else:
            # Default home advantage if not enough data
            home_bonus = 2.5 if is_home else 0
            print(f"Location: {'Home' if is_home else 'Away'}")
            print(f"  Using default home advantage")
            print(f"  Adjustment: {home_bonus:+.1f} pts\n")
    else:
        home_bonus = 2.5 if is_home else 0
        print(f"Location: {'Home' if is_home else 'Away'}")
        print(f"  Adjustment: {home_bonus:+.1f} pts\n")
    
    # ========================================
    # REST DAYS ADJUSTMENT
    # ========================================
    
    # Check actual rest days effect for this player
    if 'Days_Rest' in df.columns:
        b2b_games = df[df['Days_Rest'] == 1]
        rested_games = df[df['Days_Rest'] >= 2]
        
        if len(b2b_games) > 5 and len(rested_games) > 10:
            actual_b2b_penalty = rested_games['PTS'].mean() - b2b_games['PTS'].mean()
            # Use player's actual back-to-back penalty (capped at 2-5 points)
            fatigue_penalty = -min(5.0, max(2.0, actual_b2b_penalty)) if days_rest == 1 else 0
            
            print(f"Rest: {days_rest} day(s)")
            if days_rest == 1:
                print(f"  Back-to-back game")
                print(f"  Player's B2B avg: {b2b_games['PTS'].mean():.1f} pts")
                print(f"  Player's rested avg: {rested_games['PTS'].mean():.1f} pts")
                print(f"  Actual B2B penalty: {actual_b2b_penalty:.1f} pts")
            print(f"  Adjustment: {fatigue_penalty:+.1f} pts\n")
        else:
            # Default fatigue penalty
            fatigue_penalty = -3.0 if days_rest == 1 else 0
            print(f"Rest: {days_rest} day(s)")
            if days_rest == 1:
                print(f"  Back-to-back game (using default penalty)")
            print(f"  Adjustment: {fatigue_penalty:+.1f} pts\n")
    else:
        fatigue_penalty = -3.0 if days_rest == 1 else 0
        print(f"Rest: {days_rest} day(s)")
        print(f"  Adjustment: {fatigue_penalty:+.1f} pts\n")
    
    # ========================================
    # FINAL PREDICTION
    # ========================================
    
    prediction = recent_pts + def_adjustment + home_bonus + fatigue_penalty
    
    print(f"{'='*60}")
    print(f"CALCULATION:")
    print(f"{'='*60}\n")
    print(f"  {recent_pts:.1f}  (recent average)")
    print(f"  {def_adjustment:+.1f}  (opponent defense)")
    print(f"  {home_bonus:+.1f}  (home/away)")
    print(f"  {fatigue_penalty:+.1f}  (rest/fatigue)")
    print(f"  ────────")
    print(f"  {prediction:.1f}  points\n")
    
    print(f"{'='*60}")
    print(f"🎯 PREDICTED POINTS: {prediction:.1f}")
    print(f"{'='*60}\n")
    
    # Confidence assessment
    pts_std = df.tail(10)["PTS"].std()
    print(f"Confidence Assessment:")
    print(f"  Recent volatility (std): {pts_std:.1f} pts")
    
    if pts_std < 5:
        print(f"  ✓ High confidence (consistent scorer)")
    elif pts_std < 7:
        print(f"  ✓ Moderate confidence")
    else:
        print(f"  ⚠️  Lower confidence (volatile scorer)")
    
    print(f"  Typical range: {prediction - pts_std:.1f} to {prediction + pts_std:.1f} pts\n")
    
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple but effective NBA points prediction"
    )
    parser.add_argument(
        "--player",
        required=True,
        help="Player's full name"
    )
    parser.add_argument(
        "--opponent",
        required=True,
        help="Opponent team name"
    )
    parser.add_argument(
        "--home",
        action="store_true",
        help="Is this a home game?"
    )
    parser.add_argument(
        "--rest",
        type=int,
        default=2,
        help="Days of rest since last game (default: 2)"
    )
    
    args = parser.parse_args()
    
    simple_prediction(args.player, args.opponent, args.home, args.rest)
from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import os

def fetch_player_stats(year: int):
    season = f"{year}-{str(year + 1)[-2:]}"
    
    basic_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star='Regular Season',
        measure_type_detailed_defense='Base'
    ).get_data_frames()[0]
    basic_stats["Year"] = year

    advanced_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star='Regular Season',
        measure_type_detailed_defense='Advanced'
    ).get_data_frames()[0]
    advanced_stats["Year"] = year

    return basic_stats, advanced_stats

years = list(range(2021, 2025))

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data_files")

os.makedirs(data_dir, exist_ok=True)

for year in years:
    basic_df, advanced_df = fetch_player_stats(year)
    basic_df.to_csv(os.path.join(data_dir, f"{year}_nba.csv"), index=False)
    advanced_df.to_csv(os.path.join(data_dir, f"{year}_advanced.csv"), index=False)

print("✅ Player stats downloaded and saved to:", data_dir)


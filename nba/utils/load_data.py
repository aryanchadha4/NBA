from pathlib import Path
import pandas as pd
import glob

DATA_DIR = Path(__file__).resolve().parent.parent / "data_files"

def load_unique_player_names():
    """Scans all *_nba.csv and *_advanced.csv files to get unique player names"""
    pattern = str(DATA_DIR / "*_nba.csv")
    pattern_adv = str(DATA_DIR / "*_advanced.csv")
    files = glob.glob(pattern) + glob.glob(pattern_adv)

    player_names = set()
    for file in files:
        df = pd.read_csv(file)
        if "Player" in df.columns:
            names = df["Player"].dropna().unique()
            player_names.update(names)

    return list(player_names)

def load_unique_team_names():
    """Loads team names from NBA_Teams_Stats.csv"""
    file = DATA_DIR / "NBA_Teams_Stats.csv"
    df = pd.read_csv(file)
    return df["Team"].dropna().unique().tolist()

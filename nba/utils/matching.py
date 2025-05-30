import json
from pathlib import Path
from fuzzywuzzy import process

DATA_DIR = Path(__file__).resolve().parent.parent / "data_files"

with open(DATA_DIR / "cached_players.json") as f:
    PLAYER_NAMES = json.load(f)

with open(DATA_DIR / "cached_teams.json") as f:
    TEAM_NAMES = json.load(f)

def fuzzy_match_player(name):
    match, score = process.extractOne(name, PLAYER_NAMES)
    return match if score > 80 else None

def fuzzy_match_team(name):
    match, score = process.extractOne(name, TEAM_NAMES)
    return match if score > 80 else None


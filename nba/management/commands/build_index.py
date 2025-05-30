from django.core.management.base import BaseCommand
from nba.utils.load_data import load_unique_player_names, load_unique_team_names
import json
from pathlib import Path

class Command(BaseCommand):
    help = "Builds and saves cached player and team name indexes for fuzzy matching"

    def handle(self, *args, **kwargs):
        self.stdout.write("Building player and team name indexes...")

        players = load_unique_player_names()
        teams = load_unique_team_names()

        data_dir = Path(__file__).resolve().parent.parent.parent / "data_files"

        with open(data_dir / "cached_players.json", "w") as f:
            json.dump(players, f)

        with open(data_dir / "cached_teams.json", "w") as f:
            json.dump(teams, f)

        self.stdout.write(self.style.SUCCESS("✔ Indexes built and cached in data_files/."))

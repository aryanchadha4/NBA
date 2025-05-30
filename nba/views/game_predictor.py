from django.http import JsonResponse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from thefuzz import process
import os

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data_files")
DATA_FOLDER = os.path.abspath(DATA_FOLDER)

file_paths = [
    os.path.join(DATA_FOLDER, "NBA_Game_Outcomes_October.csv"),
    os.path.join(DATA_FOLDER, "NBA_Game_Outcomes_November.csv"),
    os.path.join(DATA_FOLDER, "NBA_Game_Outcomes_December.csv"),
    os.path.join(DATA_FOLDER, "NBA_Game_Outcomes_January.csv"),
]

game_logs = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)

game_logs = game_logs.rename(columns={
    "Visitor/Neutral": "visitor_team",
    "PTS": "visitor_pts",
    "Home/Neutral": "home_team",
    "PTS.1": "home_pts"
})

game_logs["winner"] = (game_logs["home_pts"] > game_logs["visitor_pts"]).astype(int)
game_logs["Home_Indicator"] = 1

team_stats = pd.read_csv(os.path.join(DATA_FOLDER, "NBA_Teams_Stats.csv"))
team_stats = team_stats.rename(columns={
    "eFG%.1": "Def_eFG%",
    "TOV%.1": "Def_TOV%",
    "FT/FGA.1": "Def_FT/FGA",
    "DRB%": "Def_DRB%"
})

team_stats["Team"] = team_stats["Team"].str.strip()
game_logs["home_team"] = game_logs["home_team"].str.strip()
game_logs["visitor_team"] = game_logs["visitor_team"].str.strip()

FEATURES = ["ORtg", "DRtg", "NRtg", "Pace", "FTr", "3PAr", "TS%", "eFG%", "TOV%", "ORB%", 
            "FT/FGA", "Def_eFG%", "Def_TOV%", "Def_DRB%", "Def_FT/FGA"]

team_stats = team_stats[["Team"] + FEATURES]

game_logs = game_logs.merge(team_stats, left_on="home_team", right_on="Team", how="left")
game_logs = game_logs.rename(columns={col: col + "_home" for col in FEATURES})

game_logs = game_logs.merge(team_stats, left_on="visitor_team", right_on="Team", how="left")
game_logs = game_logs.rename(columns={col: col + "_away" for col in FEATURES})

game_logs = game_logs.drop(columns=["Team_home", "Team_away"], errors="ignore")

for stat in FEATURES:
    game_logs[f"{stat}_diff"] = game_logs[f"{stat}_home"] - game_logs[f"{stat}_away"]

FEATURE_COLS = [col for col in game_logs.columns if "_diff" in col] + ["Home_Indicator"]
X_train, X_test, y_train, y_test = train_test_split(game_logs[FEATURE_COLS], game_logs["winner"], test_size=0.2, random_state=42)

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

neural_network = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=35, random_state=42)
neural_network.fit(X_train, y_train)

print("Logistic Regression Accuracy:", accuracy_score(y_test, logistic_model.predict(X_test)))
print("Random Forest Accuracy:", accuracy_score(y_test, random_forest_model.predict(X_test)))
print("Neural Network Accuracy:", accuracy_score(y_test, neural_network.predict(X_test)))

def fuzzy_match_team(team_name):
    """ Find closest matching team name """
    match, score = process.extractOne(team_name, team_stats["Team"].tolist())
    return match if score > 80 else None


def predict_game(request):
    """ API: Predict game winner based on team inputs """
    home_team = request.GET.get("home_team", "").strip()
    away_team = request.GET.get("away_team", "").strip()

    if home_team.lower() == away_team.lower():
        return JsonResponse({"error": f"Teams cannot be the same ({home_team})."}, status=400)

    home_team = fuzzy_match_team(home_team)
    away_team = fuzzy_match_team(away_team)

    if not home_team or not away_team:
        return JsonResponse({"error": "Invalid team names."}, status=400)

    home_stats = team_stats[team_stats["Team"] == home_team]
    away_stats = team_stats[team_stats["Team"] == away_team]

    match_features = {f"{stat}_diff": home_stats[stat].values[0] - away_stats[stat].values[0] for stat in FEATURES}
    match_features["Home_Indicator"] = 1

    match_df = pd.DataFrame([match_features])

    missing_cols = set(FEATURE_COLS) - set(match_df.columns)
    for col in missing_cols:
        match_df[col] = 0

    pred_logistic = logistic_model.predict(match_df)[0]
    pred_rf = random_forest_model.predict(match_df)[0]
    pred_nn = neural_network.predict(match_df)[0]


    return JsonResponse({
        "home_team": home_team,
        "away_team": away_team,
        "logistic_winner": home_team if pred_logistic == 1 else away_team,
        "random_forest_winner": home_team if pred_rf == 1 else away_team,
        "neural_network_winner": home_team if pred_nn == 1 else away_team,
    })

# API view
def api_game_prediction(request):
    home = request.GET.get("home_team", "").strip()
    away = request.GET.get("away_team", "").strip()

    if not home or not away:
        return JsonResponse({"error": "Missing team names"}, status=400)
    if home.lower() == away.lower():
        return JsonResponse({"error": "Teams must be different"}, status=400)

    home_team = fuzzy_match_team(home)
    away_team = fuzzy_match_team(away)

    if not home_team or not away_team:
        return JsonResponse({"error": "Invalid team names"}, status=400)

    # Pull stats for each team
    home_stats = team_stats[team_stats["Team"] == home_team]
    away_stats = team_stats[team_stats["Team"] == away_team]

    if home_stats.empty or away_stats.empty:
        return JsonResponse({"error": "Stats not found for teams"}, status=400)

    # Build feature vector inline
    feature_vector = {}
    for stat in FEATURES:
        feature_vector[f"{stat}_diff"] = float(home_stats[stat].values[0]) - float(away_stats[stat].values[0])
    feature_vector["Home_Indicator"] = 1

    # Convert to dataframe
    import pandas as pd
    match_df = pd.DataFrame([feature_vector])

    # Align column order with model expectations
    FEATURE_COLS = list(feature_vector.keys())  # or hardcode if needed

    # Run predictions
    pred_log = logistic_model.predict(match_df)[0]
    pred_rf = random_forest_model.predict(match_df)[0]
    pred_nn = neural_network.predict(match_df)[0]

    return JsonResponse({
        "home_team": home_team,
        "away_team": away_team,
        "logistic_winner": home_team if pred_log == 1 else away_team,
        "random_forest_winner": home_team if pred_rf == 1 else away_team,
        "neural_network_winner": home_team if pred_nn == 1 else away_team,
    })


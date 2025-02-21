from django.http import JsonResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from thefuzz import process
import os

# Load Data
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data_files")

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
game_logs["Home_Indicator"] = 1  # Home team is always 1

team_stats = pd.read_csv(os.path.join(DATA_FOLDER, "NBA_Teams_Stats.csv"))
team_stats = team_stats.rename(columns={
    "eFG%.1": "Def_eFG%",
    "TOV%.1": "Def_TOV%",
    "FT/FGA.1": "Def_FT/FGA",
    "DRB%": "Def_DRB%"
})

# Ensure no extra spaces in team names
team_stats["Team"] = team_stats["Team"].str.strip()
game_logs["home_team"] = game_logs["home_team"].str.strip()
game_logs["visitor_team"] = game_logs["visitor_team"].str.strip()

# Selected Features for ML
FEATURES = ["ORtg", "DRtg", "NRtg", "Pace", "FTr", "3PAr", "TS%", "eFG%", "TOV%", "ORB%", 
            "FT/FGA", "Def_eFG%", "Def_TOV%", "Def_DRB%", "Def_FT/FGA"]

team_stats = team_stats[["Team"] + FEATURES]

# Merge team stats with game logs
game_logs = game_logs.merge(team_stats, left_on="home_team", right_on="Team", how="left")
game_logs = game_logs.rename(columns={col: col + "_home" for col in FEATURES})

game_logs = game_logs.merge(team_stats, left_on="visitor_team", right_on="Team", how="left")
game_logs = game_logs.rename(columns={col: col + "_away" for col in FEATURES})

game_logs = game_logs.drop(columns=["Team_home", "Team_away"], errors="ignore")

# Compute stat differentials (home - away)
for stat in FEATURES:
    game_logs[f"{stat}_diff"] = game_logs[f"{stat}_home"] - game_logs[f"{stat}_away"]

# Define training data
FEATURE_COLS = [col for col in game_logs.columns if "_diff" in col] + ["Home_Indicator"]
X_train, X_test, y_train, y_test = train_test_split(game_logs[FEATURE_COLS], game_logs["winner"], test_size=0.2, random_state=42)

# Train Models
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

neural_network = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=35, random_state=42)
neural_network.fit(X_train, y_train)

# Model Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, logistic_model.predict(X_test)))
print("Random Forest Accuracy:", accuracy_score(y_test, random_forest_model.predict(X_test)))
print("Neural Network Accuracy:", accuracy_score(y_test, neural_network.predict(X_test)))

# ---------------- API Functions ---------------- #

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

    # Ensure all required columns exist
    missing_cols = set(FEATURE_COLS) - set(match_df.columns)
    for col in missing_cols:
        match_df[col] = 0

    # Predictions from each model
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


def home(request):
    """ API: Welcome Message """
    return JsonResponse({"message": "Welcome to NBA Predictions API!"})


from django.http import JsonResponse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from thefuzz import process

# Load Data
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data_files")
YEARS = list(range(2021, 2025))

def load_data():
    """Load and merge player stats from multiple years."""
    df_list = []
    for year in YEARS:
        df_basic_path = os.path.join(DATA_FOLDER, f"{year}_nba.csv")
        df_adv_path = os.path.join(DATA_FOLDER, f"{year}_advanced.csv")

        df_basic = pd.read_csv(df_basic_path)
        df_basic["Year"] = year
        df_adv = pd.read_csv(df_adv_path)
        df_adv["Year"] = year

        df_merged = df_basic.merge(df_adv, on=["Player", "Year"], how="left")
        df_list.append(df_merged)

    return pd.concat(df_list, ignore_index=True).sort_values(by=["Player", "Year"])

df = load_data()

# Define features and targets
LIMITED_FEATURES = ["PTS", "AST", "TRB", "FG%", "3P%", "TS%", "BPM", "Age_x", "FGA", "3PA", "USG%"]
FULL_FEATURES = [
    "PTS", "AST", "TRB", "FG%", "3P%", "TS%", "BPM", "Age_x", "FGA", "3PA", "USG%", "STL", "BLK", 
    "TOV", "PF", "PER", "FTr", "ORB%", "DRB%", "TRB%", "AST%", "STL%", "BLK%", "TOV%", "OWS", 
    "DWS", "WS", "WS/48", "VORP"
]

TARGETS = ["PTS", "AST", "TRB", "FG%", "3P%"]

# Create previous season features
for feature in FULL_FEATURES:
    df[f'Prev_{feature}'] = df.groupby('Player')[feature].shift(1)

df.dropna(subset=[f'Prev_{feat}' for feat in FULL_FEATURES] + TARGETS, inplace=True)

# Train models
models = {}
scalers = {}

X_limited = df[[f'Prev_{feat}' for feat in LIMITED_FEATURES]]
X_full = df[[f'Prev_{feat}' for feat in FULL_FEATURES]]

for target in TARGETS:
    y = df[target]

    # Train-test split for each feature set
    X_train_limited, X_test_limited, y_train, y_test = train_test_split(X_limited, y, test_size=0.2, random_state=42)
    X_train_full, X_test_full, _, _ = train_test_split(X_full, y, test_size=0.2, random_state=42)

    # Scale features for linear and deep learning models
    scaler_limited = StandardScaler()
    X_train_limited_scaled = scaler_limited.fit_transform(X_train_limited)
    X_test_limited_scaled = scaler_limited.transform(X_test_limited)

    scaler_full = StandardScaler()
    X_train_full_scaled = scaler_full.fit_transform(X_train_full)
    X_test_full_scaled = scaler_full.transform(X_test_full)

    # Train models
    models[target] = {
        "linear": LinearRegression().fit(X_train_limited_scaled, y_train),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_full, y_train),
        "neural_network": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=50, random_state=42).fit(X_train_full_scaled, y_train)
    }

    # Store scalers
    scalers[target] = {
        "limited": scaler_limited,
        "full": scaler_full
    }

# Prediction function
def predict_player_stats(request):
    """API endpoint to predict a player's next season stats."""
    player_name = request.GET.get("player", "").strip().lower()
    df_temp = df.copy()
    df_temp['Player'] = df_temp['Player'].str.strip().str.lower()

    best_match, score = process.extractOne(player_name, df_temp['Player'].unique())
    if score < 80:
        return JsonResponse({"error": f"Player '{player_name.title()}' not found."}, status=400)

    player_data = df_temp[df_temp['Player'] == best_match].sort_values(by='Year', ascending=False).iloc[0]

    input_limited = np.array([[player_data[f'Prev_{feat}'] for feat in LIMITED_FEATURES]])
    input_full = np.array([[player_data[f'Prev_{feat}'] for feat in FULL_FEATURES]])

    predictions = {}
    for target in TARGETS:
        scalers_dict = scalers[target]
        models_dict = models[target]

        # Scale inputs
        input_limited_scaled = scalers_dict["limited"].transform(input_limited)
        input_full_scaled = scalers_dict["full"].transform(input_full)

        predictions[target] = {
            "linear": models_dict["linear"].predict(input_limited_scaled)[0],
            "random_forest": models_dict["random_forest"].predict(input_full)[0],
            "neural_network": models_dict["neural_network"].predict(input_full_scaled)[0],
        }

    return JsonResponse({"player": best_match.title(), "predictions": predictions})


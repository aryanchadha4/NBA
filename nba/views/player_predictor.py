import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from thefuzz import process
from django.http import JsonResponse

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data_files")
DATA_FOLDER = os.path.abspath(DATA_FOLDER)
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

LIMITED_FEATURES = ["PTS", "AST", "TRB", "FG%", "3P%", "TS%", "BPM", "Age_x", "FGA", "3PA", "USG%"]
FULL_FEATURES = [
    "PTS", "AST", "TRB", "FG%", "3P%", "TS%", "BPM", "Age_x", "FGA", "3PA", "USG%", "STL", "BLK", 
    "TOV", "PF", "PER", "FTr", "ORB%", "DRB%", "TRB%", "AST%", "STL%", "BLK%", "TOV%", "OWS", 
    "DWS", "WS", "WS/48", "VORP"
]

TARGETS = ["PTS", "AST", "TRB", "FG%", "3P%"]

for feature in FULL_FEATURES:
    df[f'Prev_{feature}'] = df.groupby('Player')[feature].shift(1)

df.dropna(subset=[f'Prev_{feat}' for feat in FULL_FEATURES] + TARGETS, inplace=True)

models = {}
scalers = {}

X_limited = df[[f'Prev_{feat}' for feat in LIMITED_FEATURES]]
X_full = df[[f'Prev_{feat}' for feat in FULL_FEATURES]]

for target in TARGETS:
    y = df[target]

    X_train_limited, X_test_limited, y_train, y_test = train_test_split(X_limited, y, test_size=0.2, random_state=42)
    X_train_full, X_test_full, _, _ = train_test_split(X_full, y, test_size=0.2, random_state=42)

    scaler_limited = StandardScaler()
    X_train_limited_scaled = scaler_limited.fit_transform(X_train_limited)
    X_test_limited_scaled = scaler_limited.transform(X_test_limited)

    scaler_full = StandardScaler()
    X_train_full_scaled = scaler_full.fit_transform(X_train_full)
    X_test_full_scaled = scaler_full.transform(X_test_full)

    r2_scores = {}

    models[target] = {
        "linear": LinearRegression().fit(X_train_limited_scaled, y_train),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_full, y_train),
        "neural_network": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42).fit(X_train_full_scaled, y_train)
    }

    r2_scores["linear"] = models[target]["linear"].score(X_test_limited_scaled, y_test)
    r2_scores["random_forest"] = models[target]["random_forest"].score(X_test_full, y_test)
    r2_scores["neural_network"] = models[target]["neural_network"].score(X_test_full_scaled, y_test)

    scalers[target] = {
        "limited": scaler_limited,
        "full": scaler_full
    }

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

        input_limited_scaled = scalers_dict["limited"].transform(input_limited)
        input_full_scaled = scalers_dict["full"].transform(input_full)

        predictions[target] = {
            "linear": models_dict["linear"].predict(input_limited_scaled)[0],
            "random_forest": models_dict["random_forest"].predict(input_full)[0],
            "neural_network": models_dict["neural_network"].predict(input_full_scaled)[0],
        }

    return JsonResponse({"player": best_match.title(), "predictions": predictions})
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
from scipy.special import expit, logit

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

        df_merged = df_basic.merge(df_adv, on=["PLAYER_NAME", "Year"], how="left")
        df_list.append(df_merged)

    return pd.concat(df_list, ignore_index=True).sort_values(by=["PLAYER_NAME", "Year"])

df = load_data()


LIMITED_FEATURES = ["PTS", "AST", "REB", "FG_PCT_x", "FG3_PCT", "TS_PCT", "NET_RATING", "AGE_x", "FGA_x", "FG3A", "USG_PCT"]
FULL_FEATURES = [
    "PTS", "AST", "REB", "FG_PCT_x", "FG3_PCT", "TS_PCT", "NET_RATING", "AGE_x", "FGA_x", "FG3A", "USG_PCT", "STL", "BLK", 
    "TOV", "PF", "NET_RATING", "OREB_PCT", "DREB_PCT", "REB_PCT", "AST_PCT", "TM_TOV_PCT", "OFF_RATING", 
    "DEF_RATING"
]

TARGETS = ["PTS", "AST", "REB", "FG_PCT_x", "FG3_PCT"]

for feature in FULL_FEATURES:
    df[f'Prev_{feature}'] = df.groupby('PLAYER_NAME')[feature].shift(1)

df.dropna(subset=[f'Prev_{feat}' for feat in FULL_FEATURES] + TARGETS, inplace=True)


X_limited = df[[f'Prev_{feat}' for feat in LIMITED_FEATURES]]
X_full = df[[f'Prev_{feat}' for feat in FULL_FEATURES]]

model_scores = {} 

models = {}
scalers = {}
model_scores = {}

X_limited = df[[f'Prev_{feat}' for feat in LIMITED_FEATURES]]
X_full = df[[f'Prev_{feat}' for feat in FULL_FEATURES]]

EPSILON = 1e-5

# Train models once per target
for target in TARGETS:
    y_raw = df[target]

    if target in ["FG_PCT_x", "FG3_PCT"]:
        y = logit(np.clip(y_raw, EPSILON, 1 - EPSILON))
    else:
        y = y_raw

    # Train/test split
    X_train_limited, X_test_limited, y_train, y_test = train_test_split(X_limited, y, test_size=0.2, random_state=42)
    X_train_full, X_test_full, _, _ = train_test_split(X_full, y, test_size=0.2, random_state=42)

    # Scaling
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
        "neural_network": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42).fit(X_train_full_scaled, y_train)
    }

    # Save accuracy
    model_scores[target] = {
        "linear": models[target]["linear"].score(X_test_limited_scaled, y_test),
        "random_forest": models[target]["random_forest"].score(X_test_full, y_test),
        "neural_network": models[target]["neural_network"].score(X_test_full_scaled, y_test),
    }

    scalers[target] = {
        "limited": scaler_limited,
        "full": scaler_full
    }

    print(f"\nAccuracy scores for {target}:")
    for name, score in model_scores[target].items():
        print(f"  {name}: {score:.3f}")


def predict_player_stats(request):
    """API endpoint to predict a player's next season stats."""
    player_name = request.GET.get("player", "").strip().lower()
    df_temp = df.copy()
    df_temp['PLAYER_NAME'] = df_temp['PLAYER_NAME'].str.strip().str.lower()

    best_match, score = process.extractOne(player_name, df_temp['PLAYER_NAME'].unique())
    if score < 80:
        return JsonResponse({"error": f"Player '{player_name.title()}' not found."}, status=400)

    player_data = df_temp[df_temp['PLAYER_NAME'] == best_match].sort_values(by='Year', ascending=False).iloc[0]

    input_limited = pd.DataFrame(
        [[player_data[f'Prev_{feat}'] for feat in LIMITED_FEATURES]],
        columns=[f'Prev_{feat}' for feat in LIMITED_FEATURES]
    )
    input_full = pd.DataFrame(
        [[player_data[f'Prev_{feat}'] for feat in FULL_FEATURES]],
        columns=[f'Prev_{feat}' for feat in FULL_FEATURES]
    )

    games_played = player_data.get("GP_x", 82)
    if games_played == 0:
        games_played = 82

    predictions = {}

    for target in TARGETS:
        pred_by_model = {}

        for model_name in ["linear", "random_forest", "neural_network"]:
            model = models[target][model_name]

            if model_name == "linear":
                input_scaled = scalers[target]["limited"].transform(input_limited)
            elif model_name == "random_forest":
                input_scaled = input_full  # no scaling
            else:
                input_scaled = scalers[target]["full"].transform(input_full)

            raw_pred = model.predict(input_scaled)[0]

            if target in ["FG_PCT_x", "FG3_PCT"]:
                pred = round(float(expit(raw_pred)), 3)
            elif target in ["PTS", "AST", "REB"]:
                pred = round(raw_pred / games_played, 2)
            else:
                pred = round(raw_pred, 3)


            if target in ["PTS", "AST", "REB"]:
                pred = round(pred / games_played, 2)
            else:
                pred = round(pred, 3)

            pred_by_model[model_name] = pred

        predictions[target] = pred_by_model

    return JsonResponse({
        "player": best_match.title(),
        "season": "2025-2026",
        "predictions": predictions
    })
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from thefuzz import process
import matplotlib.pyplot as plt
import os

data_folder = os.path.join(os.path.dirname(__file__), "data_files")
# Define the years of data and load them
years = list(range(2021, 2025))
df_list = []

for year in years:
    df_basic_path = os.path.join(data_folder, f"{year}_nba.csv")
    df_adv_path = os.path.join(data_folder, f"{year}_advanced.csv")
    
    df_basic = pd.read_csv(df_basic_path)
    df_basic["Year"] = year
    
    df_adv = pd.read_csv(df_adv_path)
    df_adv["Year"] = year
    df_merged = df_basic.merge(df_adv, on=["Player", "Year"], how="left")
    df_list.append(df_merged)

df = pd.concat(df_list, ignore_index=True)
df = df.sort_values(by=["Player", "Year"])

# Define features and target variables
features = [
    "Age_x", "Team_x", "Pos_x", "G_x", "GS_x", "MP_x", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
    "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST",
    "STL", "BLK", "TOV", "PF", "PTS", "PER", "TS%", "3PAr", "FTr", "ORB%", "DRB%",
    "TRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%", "OWS", "DWS", "WS", "WS/48",
    "BPM", "VORP"
]

targets = ["PTS", "AST", "TRB", "FG%", "3P%", "TS%"]

# Create previous season features
for feature in features:
    df[f'Prev_{feature}'] = df.groupby('Player')[feature].shift(1)

numeric_cols = df.select_dtypes(include=['number']).columns

df[numeric_cols] = df.groupby('Player')[numeric_cols].transform(lambda x: x.fillna(x.mean()))
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

df['Prev_Team_x'] = df.groupby('Player')['Team_x'].ffill()
df['Prev_Pos_x'] = df.groupby('Player')['Pos_x'].transform(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')

df.dropna(subset=targets, inplace=True)

X = df[[f'Prev_{feat}' for feat in features]]

y_dict = {target: df[target] for target in targets}  # Dictionary to hold separate target values

X_train, X_test, y_train, y_test = train_test_split(X, df[targets], test_size=0.2, random_state=42)

# Split each target separately into dictionary form
y_train_dict = {target: y_train[target] for target in targets}
y_test_dict = {target: y_test[target] for target in targets}


categorical_features = ['Prev_Team_x', 'Prev_Pos_x']
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Train separate models for each target variable
models = {}
for target in targets:
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train_dict[target])
    models[target] = model
    print(f"Model R² Score for {target}: {model.score(X_test, y_test_dict[target]):.3f}")

# Function to predict stats for next season
def predict_next_season(player_name, df, models):
    player_name = player_name.strip().lower()
    df_temp = df.copy()
    df_temp['Player'] = df_temp['Player'].str.strip().str.lower()

    unique_players = df_temp['Player'].unique()
    best_match, score = process.extractOne(player_name, unique_players)

    if score < 80:
        print(f"Player '{player_name.title()}' not found with high confidence!")
        return

    player_data = df_temp[df_temp['Player'] == best_match].sort_values(by='Year', ascending=False)

    if player_data.empty:
        print(f"Player '{best_match.title()}' not found in dataset!")
        return

    player_data = player_data.iloc[0]

    input_features = pd.DataFrame([[
        player_data[f'Prev_{feat}'] for feat in features
    ]], columns=[f'Prev_{feat}' for feat in features])

    input_features = input_features.reindex(columns=X_train.columns, fill_value=0)

    predicted_values = {target: models[target].predict(input_features)[0] for target in targets}

    print(f"\nProjected Stats for {best_match.title()} (Next Season):")
    for stat, value in predicted_values.items():
        print(f"  - {stat}: {value:.3f}")

# Feature importance visualization for a specific model
chosen_target = "PTS"  # Choose which target variable to display feature importance for
importances = models[chosen_target].named_steps['rf'].feature_importances_

# Get feature names after preprocessing
feature_names = numerical_features + list(models[chosen_target].named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Predict for a player
player_name = input("Enter the player's name: ")
predict_next_season(player_name, df, models)



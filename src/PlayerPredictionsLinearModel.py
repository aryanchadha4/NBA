import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from thefuzz import process

def load_data(years, base_filename, adv_filename):
    """Loads and merges base and advanced stats for given years."""
    dataframes = []
    for year in years:
        df_base = pd.read_csv(f"{year}_{base_filename}.csv")
        df_adv = pd.read_csv(f"{year}_{adv_filename}.csv")
        df_base["Year"] = year
        df_adv["Year"] = year
        df_merged = df_base.merge(df_adv, on=["Player", "Year"], how="left")
        dataframes.append(df_merged)
    return pd.concat(dataframes, ignore_index=True)

def preprocess_data(df, key_features):
    """Prepares dataset by sorting and creating previous season features."""
    df = df.sort_values(by=["Player", "Year"])
    for feature in key_features:
        df[f'Prev_{feature}'] = df.groupby('Player')[feature].shift(1)
    return df.dropna(subset=[f'Prev_{feat}' for feat in key_features] + target_vars)

def train_models(df, features, target_vars):
    """Trains separate linear models for each target variable."""
    models = {}
    scalers = {}

    X = df[features]
    for target in target_vars:
        y = df[target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Store model and scaler
        models[target] = model
        scalers[target] = scaler

        print(f"Model R² Score for {target}: {model.score(X_test_scaled, y_test):.3f}")

    return models, scalers

def predict_next_season(player_name, df, models, scalers, key_features, target_vars):
    """Predicts a player's next season stats using multiple models."""
    player_name = player_name.strip().lower()
    df_temp = df.copy()
    df_temp['Player'] = df_temp['Player'].str.strip().str.lower()
    
    best_match, score = process.extractOne(player_name, df_temp['Player'].unique())
    if score < 80:
        print(f"Player '{player_name.title()}' not found with high confidence!")
        return
    
    player_data = df_temp[df_temp['Player'] == best_match].sort_values(by='Year', ascending=False)
    if player_data.empty:
        print(f"Player '{best_match.title()}' not found in dataset!")
        return
    
    player_data = player_data.iloc[0]
    input_features = np.array([player_data[f'Prev_{feat}'] for feat in key_features]).reshape(1, -1)

    predicted_values = {}

    for target in target_vars:
        scaler = scalers[target]
        model = models[target]

        input_scaled = scaler.transform(input_features)
        predicted_stat = model.predict(input_scaled)[0]
        predicted_values[target] = predicted_stat
    
    print(f"\nProjected Stats for {best_match.title()} (Next Season):")
    for stat, value in predicted_values.items():
        print(f"  - {stat}: {value:.3f}")

# Define parameters
years = [2021, 2022, 2023, 2024, 2025]
base_filename = "nba"
adv_filename = "advanced"
key_features = ['PTS', 'AST', 'TRB', 'FG%', '3P%', 'TS%▼', '3PAr', 'BPM', 'Age_x', 'FGA', '3PA', 'USG%']
target_vars = ['PTS', 'AST', 'TRB', 'FG%', '3P%']

# Load and preprocess data
df = load_data(years, base_filename, adv_filename)
df = preprocess_data(df, key_features)

# Train separate models
features = [f'Prev_{feat}' for feat in key_features]
models, scalers = train_models(df, features, target_vars)

# Predict for a player
player_name = input("Enter the player's name: ")
predict_next_season(player_name, df, models, scalers, key_features, target_vars)





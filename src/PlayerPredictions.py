import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load CSVs
df_2021 = pd.read_csv("2023_nba.csv")
df_2021["Year"] = 2023

df_2022 = pd.read_csv("2024_nba.csv")
df_2022["Year"] = 2024

df_2023 = pd.read_csv("2025_nba.csv")
df_2023["Year"] = 2025

# Single DataFrame
df = pd.concat([df_2021, df_2022, df_2023], ignore_index=True)

# Sort by Player and Year for consistency
df = df.sort_values(by=["Player", "Year"])

# Numerical columns
numerical_features = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%',
                      '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB',
                      'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

# Compute previous season’s stats for each feature
for feature in numerical_features:
    df[f'Prev_{feature}'] = df.groupby('Player')[feature].shift(1)

# Compute league-wide average change per year
league_avg_changes = df.groupby("Year")[numerical_features].mean().pct_change()
df = df.merge(league_avg_changes, on="Year", suffixes=("", "_League_Change"))

# Drop rows with NaNs
df.dropna(inplace=True)

# Define Features and Target Variables
features = [f'Prev_{feat}' for feat in numerical_features] + [f"{feat}_League_Change" for feat in numerical_features]
target = ['PTS', 'AST', 'TRB']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Scale the data for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate the Model
print(f"Model R² Score: {model.score(X_test_scaled, y_test):.3f}")

def predict_next_season(player_name, df, model, scaler):
    # Normalize input player name
    player_name = player_name.strip().lower()  # Ensure lowercase and no spaces
    
    # Make a copy to avoid modifying the original DataFrame
    df_temp = df.copy()
    df_temp['Player'] = df_temp['Player'].str.strip().str.lower()  # Normalize player names in the dataset

    # Filter dataset for the player
    player_data = df_temp[df_temp['Player'] == player_name].sort_values(by='Year', ascending=False)

    # Check if player exists before accessing iloc[0]
    if player_data.empty:
        print(f"Player '{player_name.title()}' not found in dataset!")
        return

    player_data = player_data.iloc[0]  # Get latest season stats

    # Prepare input features: Use ALL features that were used in training
    input_features = np.array([
        player_data[f'Prev_{feat}'] for feat in numerical_features  # Previous stats
    ] + [
        player_data[f"{feat}_League_Change"] for feat in numerical_features  # League-wide trends
    ]).reshape(1, -1)

    # Scale input
    input_scaled = scaler.transform(input_features)

    # Predict next season's stats
    predicted_stats = model.predict(input_scaled)
    predicted_values = dict(zip(["PTS", "AST", "TRB"], predicted_stats[0]))

    print(f"\nProjected Stats for {player_name.title()} (Next Season):")
    for stat, value in predicted_values.items():
        print(f"  - {stat}: {value:.1f}")

# Example usage
predict_next_season("Trae Young", df, model, scaler)


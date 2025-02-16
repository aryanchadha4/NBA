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

df_adv_2021 = pd.read_csv("2023_advanced.csv")
df_adv_2021["Year"] = 2023
df_adv_2022 = pd.read_csv("2024_advanced.csv")
df_adv_2022["Year"] = 2024
df_adv_2023 = pd.read_csv("2025_advanced.csv")
df_adv_2023["Year"] = 2025

df_2021 = df_2021.merge(df_adv_2021, on=["Player", "Year"], how="left")
df_2022 = df_2022.merge(df_adv_2022, on=["Player", "Year"], how="left")
df_2023 = df_2023.merge(df_adv_2023, on=["Player", "Year"], how="left")

df = pd.concat([df_2021, df_2022, df_2023], ignore_index=True)

df = df.sort_values(by=["Player", "Year"])

print("Columns before dropping 'Awards':", df.columns)

df = df.drop(columns=["Awards_x"], errors="ignore")
df = df.drop(columns=["Awards_y"], errors="ignore")

print("Columns after dropping 'Awards':", df.columns)

print("Number of unique players:", df["Player"].nunique())

key_features = ['PTS', 'AST', 'TRB', 'FG%', '3P%', 'TS%▼', 'PER']

for feature in key_features:
    df[f'Prev_{feature}'] = df.groupby('Player')[feature].shift(1)


print(df.isna().sum())

features = [f'Prev_{feat}' for feat in key_features]
target = ['PTS', 'AST', 'TRB', 'FG%', '3P%']

df.dropna(subset=[f'Prev_{feat}' for feat in key_features] + target, inplace=True)

print("Number of unique players:", df["Player"].nunique())

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print(f"Model R² Score: {model.score(X_test_scaled, y_test):.3f}")

def predict_next_season(player_name, df, model, scaler):
    player_name = player_name.strip().lower()
    
    df_temp = df.copy()
    df_temp['Player'] = df_temp['Player'].str.strip().str.lower()

    player_data = df_temp[df_temp['Player'] == player_name].sort_values(by='Year', ascending=False)

    if player_data.empty:
        print(f"Player '{player_name.title()}' not found in dataset!")
        return

    player_data = player_data.iloc[0]

    input_features = np.array([
        player_data[f'Prev_{feat}'] for feat in key_features
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_features)

    predicted_stats = model.predict(input_scaled)
    predicted_values = dict(zip(["PTS", "AST", "TRB", "FG%", "3P%"], predicted_stats[0]))

    print(f"\nProjected Stats for {player_name.title()} (Next Season):")
    for stat, value in predicted_values.items():
        print(f"  - {stat}: {value:.3f}")

player_name = input("Enter the player's name: ")
predict_next_season(player_name, df, model, scaler)


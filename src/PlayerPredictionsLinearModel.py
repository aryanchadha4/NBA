import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

df_2019 = pd.read_csv("2021_nba.csv")
df_2019["Year"] = 2021

df_2020 = pd.read_csv("2022_nba.csv")
df_2020["Year"] = 2022

df_2021 = pd.read_csv("2023_nba.csv")
df_2021["Year"] = 2023

df_2022 = pd.read_csv("2024_nba.csv")
df_2022["Year"] = 2024

df_2023 = pd.read_csv("2025_nba.csv")
df_2023["Year"] = 2025


df_adv_2019 = pd.read_csv("2021_advanced.csv")
df_adv_2019["Year"] = 2021

df_adv_2020 = pd.read_csv("2022_advanced.csv")
df_adv_2020["Year"] = 2022

df_adv_2021 = pd.read_csv("2023_advanced.csv")
df_adv_2021["Year"] = 2023

df_adv_2022 = pd.read_csv("2024_advanced.csv")
df_adv_2022["Year"] = 2024

df_adv_2023 = pd.read_csv("2025_advanced.csv")
df_adv_2023["Year"] = 2025

df_2019 = df_2019.merge(df_adv_2019, on=["Player", "Year"], how="left")
df_2020 = df_2020.merge(df_adv_2020, on=["Player", "Year"], how="left")
df_2021 = df_2021.merge(df_adv_2021, on=["Player", "Year"], how="left")
df_2022 = df_2022.merge(df_adv_2022, on=["Player", "Year"], how="left")
df_2023 = df_2023.merge(df_adv_2023, on=["Player", "Year"], how="left")


df = pd.concat([df_2019, df_2020, df_2021, df_2022, df_2023], ignore_index=True)

df = df.sort_values(by=["Player", "Year"])


key_features = ['PTS', 'AST', 'TRB', 'FG%', '3P%', 'TS%▼', '3PAr', 'BPM', 'Age_x']

for feature in key_features:
    df[f'Prev_{feature}'] = df.groupby('Player')[feature].shift(1)


features = [f'Prev_{feat}' for feat in key_features]
target = ['PTS', 'AST', 'TRB', 'FG%', '3P%']

df.dropna(subset=[f'Prev_{feat}' for feat in key_features] + target, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print(f"Model R² Score: {model.score(X_test_scaled, y_test):.3f}")

from thefuzz import process

def predict_next_season(player_name, df, model, scaler):
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

    input_features = np.array([
        player_data[f'Prev_{feat}'] for feat in key_features
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_features)

    predicted_stats = model.predict(input_scaled)
    predicted_values = dict(zip(["PTS", "AST", "TRB", "FG%", "3P%"], predicted_stats[0]))

    print(f"\nProjected Stats for {best_match.title()} (Next Season):")
    for stat, value in predicted_values.items():
        print(f"  - {stat}: {value:.3f}")

player_name = input("Enter the player's name: ")
predict_next_season(player_name, df, model, scaler)



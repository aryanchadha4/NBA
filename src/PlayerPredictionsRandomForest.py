import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from thefuzz import process

years = list(range(2021, 2025))
df_list = []

for year in years:
    df_basic = pd.read_csv(f"{year}_nba.csv")
    df_basic["Year"] = year
    
    df_adv = pd.read_csv(f"{year}_advanced.csv")
    df_adv["Year"] = year
    
    df_merged = df_basic.merge(df_adv, on=["Player", "Year"], how="left")
    df_list.append(df_merged)

df = pd.concat(df_list, ignore_index=True)
df = df.sort_values(by=["Player", "Year"])

features = [
    "Age_x", "Team_x", "Pos_x", "G_x", "GS_x", "MP_x", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
    "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST",
    "STL", "BLK", "TOV", "PF", "PTS", "PER", "TS%", "3PAr", "FTr", "ORB%", "DRB%",
    "TRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%", "OWS", "DWS", "WS", "WS/48",
    "OBPM", "DBPM", "BPM", "VORP"
]

target = ["PTS", "AST", "TRB", "FG%", "3P%"]

for feature in features:
    df[f'Prev_{feature}'] = df.groupby('Player')[feature].shift(1)

numeric_cols = df.select_dtypes(include=['number']).columns

df[numeric_cols] = df.groupby('Player')[numeric_cols].transform(lambda x: x.fillna(x.mean()))

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

df['Prev_Team_x'] = df.groupby('Player')['Team_x'].ffill()

df['Prev_Pos_x'] = df.groupby('Player')['Pos_x'].transform(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')

# Verify if Jalen Johnson exists before dropping NaNs
print("Before dropping NaNs:")
print(df[df['Player'].str.contains('Jalen Johnson', case=False, na=False)])

df.dropna(subset=target, inplace=True)

# Check if Jalen Johnson still exists after dropping NaNs
print("After dropping NaNs:")
print(df[df['Player'].str.contains('Jalen Johnson', case=False, na=False)])


X = df[[f'Prev_{feat}' for feat in features]]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['Prev_Team_x', 'Prev_Pos_x']
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])

model_pipeline.fit(X_train, y_train)

print(f"Model R² Score: {model_pipeline.score(X_test, y_test):.3f}")

def predict_next_season(player_name, df, model_pipeline):
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

    predicted_stats = model_pipeline.predict(input_features)
    predicted_values = dict(zip(target, predicted_stats[0]))

    print(f"\nProjected Stats for {best_match.title()} (Next Season):")
    for stat, value in predicted_values.items():
        print(f"  - {stat}: {value:.3f}")

player_name = input("Enter the player's name: ")
predict_next_season(player_name, df, model_pipeline)


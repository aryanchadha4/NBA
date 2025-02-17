import pandas as pd
import numpy as np
from PlayerPredictionsLinearModel import train_linear_model, predict_linear
from PlayerPredictionsRandomForest import train_random_forest, predict_random_forest
from thefuzz import process
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
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

# Create previous season features
for feature in features:
    df[f'Prev_{feature}'] = df.groupby('Player')[feature].shift(1)

# Fill missing values
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df.groupby('Player')[numeric_cols].transform(lambda x: x.fillna(x.mean()))
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

df['Prev_Team_x'] = df.groupby('Player')['Team_x'].ffill()
df['Prev_Pos_x'] = df.groupby('Player')['Pos_x'].transform(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')

# Drop rows with missing target values
df.dropna(subset=target, inplace=True)

# Prepare training data
X = df[[f'Prev_{feat}' for feat in features]]
y = df[target]

# Split data into training, testing, and validation sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Train both models
rf_model = train_random_forest(X_train, y_train)
lr_model = train_linear_model(X_train, y_train)

print(f"Random Forest R² Score: {rf_model.score(X_test, y_test):.3f}")
print(f"Linear Regression R² Score: {lr_model.score(X_test, y_test):.3f}")

# Optimize blending weights
def optimize_blend_weights(rf_model, lr_model, X_val, y_val):
    """Find the best weights for blending models to maximize R² score."""
    best_r2 = -np.inf
    best_w_rf = 0.5  # Default: 50% RF, 50% Linear
    best_w_lr = 0.5

    for w_rf in np.linspace(0, 1, 21):  # Try weights from 0% RF to 100% RF in 5% steps
        w_lr = 1 - w_rf  # Complementary weight

        rf_pred = rf_model.predict(X_val)
        lr_pred = lr_model.predict(X_val)

        blended_pred = (w_rf * rf_pred) + (w_lr * lr_pred)  # Weighted average

        r2 = r2_score(y_val, blended_pred)  # Compute R²

        if r2 > best_r2:
            best_r2 = r2
            best_w_rf = w_rf
            best_w_lr = w_lr

    print(f"Optimal Weights → RF: {best_w_rf:.2f}, LR: {best_w_lr:.2f} | Max R²: {best_r2:.3f}")
    return best_w_rf, best_w_lr

# Find best blend weights
w_rf, w_lr = optimize_blend_weights(rf_model, lr_model, X_val, y_val)

def predict_next_season(player_name, df, rf_model, lr_model, key_features, w_rf, w_lr):
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

    # 🔥 Compute career averages for missing numeric features
    career_averages = df[df['Player'] == best_match].mean(numeric_only=True)

    # 🔥 Compute most frequent categorical values (handle missing modes)
    mode_df = df[df['Player'] == best_match].mode()
    if mode_df.empty:
        most_common_values = {}
    else:
        most_common_values = mode_df.iloc[0].to_dict()  # Convert row to dictionary

    # 🔥 Create input DataFrame with all expected columns
    input_features = pd.DataFrame([[
        player_data.get(f'Prev_{feat}', career_averages.get(f'Prev_{feat}', 0))  # Numeric fallback to career avg
        if f'Prev_{feat}' in career_averages else most_common_values.get(f'Prev_{feat}', 'Unknown')  # Categorical fallback
        for feat in key_features
    ]], columns=[f'Prev_{feat}' for feat in key_features])

    # 🔥 Ensure input features match training model features exactly
    expected_features = rf_model.named_steps['preprocessor'].get_feature_names_out()

    # Create a DataFrame with all expected features, filling missing ones
    input_features = input_features.reindex(columns=expected_features, fill_value=0)

    # 🔥 Ensure column ordering matches what Random Forest expects
    input_features = input_features[expected_features]

    # Get predictions from both models
    rf_pred = rf_model.predict(input_features)
    lr_pred = lr_model.predict(input_features)

    # Blend predictions using optimized weights
    blended_pred = (w_rf * rf_pred) + (w_lr * lr_pred)

    predicted_values = dict(zip(target, blended_pred[0]))

    print(f"\nProjected Stats for {best_match.title()} (Blended Model):")
    for stat, value in predicted_values.items():
        print(f"  - {stat}: {value:.3f}")



# Run prediction
player_name = input("Enter the player's name: ")
predict_next_season(player_name, df, rf_model, lr_model, features, w_rf, w_lr)



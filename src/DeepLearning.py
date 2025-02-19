import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from thefuzz import process
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Define the years of data and load them
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

# Define features and target variables
features = [
    "Age_x", "G_x", "GS_x", "MP_x", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
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

df.dropna(subset=targets, inplace=True)

X = df[[f'Prev_{feat}' for feat in features]]
y = df[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize targets
y_train_normalized = {}
y_test_normalized = {}
y_mean = {}
y_std = {}

for target in targets:
    y_mean[target] = y_train[target].mean()
    y_std[target] = y_train[target].std()
    y_train_normalized[target] = (y_train[target] - y_mean[target]) / y_std[target]
    y_test_normalized[target] = (y_test[target] - y_mean[target]) / y_std[target]

# Preprocess data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define learning rate schedule
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=1000,
    decay_rate=0.85
)

# Train separate models for each statistic
models = {}

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

for target in targets:
    print(f"\nTraining model for {target}...")
    model = build_model()
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    model.fit(X_train, y_train_normalized[target], epochs=200, batch_size=8, validation_data=(X_test, y_test_normalized[target]), verbose=1, callbacks=[early_stop])
    models[target] = model

# Evaluate models
def evaluate_models(models, X_test, y_test):
    for target in targets:
        print(f"\nEvaluating model for {target}...")
        y_test_target = y_test[target]
        y_pred = models[target].predict(X_test).flatten()
        y_pred = (y_pred * y_std[target]) + y_mean[target]
        r2 = r2_score(y_test_target, y_pred)
        mae = mean_absolute_error(y_test_target, y_pred)
        mse = mean_squared_error(y_test_target, y_pred)
        rmse = np.sqrt(mse)
        print(f"  - R² Score: {r2:.3f}")
        print(f"  - Mean Absolute Error (MAE): {mae:.3f}")
        print(f"  - Mean Squared Error (MSE): {mse:.3f}")
        print(f"  - Root Mean Squared Error (RMSE): {rmse:.3f}")

evaluate_models(models, X_test, y_test)

# Predict next season stats
def predict_next_season(player_name, df, models, scaler):
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
    input_features = np.array([[player_data[f'Prev_{feat}'] for feat in features]])
    input_features = scaler.transform(input_features)
    print(f"\nProjected Stats for {best_match.title()} (Next Season):")
    for target in targets:
        predicted_value = models[target].predict(input_features)[0][0]
        predicted_value = (predicted_value * y_std[target]) + y_mean[target]
        print(f"  - {target}: {predicted_value:.3f}")

player_name = input("Enter the player's name: ")
predict_next_season(player_name, df, models, scaler)












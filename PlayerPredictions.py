import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class PlayerPrediction:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.data = self.load_and_merge_data()

    def add_yoy_change(self):
        excluded_columns = [
            "PLAYER", "TEAM", "AGE", "GP", "W", "L", "FGM", "FGA", "3PM", "3PA", "FTM", "FTA", 
            "PF", "FP", "DD2", "TD3"
        ]

        stat_columns = [col for col in predictor.data.columns if col not in excluded_columns]
        yoy_changes = []

        for stat in stat_columns:
            stat_season_cols = [col for col in self.data.columns if col.startswith(stat)]
            # Ensure columns are numeric before processing
            numeric_cols = []
            for col in stat_season_cols:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    numeric_cols.append(col)
                else:
                    # Attempt to convert to numeric, coerce invalid values to NaN
                    self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
                    numeric_cols.append(col)
        
        for i in range(1, len(stat_season_cols)):
            self.data[f"{stat}_YoY_Change_{i}"] = (
                self.data[stat_season_cols[i]] - self.data[stat_season_cols[i - 1]]
            )
        
        yoy_change_cols = [col for col in self.data.columns if col.startswith(f"{stat}_YoY_Change")]
        
        self.data[f"{stat}_Avg_YoY_Change"] = self.data[yoy_change_cols].mean(1, True)

    def load_and_merge_data(self):
        df = []
        for idx, file_path in enumerate(self.file_paths):
            season_data = pd.read_csv(file_path)
            season_data['Season'] = f"Year{idx + 1}"
            df.append(season_data)

        merged_data = df[0]
        for idx, next_df in enumerate(df[1:], start=1):
            merged_data = pd.merge(merged_data, next_df, on=["PLAYER", "TEAM"], suffixes=(f"_{idx-1}", f"_{idx}"), how="outer")
        
        return merged_data

    def preprocess_data(self, target_column, drop_columns=None):
        self.add_yoy_change()
        data = self.data.dropna()

        default_drop = ["PLAYER", "TEAM", "Season", target_column]
        if drop_columns:
            default_drop.extend(drop_columns)

        features = [col for col in data.columns if col not in default_drop]

        X = data[features]
        y = data[target_column]

        X_mean = X.mean()
        X_std = X.std()
        y_mean = y.mean()
        y_std = y.std()

        X_normalized = (X - X_mean) / X_std
        y_normalized = (y - y_mean) / y_std

        self.feature_columns = features

        return X_normalized, y_normalized, X_mean, X_std, y_mean, y_std

    def linear_train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MSE = {mse}, R2 = {r2}")

        return model
    
    def random_forest_train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MSE = {mse}, R2 = {r2}")

        return model
    def neural_network_train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)), 
            Dense(64, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])

        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MSE = {mse}, R2 = {r2}")

        return model
    

    def predict(self, model, input_data):
        return model.predict(input_data)

    def predict_for_player(self, model, player_name, x_mean, x_std, y_mean, y_std):
        player_data = self.data[self.data["PLAYER"].str.lower() == player_name.lower()]

        if not player_data.empty:
            player_features = player_data[self.feature_columns]
            player_features = (player_features - x_mean) / x_std

            prediction_normalized = model.predict(player_features)
            prediction_denormalized = prediction_normalized[0] * y_std + y_mean
            print(f"Predicted stat for {player_name} in next season: {prediction_denormalized}")
            return prediction_denormalized
        else:
            return None


if __name__ == "__main__":
    file_paths = [
        "nba_all_player_stats_2025.csv",
        "nba_all_player_stats_2024.csv",
        "nba_all_player_stats_2023.csv"
    ]

    predictor = PlayerPrediction(file_paths)

    player_name = "LeBron James"
    category = "PTS_0"

    X, y, X_mean, X_std, y_mean, y_std = predictor.preprocess_data(category)

    model = predictor.linear_train_model(X, y)
    model2 = predictor.random_forest_train_model(X, y)

    prediction = predictor.predict_for_player(model, player_name, X_mean, X_std, y_mean, y_std)
    prediction2 = predictor.predict_for_player(model2, player_name, X_mean, X_std, y_mean, y_std)

    print(prediction)
    print(prediction2)



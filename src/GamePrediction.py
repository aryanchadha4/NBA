import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ----------------------------
# Step 1: Load and Clean Game Logs
# ----------------------------
file_paths = [
    "NBA_Game_Outcomes_October.csv",
    "NBA_Game_Outcomes_November.csv",
    "NBA_Game_Outcomes_December.csv",
    "NBA_Game_Outcomes_January.csv",
]

# Load and concatenate all game logs
game_logs = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)

# Rename columns explicitly based on your dataset
game_logs = game_logs.rename(columns={
    "Visitor/Neutral": "visitor_team",
    "PTS": "visitor_pts",  
    "Home/Neutral": "home_team",
    "PTS.1": "home_pts"
})

# Convert Date column to datetime format
game_logs["Date"] = pd.to_datetime(game_logs["Date"], format="%a, %b %d, %Y")

# Drop unnecessary columns
columns_to_drop = ["Unnamed: 6", "Unnamed: 7", "Attend.", "LOG", "Arena", "Notes"]
game_logs = game_logs.drop(columns=[col for col in columns_to_drop if col in game_logs.columns])

# Create 'Winner' column (1 if home team wins, 0 if away team wins)
game_logs["winner"] = (game_logs["home_pts"] > game_logs["visitor_pts"]).astype(int)

# Create 'Home Indicator' column (1 for home team, 0 for away team)
game_logs["Home_Indicator"] = 1  # Home team always 1

# ----------------------------
# Step 2: Load and Clean Team Stats
# ----------------------------
team_stats = pd.read_csv("NBA_Teams_Stats.csv")

# Drop unnecessary unnamed columns if they exist
team_stats = team_stats.drop(columns=["Unnamed: 17", "Unnamed: 22", "Unnamed: 27"], errors="ignore")

# Rename duplicate columns (defensive stats)
team_stats = team_stats.rename(columns={
    "eFG%.1": "Def_eFG%",
    "TOV%.1": "Def_TOV%",
    "FT/FGA.1": "Def_FT/FGA",
    "DRB%": "Def_DRB%"
})

# Ensure there are no extra spaces in team names
team_stats["Team"] = team_stats["Team"].str.strip()
game_logs["home_team"] = game_logs["home_team"].str.strip()
game_logs["visitor_team"] = game_logs["visitor_team"].str.strip()

# Keep only the selected features
selected_features = [
    "Team", "ORtg", "DRtg", "NRtg", "Pace", "FTr", "3PAr", "TS%", "eFG%", "TOV%", "ORB%", 
    "FT/FGA", "Def_eFG%", "Def_TOV%", "Def_DRB%", "Def_FT/FGA"
]
team_stats = team_stats[selected_features]

# ----------------------------
# Step 3: Merge Team Stats with Game Logs
# ----------------------------
# Merge home team stats
game_logs = game_logs.merge(team_stats, left_on="home_team", right_on="Team", how="left")
game_logs = game_logs.rename(columns={col: col + "_home" for col in team_stats.columns if col != "Team"})

# Merge visitor team stats
game_logs = game_logs.merge(team_stats, left_on="visitor_team", right_on="Team", how="left")
game_logs = game_logs.rename(columns={col: col + "_away" for col in team_stats.columns if col != "Team"})

# Drop redundant 'Team' columns
game_logs = game_logs.drop(columns=["Team_home", "Team_away"], errors="ignore")

# Fill missing values with 0
game_logs.fillna(0, inplace=True)

# Verify that ORtg_home exists
print("\nGame Logs Columns After Merging:\n", game_logs.columns)

# ----------------------------
# Step 4: Feature Engineering
# ----------------------------
# Compute stat differentials (home - away)
for stat in ["ORtg", "DRtg", "NRtg", "Pace", "FTr", "3PAr", "TS%", "eFG%", "TOV%", "ORB%", 
             "FT/FGA", "Def_eFG%", "Def_TOV%", "Def_DRB%", "Def_FT/FGA"]:
    if f"{stat}_home" in game_logs.columns and f"{stat}_away" in game_logs.columns:
        game_logs[f"{stat}_diff"] = game_logs[f"{stat}_home"] - game_logs[f"{stat}_away"]
    else:
        print(f"Warning: {stat}_home or {stat}_away missing!")

# Define features including Home Indicator
features = [col for col in game_logs.columns if "_diff" in col] + ["Home_Indicator"]
target = "winner"

# ----------------------------
# Step 5: Train-Test Split and Model Training
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(game_logs[features], game_logs[target], test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))


def predict_game(team1, team2, model, team_stats):
    """
    Predicts the winner between two teams based on the trained model.
    
    Parameters:
    - team1: Home team (string)
    - team2: Away team (string)
    - model: Trained logistic regression model
    - team_stats: DataFrame containing team statistics
    
    Returns:
    - Predicted winner (team name)
    """
    # Ensure team names are properly formatted
    team1 = team1.strip()
    team2 = team2.strip()

    # Get team stats for home and away teams
    team1_stats = team_stats[team_stats["Team"] == team1]
    team2_stats = team_stats[team_stats["Team"] == team2]

    # Check if both teams exist in the stats dataset
    if team1_stats.empty or team2_stats.empty:
        print("Error: One or both teams not found in the dataset.")
        return None

    # Compute differentials
    match_features = {}
    for stat in ["ORtg", "DRtg", "NRtg", "Pace", "FTr", "3PAr", "TS%", "eFG%", "TOV%", "ORB%", 
                 "FT/FGA", "Def_eFG%", "Def_TOV%", "Def_DRB%", "Def_FT/FGA"]:
        match_features[f"{stat}_diff"] = team1_stats[stat].values[0] - team2_stats[stat].values[0]

    # Add home indicator
    match_features["Home_Indicator"] = 1  # Since team1 is considered the home team

    # Convert to DataFrame
    match_df = pd.DataFrame([match_features])

    # Ensure feature columns match training data
    missing_cols = set(features) - set(match_df.columns)
    for col in missing_cols:
        match_df[col] = 0  # Add missing columns with default value 0

    # Predict winner
    prediction = model.predict(match_df)[0]

    # Output the predicted winner
    winner = team1 if prediction == 1 else team2
    print(f"Predicted Winner: {winner}")

    return winner


# ----------------------------
# Step 7: User Input for Team Matchups
# ----------------------------
team1 = input("Enter the home team: ")
team2 = input("Enter the away team: ")

predict_game(team1, team2, model, team_stats)


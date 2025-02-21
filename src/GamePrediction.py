import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

file_paths = [
    "NBA_Game_Outcomes_October.csv",
    "NBA_Game_Outcomes_November.csv",
    "NBA_Game_Outcomes_December.csv",
    "NBA_Game_Outcomes_January.csv",
]

game_logs = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)

game_logs = game_logs.rename(columns={
    "Visitor/Neutral": "visitor_team",
    "PTS": "visitor_pts",  
    "Home/Neutral": "home_team",
    "PTS.1": "home_pts"
})

game_logs["Date"] = pd.to_datetime(game_logs["Date"], format="%a, %b %d, %Y")

columns_to_drop = ["Unnamed: 6", "Unnamed: 7", "Attend.", "LOG", "Arena", "Notes"]
game_logs = game_logs.drop(columns=[col for col in columns_to_drop if col in game_logs.columns])

game_logs["winner"] = (game_logs["home_pts"] > game_logs["visitor_pts"]).astype(int)

game_logs["Home_Indicator"] = 1 

team_stats = pd.read_csv("NBA_Teams_Stats.csv")

team_stats = team_stats.drop(columns=["Unnamed: 17", "Unnamed: 22", "Unnamed: 27"], errors="ignore")

team_stats = team_stats.rename(columns={
    "eFG%.1": "Def_eFG%",
    "TOV%.1": "Def_TOV%",
    "FT/FGA.1": "Def_FT/FGA",
    "DRB%": "Def_DRB%"
})

team_stats["Team"] = team_stats["Team"].str.strip()
game_logs["home_team"] = game_logs["home_team"].str.strip()
game_logs["visitor_team"] = game_logs["visitor_team"].str.strip()

selected_features = [
    "Team", "ORtg", "DRtg", "NRtg", "Pace", "FTr", "3PAr", "TS%", "eFG%", "TOV%", "ORB%", 
    "FT/FGA", "Def_eFG%", "Def_TOV%", "Def_DRB%", "Def_FT/FGA"
]
team_stats = team_stats[selected_features]

game_logs = game_logs.merge(team_stats, left_on="home_team", right_on="Team", how="left")
game_logs = game_logs.rename(columns={col: col + "_home" for col in team_stats.columns if col != "Team"})

game_logs = game_logs.merge(team_stats, left_on="visitor_team", right_on="Team", how="left")
game_logs = game_logs.rename(columns={col: col + "_away" for col in team_stats.columns if col != "Team"})

game_logs = game_logs.drop(columns=["Team_home", "Team_away"], errors="ignore")

game_logs.fillna(0, inplace=True)

for stat in ["ORtg", "DRtg", "NRtg", "Pace", "FTr", "3PAr", "TS%", "eFG%", "TOV%", "ORB%", 
             "FT/FGA", "Def_eFG%", "Def_TOV%", "Def_DRB%", "Def_FT/FGA"]:
    if f"{stat}_home" in game_logs.columns and f"{stat}_away" in game_logs.columns:
        game_logs[f"{stat}_diff"] = game_logs[f"{stat}_home"] - game_logs[f"{stat}_away"]

features = [col for col in game_logs.columns if "_diff" in col] + ["Home_Indicator"]
target = "winner"

X_train, X_test, y_train, y_test = train_test_split(game_logs[features], game_logs[target], test_size=0.2, random_state=42)

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)
y_pred_rf = random_forest_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


def predict_game(team1, team2, logistic_model, rf_model, team_stats):
    team1 = team1.strip()
    team2 = team2.strip()

    team1_stats = team_stats[team_stats["Team"] == team1]
    team2_stats = team_stats[team_stats["Team"] == team2]

    if team1_stats.empty or team2_stats.empty:
        print("Error: One or both teams not found in the dataset.")
        return None

    match_features = {}
    for stat in ["ORtg", "DRtg", "NRtg", "Pace", "FTr", "3PAr", "TS%", "eFG%", "TOV%", "ORB%", 
                 "FT/FGA", "Def_eFG%", "Def_TOV%", "Def_DRB%", "Def_FT/FGA"]:
        match_features[f"{stat}_diff"] = team1_stats[stat].values[0] - team2_stats[stat].values[0]

    match_features["Home_Indicator"] = 1

    match_df = pd.DataFrame([match_features])

    missing_cols = set(features) - set(match_df.columns)
    for col in missing_cols:
        match_df[col] = 0  

    prediction_logistic = logistic_model.predict(match_df)[0]
    prediction_rf = rf_model.predict(match_df)[0]

    winner_logistic = team1 if prediction_logistic == 1 else team2
    winner_rf = team1 if prediction_rf == 1 else team2

    print(f"\n**Prediction Results:**")
    print(f"Logistic Regression Winner: {winner_logistic}")
    print(f"*Random Forest Winner: {winner_rf}\n")

    return winner_logistic, winner_rf


team1 = input("Enter the home team: ")
team2 = input("Enter the away team: ")

predict_game(team1, team2, logistic_model, random_forest_model, team_stats)



from django.urls import path
from nba.views import game_predictor, player_predictor, general

urlpatterns = [
    path('', general.home, name='home'),
    path('game/', game_predictor.predict_game, name='predict_game'),
    path("player-prediction/", player_predictor.predict_player_stats, name="predict_player_stats"),
    path("api/game-prediction/", game_predictor.api_game_prediction, name="api_game_prediction"),
]

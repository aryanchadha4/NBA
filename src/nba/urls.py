from django.urls import path
from .views import predict_game, predict_player_stats, home

urlpatterns = [
    path("api/game-prediction/", predict_game),
    path("api/player-prediction/", predict_player_stats),
    path('', home),
]
